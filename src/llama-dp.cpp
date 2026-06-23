// src/llama-dp.cpp — see llama-dp.h for the design overview.

#include "llama-dp.h"
#include "llama-impl.h"    // LLAMA_LOG_*
#include "llama-memory.h"  // call llama_memory_i methods directly when fanning memory ops
#include "llama-context.h" // call llama_context methods directly (bypass the C-API registry)

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// side state attached to a lead (replica-0) handle
// ---------------------------------------------------------------------------

struct llama_dp_model {
    int                        n = 0;
    std::vector<llama_model *> replicas; // replicas[0] is the handle returned to the caller
};

// N replica contexts + a worker pool (one thread per replica). The hot path shards a batch's
// sequences across the replicas and runs the per-replica decodes concurrently. One context is
// only ever touched by its own worker thread, so no locking is needed around a llama_context.
struct llama_dp_context {
    int                          n = 0;
    int32_t                      n_kv_max = 0;
    int32_t                      n_vocab  = 0;
    std::vector<llama_context *> ctxs;     // ctxs[0] is the handle; owned
    std::vector<llama_memory_t>  mems;     // ctxs[r]'s memory (mems[0] is the memory handle)
    std::vector<llama_batch>     scratch;  // per-replica decode scratch (owned)
    std::vector<int8_t>          ret;      // per-replica decode return code

    // per-decode output routing (for llama_get_logits[_ith]); indexed by global batch position.
    std::vector<int>             route_r;
    std::vector<int>             route_li;
    std::vector<int8_t>          route_out;
    std::vector<float>           logits_buf;

    // --- worker pool ---
    std::vector<std::thread>     workers;
    std::function<void(int)>     job;
    std::mutex                   mtx;
    std::condition_variable      cv_start;
    std::condition_variable      cv_done;
    uint64_t                     gen   = 0;
    std::vector<uint64_t>        seen;
    int                          ndone = 0;
    bool                         stop  = false;

    void run_on_all(const std::function<void(int)> & fn) {
        {
            std::unique_lock<std::mutex> lk(mtx);
            job   = fn;
            ndone = 0;
            ++gen;
        }
        cv_start.notify_all();
        std::unique_lock<std::mutex> lk(mtx);
        cv_done.wait(lk, [this] { return ndone == n; });
    }

    void worker_loop(int r) {
        for (;;) {
            std::function<void(int)> myjob;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv_start.wait(lk, [this, r] { return stop || gen > seen[r]; });
                if (stop) {
                    return;
                }
                seen[r] = gen;
                myjob   = job;
            }
            // failure isolation: a throw on one replica must not wedge the barrier.
            try {
                myjob(r);
            } catch (const std::exception & e) {
                LLAMA_LOG_ERROR("llama_dp: replica %d job threw: %s\n", r, e.what());
            } catch (...) {
                LLAMA_LOG_ERROR("llama_dp: replica %d job threw (unknown)\n", r);
            }
            {
                std::unique_lock<std::mutex> lk(mtx);
                ++ndone;
                if (ndone == n) {
                    cv_done.notify_one();
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// registries: lead (replica-0) handle -> side state
// ---------------------------------------------------------------------------

namespace {
    std::mutex g_mtx;
    std::unordered_map<const void *, llama_dp_model *>   g_models; // model0 -> dp_model
    std::unordered_map<const void *, llama_dp_context *> g_ctxs;   // ctx0   -> dp_context
    std::unordered_map<const void *, llama_dp_context *> g_mems;   // mem0   -> dp_context

    template <class Map>
    typename Map::mapped_type lookup(Map & m, const void * p) {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto it = m.find(p);
        return it == m.end() ? nullptr : it->second;
    }
}

bool llama_dp_enabled(const llama_model_params & params) { return params.n_data_parallel > 1; }

bool llama_dp_is_model(const llama_model * m) { std::lock_guard<std::mutex> lk(g_mtx); return g_models.count(m); }
bool llama_dp_is_ctx  (const llama_context * c) { std::lock_guard<std::mutex> lk(g_mtx); return g_ctxs.count(c); }
bool llama_dp_is_mem  (const llama_memory_t m)  { std::lock_guard<std::mutex> lk(g_mtx); return g_mems.count(m); }

// ---------------------------------------------------------------------------
// model side
// ---------------------------------------------------------------------------

llama_model * llama_dp_model_load(const char * path, llama_model_params params) {
    const int n = params.n_data_parallel;

    auto * dp = new llama_dp_model();
    dp->n = n;
    dp->replicas.reserve(n);

    for (int r = 0; r < n; ++r) {
        llama_model_params mp = params;
        mp.n_data_parallel = 0;                    // replica load takes the STOCK path (no recursion)
        mp.split_mode      = LLAMA_SPLIT_MODE_NONE; // one whole replica per device
        mp.main_gpu        = r;                     // pin replica r to device r
        mp.devices         = nullptr;               // main_gpu selects among all visible devices
        mp.tensor_split    = nullptr;

        llama_model * m = llama_model_load_from_file(path, mp);
        if (!m) {
            LLAMA_LOG_ERROR("llama_dp: failed to load replica %d/%d on device %d\n", r, n, r);
            for (llama_model * prev : dp->replicas) {
                llama_model_free(prev);
            }
            delete dp;
            return nullptr;
        }
        dp->replicas.push_back(m);
    }

    LLAMA_LOG_INFO("llama_dp: loaded %d data-parallel replicas (one per device 0..%d)\n", n, n - 1);

    // the lead replica's real model is the handle; introspection runs natively on it.
    llama_model * handle = dp->replicas[0];
    { std::lock_guard<std::mutex> lk(g_mtx); g_models[handle] = dp; }
    return handle;
}

void llama_dp_model_free(llama_model * model) {
    llama_dp_model * dp = lookup(g_models, model);
    if (!dp) {
        return;
    }
    { std::lock_guard<std::mutex> lk(g_mtx); g_models.erase(model); } // unregister first => stock free below
    for (llama_model * m : dp->replicas) {
        llama_model_free(m);
    }
    delete dp;
}

// ---------------------------------------------------------------------------
// context side
// ---------------------------------------------------------------------------

llama_context * llama_dp_init(llama_model * model, llama_context_params params) {
    llama_dp_model * dpm = lookup(g_models, model);
    if (!dpm) {
        return nullptr;
    }
    const int n = dpm->n;

    auto * dp = new llama_dp_context();
    dp->n = n;
    dp->ctxs.reserve(n);
    dp->mems.reserve(n);

    // unregister the lead model so llama_init_from_model(replicas[0]) takes the stock path.
    { std::lock_guard<std::mutex> lk(g_mtx); g_models.erase(model); }
    bool ok = true;
    for (int r = 0; r < n; ++r) {
        llama_context * c = llama_init_from_model(dpm->replicas[r], params);
        if (!c) {
            LLAMA_LOG_ERROR("llama_dp: failed to create context for replica %d/%d\n", r, n);
            ok = false;
            break;
        }
        dp->ctxs.push_back(c);
        dp->mems.push_back(llama_get_memory(c));
    }
    { std::lock_guard<std::mutex> lk(g_mtx); g_models[model] = dpm; } // restore

    if (!ok) {
        for (llama_context * c : dp->ctxs) {
            llama_free(c);
        }
        delete dp;
        return nullptr;
    }

    dp->n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(dpm->replicas[0]));
    dp->n_kv_max = (int32_t) llama_n_ctx(dp->ctxs[0]);

    // per-replica scratch batch (single seq id per token, which is what the bench tools build).
    dp->scratch.resize(n);
    for (int r = 0; r < n; ++r) {
        dp->scratch[r] = llama_batch_init(dp->n_kv_max, 0, 1);
    }
    dp->ret.assign(n, 0);

    // start the worker pool: one persistent thread per replica.
    dp->seen.assign(n, 0);
    dp->workers.reserve(n);
    for (int r = 0; r < n; ++r) {
        dp->workers.emplace_back([dp, r] { dp->worker_loop(r); });
    }

    llama_context * handle = dp->ctxs[0];
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        g_ctxs[handle]      = dp;
        g_mems[dp->mems[0]] = dp;
    }
    return handle;
}

int32_t llama_dp_decode(llama_context * ctx, llama_batch batch) {
    llama_dp_context * dp = lookup(g_ctxs, ctx);
    const int n = dp->n;

    // shard the batch by sequence id: sequence j -> replica (j % n), local seq id (j / n).
    // record where each global token landed so llama_get_logits[_ith] can read it back.
    for (int r = 0; r < n; ++r) {
        dp->scratch[r].n_tokens = 0;
    }
    dp->route_r.assign(batch.n_tokens, 0);
    dp->route_li.assign(batch.n_tokens, 0);
    dp->route_out.assign(batch.n_tokens, 0);
    for (int i = 0; i < batch.n_tokens; ++i) {
        const llama_seq_id seq = (batch.seq_id && batch.n_seq_id && batch.n_seq_id[i] > 0)
                                     ? batch.seq_id[i][0] : 0;
        const int          r   = (int) (((seq % n) + n) % n);
        const llama_seq_id loc = seq / n;

        llama_batch & s = dp->scratch[r];
        const int k = s.n_tokens;
        s.token   [k]    = batch.token ? batch.token[i] : 0;
        s.pos     [k]    = batch.pos   ? batch.pos[i]   : 0;
        s.n_seq_id[k]    = 1;
        s.seq_id  [k][0] = loc;
        s.logits  [k]    = batch.logits ? batch.logits[i] : 0;
        s.n_tokens++;

        dp->route_r  [i] = r;
        dp->route_li [i] = k;
        dp->route_out[i] = batch.logits ? batch.logits[i] : 0;
    }

    std::fill(dp->ret.begin(), dp->ret.end(), (int8_t) 0);
    dp->run_on_all([dp](int r) {
        // call the context method directly: dp->ctxs[0] is a registered handle, so going through
        // the public llama_decode() would re-enter this DP layer and deadlock the worker pool.
        if (dp->scratch[r].n_tokens > 0) {
            dp->ret[r] = (int8_t) dp->ctxs[r]->decode(dp->scratch[r]);
        }
    });

    for (int r = 0; r < n; ++r) {
        if (dp->ret[r] != 0) {
            return dp->ret[r];
        }
    }
    return 0;
}

void llama_dp_synchronize(llama_context * ctx) {
    llama_dp_context * dp = lookup(g_ctxs, ctx);
    // method, not llama_synchronize(): ctxs[0] is registered and would re-enter this layer.
    dp->run_on_all([dp](int r) { dp->ctxs[r]->synchronize(); });
}

// read one token's logits from its replica, mirroring llama_get_logits_ith() but via the context
// methods directly (the C wrapper would re-enter this layer for the registered ctxs[0] handle).
static float * dp_logits_ith(llama_context * c, int32_t li) {
    c->synchronize();
    float * res = c->get_sampled_logits_ith(li);
    if (!res) {
        res = c->get_logits_ith(li);
    }
    return res;
}

float * llama_dp_get_logits_ith(llama_context * ctx, int32_t i) {
    llama_dp_context * dp = lookup(g_ctxs, ctx);
    if (i < 0 || i >= (int) dp->route_r.size()) {
        return nullptr;
    }
    // the whole sequence that owns token i lives on this replica and is contiguous there, so the
    // caller's contiguous read of the following rows stays valid.
    return dp_logits_ith(dp->ctxs[dp->route_r[i]], dp->route_li[i]);
}

float * llama_dp_get_logits(llama_context * ctx) {
    llama_dp_context * dp = lookup(g_ctxs, ctx);
    // rebuild the requested-output rows in global batch order (what llama_get_logits returns).
    // only reached when n_ctx > n_batch, which forces n_seq == 1 (all tokens on replica 0).
    dp->logits_buf.clear();
    for (size_t i = 0; i < dp->route_r.size(); ++i) {
        if (!dp->route_out[i]) {
            continue;
        }
        const float * row = dp_logits_ith(dp->ctxs[dp->route_r[i]], dp->route_li[i]);
        if (row) {
            dp->logits_buf.insert(dp->logits_buf.end(), row, row + dp->n_vocab);
        }
    }
    return dp->logits_buf.data();
}

void llama_dp_free(llama_context * ctx) {
    llama_dp_context * dp = lookup(g_ctxs, ctx);
    if (!dp) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(g_mtx); // unregister first => stock free below
        g_ctxs.erase(ctx);
        if (!dp->mems.empty()) {
            g_mems.erase(dp->mems[0]);
        }
    }

    // stop + join the worker pool before tearing down the contexts it touches.
    {
        std::unique_lock<std::mutex> lk(dp->mtx);
        dp->stop = true;
    }
    dp->cv_start.notify_all();
    for (std::thread & t : dp->workers) {
        if (t.joinable()) {
            t.join();
        }
    }

    for (int r = 0; r < dp->n; ++r) {
        llama_batch_free(dp->scratch[r]);
        llama_free(dp->ctxs[r]);
    }
    delete dp;
}

// ---------------------------------------------------------------------------
// memory side — the handle is replica 0's real memory; fan ops to every replica.
// fan via the llama_memory_i methods directly so we don't re-enter the C wrappers.
// ---------------------------------------------------------------------------

void llama_dp_memory_clear(llama_memory_t mem, bool data) {
    llama_dp_context * dp = lookup(g_mems, mem);
    dp->run_on_all([dp, data](int r) { dp->mems[r]->clear(data); });
}

bool llama_dp_memory_seq_rm(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    llama_dp_context * dp = lookup(g_mems, mem);
    const int          n      = dp->n;
    const int          target = (int) (((seq_id % n) + n) % n);
    const llama_seq_id loc    = seq_id / n;
    bool result = true;
    dp->run_on_all([&](int r) {
        if (r == target) {
            result = dp->mems[r]->seq_rm(loc, p0, p1);
        }
    });
    return result;
}

void llama_dp_memory_seq_cp(llama_memory_t mem, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    llama_dp_context * dp = lookup(g_mems, mem);
    const int n = dp->n;
    const int r_src = (int) (((seq_id_src % n) + n) % n);
    const int r_dst = (int) (((seq_id_dst % n) + n) % n);

    if (r_src == r_dst) {
        const llama_seq_id loc_src = seq_id_src / n;
        const llama_seq_id loc_dst = seq_id_dst / n;
        dp->run_on_all([&](int r) {
            if (r == r_src) {
                dp->mems[r]->seq_cp(loc_src, loc_dst, p0, p1);
            }
        });
        return;
    }

    // cross-replica copy (the is_pp_shared prefix-broadcast case) is not supported yet.
    static bool warned = false;
    if (!warned) {
        warned = true;
        LLAMA_LOG_WARN("llama_dp: cross-replica seq_cp (seq %d -> %d) not supported; "
                       "is_pp_shared (-pps) is not handled under data parallelism\n",
                       seq_id_src, seq_id_dst);
    }
}
