// src/llama-dp.cpp — see llama-dp.h for the design overview.

#include "llama-dp.h"
#include "llama-impl.h" // LLAMA_LOG_*

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------------------
// proxy objects
// ---------------------------------------------------------------------------

// N independent replica models, replica r pinned to device r.
struct llama_dp_model {
    int                          n = 0;
    std::vector<llama_model *>   replicas;
};

// N replica contexts + a worker pool (one thread per replica). The hot path shards a batch's
// sequences across the replicas and runs the per-replica decodes concurrently. One context is
// only ever touched by its own worker thread, so no locking is needed around a llama_context.
struct llama_dp_context {
    int                          n = 0;
    int32_t                      n_kv_max = 0;
    int32_t                      n_vocab  = 0;
    llama_model *                model_handle = nullptr; // proxy model handle (for llama_get_model)
    std::vector<llama_context *> ctxs;     // owned
    std::vector<llama_memory_t>  mems;     // ctxs[r]'s memory (cached, not owned)
    std::vector<llama_batch>     scratch;  // per-replica decode scratch (owned)
    std::vector<int8_t>          ret;      // per-replica decode return code

    // per-decode output routing (for llama_get_logits[_ith]). index by the global batch position.
    std::vector<int>             route_r;   // global token i -> replica that decoded it
    std::vector<int>             route_li;  // global token i -> its local batch index on that replica
    std::vector<int8_t>          route_out; // global token i had logits requested
    std::vector<float>           logits_buf; // scratch for the concatenated llama_get_logits()

    // --- worker pool ---
    std::vector<std::thread>     workers;
    std::function<void(int)>     job;
    std::mutex                   mtx;
    std::condition_variable      cv_start;
    std::condition_variable      cv_done;
    uint64_t                     gen   = 0;       // bumped once per dispatch
    std::vector<uint64_t>        seen;            // last gen each worker ran
    int                          ndone = 0;
    bool                         stop  = false;
    bool                         warned_cross_replica = false;

    // run fn(r) on every replica's worker thread, concurrently, and block until all finish.
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
            // failure isolation: a throw on one replica must not take down the process or
            // wedge the barrier; it is reported through the job's own side effects (e.g. ret[r]).
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
// registries — identify proxy handles handed back to the (unmodified) tool
// ---------------------------------------------------------------------------

namespace {
    std::mutex                    g_reg_mtx;
    std::unordered_set<const void *> g_ctx_handles;   // llama_dp_context* (also used as the memory handle)
    std::unordered_set<const void *> g_model_handles; // llama_dp_model*

    void reg_add(std::unordered_set<const void *> & s, const void * p) {
        std::lock_guard<std::mutex> lk(g_reg_mtx);
        s.insert(p);
    }
    void reg_del(std::unordered_set<const void *> & s, const void * p) {
        std::lock_guard<std::mutex> lk(g_reg_mtx);
        s.erase(p);
    }
    bool reg_has(const std::unordered_set<const void *> & s, const void * p) {
        std::lock_guard<std::mutex> lk(g_reg_mtx);
        return s.find(p) != s.end();
    }
}

bool llama_dp_enabled(const llama_model_params & params) {
    return params.n_data_parallel > 1;
}

bool llama_dp_is_model(const llama_model * model) { return reg_has(g_model_handles, model); }
bool llama_dp_is_ctx  (const llama_context * ctx) { return reg_has(g_ctx_handles,   ctx);   }
bool llama_dp_is_mem  (const llama_memory_t mem)  { return reg_has(g_ctx_handles,   mem);   }

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

    llama_model * handle = reinterpret_cast<llama_model *>(dp);
    reg_add(g_model_handles, handle);
    return handle;
}

const llama_vocab * llama_dp_model_vocab(const llama_model * model) {
    const auto * dp = reinterpret_cast<const llama_dp_model *>(model);
    return llama_model_get_vocab(dp->replicas[0]); // identical across replicas
}

void llama_dp_model_free(llama_model * model) {
    auto * dp = reinterpret_cast<llama_dp_model *>(model);
    reg_del(g_model_handles, model);
    for (llama_model * m : dp->replicas) {
        llama_model_free(m);
    }
    delete dp;
}

// ---------------------------------------------------------------------------
// context side
// ---------------------------------------------------------------------------

llama_context * llama_dp_init(llama_model * dp_model_handle, llama_context_params params) {
    auto * dpm = reinterpret_cast<llama_dp_model *>(dp_model_handle);
    const int n = dpm->n;

    auto * dp = new llama_dp_context();
    dp->n = n;
    dp->model_handle = dp_model_handle;
    dp->n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(dpm->replicas[0]));
    dp->ctxs.reserve(n);
    dp->mems.reserve(n);

    for (int r = 0; r < n; ++r) {
        // replica model is a real (non-proxy) model, so this takes the stock path.
        llama_context * c = llama_init_from_model(dpm->replicas[r], params);
        if (!c) {
            LLAMA_LOG_ERROR("llama_dp: failed to create context for replica %d/%d\n", r, n);
            for (llama_context * prev : dp->ctxs) {
                llama_free(prev);
            }
            delete dp;
            return nullptr;
        }
        dp->ctxs.push_back(c);
        dp->mems.push_back(llama_get_memory(c));
    }

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

    llama_context * handle = reinterpret_cast<llama_context *>(dp);
    reg_add(g_ctx_handles, handle);
    return handle;
}

int32_t llama_dp_decode(llama_context * ctx, llama_batch batch) {
    auto * dp = reinterpret_cast<llama_dp_context *>(ctx);
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
        if (dp->scratch[r].n_tokens > 0) {
            dp->ret[r] = (int8_t) llama_decode(dp->ctxs[r], dp->scratch[r]);
        }
    });

    // first nonzero return wins (matches a single-context decode reporting its own failure).
    for (int r = 0; r < n; ++r) {
        if (dp->ret[r] != 0) {
            return dp->ret[r];
        }
    }
    return 0;
}

void llama_dp_synchronize(llama_context * ctx) {
    auto * dp = reinterpret_cast<llama_dp_context *>(ctx);
    dp->run_on_all([dp](int r) { llama_synchronize(dp->ctxs[r]); });
}

uint32_t llama_dp_n_ctx(const llama_context * ctx) {
    const auto * dp = reinterpret_cast<const llama_dp_context *>(ctx);
    return llama_n_ctx(dp->ctxs[0]); // identical across replicas
}

llama_memory_t llama_dp_get_memory(llama_context * ctx) {
    // the proxy context itself is the memory handle; the memory wrappers route via the registry.
    return reinterpret_cast<llama_memory_t>(ctx);
}

uint32_t llama_dp_n_seq_max(const llama_context * ctx) {
    const auto * dp = reinterpret_cast<const llama_dp_context *>(ctx);
    return llama_n_seq_max(dp->ctxs[0]); // identical across replicas
}

const llama_model * llama_dp_get_model(const llama_context * ctx) {
    const auto * dp = reinterpret_cast<const llama_dp_context *>(ctx);
    return dp->model_handle; // the proxy model (its vocab routes to replica 0)
}

float * llama_dp_get_logits_ith(llama_context * ctx, int32_t i) {
    auto * dp = reinterpret_cast<llama_dp_context *>(ctx);
    if (i < 0 || i >= (int) dp->route_r.size()) {
        return nullptr;
    }
    // the whole sequence that owns token i lives on this replica and is contiguous there, so the
    // caller's contiguous read of the following rows stays valid.
    return llama_get_logits_ith(dp->ctxs[dp->route_r[i]], dp->route_li[i]);
}

float * llama_dp_get_logits(llama_context * ctx) {
    auto * dp = reinterpret_cast<llama_dp_context *>(ctx);
    // rebuild the requested-output rows in global batch order (what llama_get_logits returns).
    // only reached when n_ctx > n_batch, which forces n_seq == 1 (all tokens on replica 0).
    dp->logits_buf.clear();
    for (size_t i = 0; i < dp->route_r.size(); ++i) {
        if (!dp->route_out[i]) {
            continue;
        }
        const float * row = llama_get_logits_ith(dp->ctxs[dp->route_r[i]], dp->route_li[i]);
        if (row) {
            dp->logits_buf.insert(dp->logits_buf.end(), row, row + dp->n_vocab);
        }
    }
    return dp->logits_buf.data();
}

void llama_dp_perf_print(const llama_context * ctx) {
    const auto * dp = reinterpret_cast<const llama_dp_context *>(ctx);
    for (int r = 0; r < dp->n; ++r) {
        LLAMA_LOG_INFO("llama_dp: --- replica %d ---\n", r);
        llama_perf_context_print(dp->ctxs[r]);
    }
}

void llama_dp_free(llama_context * ctx) {
    auto * dp = reinterpret_cast<llama_dp_context *>(ctx);
    reg_del(g_ctx_handles, ctx);

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
// memory side — the handle is the proxy context; route ops to the replicas
// ---------------------------------------------------------------------------

void llama_dp_memory_clear(llama_memory_t mem, bool data) {
    auto * dp = reinterpret_cast<llama_dp_context *>(mem);
    dp->run_on_all([dp, data](int r) { llama_memory_clear(dp->mems[r], data); });
}

bool llama_dp_memory_seq_rm(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    auto * dp = reinterpret_cast<llama_dp_context *>(mem);
    const int          n      = dp->n;
    const int          target = (int) (((seq_id % n) + n) % n);
    const llama_seq_id loc    = seq_id / n;
    bool result = true;
    dp->run_on_all([&](int r) {
        if (r == target) {
            result = llama_memory_seq_rm(dp->mems[r], loc, p0, p1);
        }
    });
    return result;
}

void llama_dp_memory_seq_cp(llama_memory_t mem, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    auto * dp = reinterpret_cast<llama_dp_context *>(mem);
    const int n = dp->n;
    const int r_src = (int) (((seq_id_src % n) + n) % n);
    const int r_dst = (int) (((seq_id_dst % n) + n) % n);

    if (r_src == r_dst) {
        const llama_seq_id loc_src = seq_id_src / n;
        const llama_seq_id loc_dst = seq_id_dst / n;
        dp->run_on_all([&](int r) {
            if (r == r_src) {
                llama_memory_seq_cp(dp->mems[r], loc_src, loc_dst, p0, p1);
            }
        });
        return;
    }

    // cross-replica copy (the is_pp_shared prefix-broadcast case) is Phase 2; warn once.
    if (!dp->warned_cross_replica) {
        dp->warned_cross_replica = true;
        LLAMA_LOG_WARN("llama_dp: cross-replica seq_cp (seq %d -> %d) not yet supported; "
                       "is_pp_shared (-pps) is not handled under data parallelism yet\n",
                       seq_id_src, seq_id_dst);
    }
}
