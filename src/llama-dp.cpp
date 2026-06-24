// src/llama-dp.cpp — see llama-dp.h for the design overview.
//
// Single-node: N replica contexts + a worker pool, one thread per replica; llama_decode shards
// the batch's sequences across them.
//
// Multi-node (only when built with -DLLAMA_DP_MPI=ON and launched under mpirun, one rank per node):
// rank 0 is the driver and runs the (unmodified) benchmark; ranks > 0 are hijacked in
// llama_init_from_model into an MPI worker loop and never run the benchmark's main. rank 0's
// llama_decode shards sequences across ALL cluster replicas - local ones via its thread pool,
// remote ones sent to the worker ranks via MPI - so one stock benchmark process drives every GPU
// in the cluster. Only batched-bench (throughput) is supported across nodes; perplexity's logits
// readback is single-node only.

#include "llama-dp.h"
#include "llama-impl.h"    // LLAMA_LOG_*
#include "llama-memory.h"  // call llama_memory_i methods directly when fanning memory ops
#include "llama-context.h" // call llama_context methods directly (bypass the C-API registry)

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef LLAMA_DP_MPI
#include <mpi.h>
#endif

// ---------------------------------------------------------------------------
// side state attached to a lead (replica-0) handle
// ---------------------------------------------------------------------------

struct llama_dp_model {
    int                        n = 0;
    std::vector<llama_model *> replicas; // replicas[0] is the handle returned to the caller
};

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
    std::unordered_map<const void *, llama_dp_model *>   g_models;
    std::unordered_map<const void *, llama_dp_context *> g_ctxs;
    std::unordered_map<const void *, llama_dp_context *> g_mems;

    template <class Map>
    typename Map::mapped_type lookup(Map & m, const void * p) {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto it = m.find(p);
        return it == m.end() ? nullptr : it->second;
    }
}

// ---------------------------------------------------------------------------
// MPI cluster layer (gated). When off, dp_mpi_size() == 1 and the multi-node paths vanish.
// ---------------------------------------------------------------------------

#ifdef LLAMA_DP_MPI
namespace {
    bool g_mpi_inited   = false;
    bool g_mpi_we_init  = false;
    int  g_mpi_rank     = 0;
    int  g_mpi_size     = 1;
    int  g_total_reps   = 0;   // R: total replicas across the cluster (assumes uniform per rank)

    enum dp_msg : int { DP_MSG_DECODE = 1, DP_MSG_SYNC = 2, DP_MSG_CLEAR = 3, DP_MSG_SHUTDOWN = 4 };

    // MPI tags
    enum dp_tag : int { TAG_HDR = 0, TAG_TOK = 1, TAG_POS = 2, TAG_LR = 3, TAG_LSEQ = 4,
                        TAG_LOGITS = 5, TAG_STATUS = 6, TAG_SYNCED = 7 };

    void dp_mpi_init(int local_replicas) {
        if (g_mpi_inited) {
            return;
        }
        int already = 0;
        MPI_Initialized(&already);
        if (!already) {
            int provided = MPI_THREAD_SINGLE;
            // MPI calls happen only on the main thread (FUNNELED); the worker pool never touches MPI.
            MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided);
            g_mpi_we_init = true;
        }
        MPI_Comm_rank(MPI_COMM_WORLD, &g_mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &g_mpi_size);
        MPI_Allreduce(&local_replicas, &g_total_reps, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        g_mpi_inited = true;
        if (g_mpi_rank == 0 && g_mpi_size > 1) {
            LLAMA_LOG_INFO("llama_dp: MPI cluster mode: %d ranks x %d replicas = %d total replicas\n",
                           g_mpi_size, local_replicas, g_total_reps);
        }
    }
}
static int dp_mpi_size()  { return g_mpi_inited ? g_mpi_size : 1; }
static int dp_mpi_rank()  { return g_mpi_inited ? g_mpi_rank : 0; }
static int dp_total_reps(){ return g_total_reps; }
#else
static void dp_mpi_init(int) {}
static int  dp_mpi_size()  { return 1; }
static int  dp_mpi_rank()  { return 0; }
static int  dp_total_reps(){ return 0; }
#endif

bool llama_dp_enabled(const llama_model_params & params) { return params.n_data_parallel > 1; }

bool llama_dp_is_model(const llama_model * m) { std::lock_guard<std::mutex> lk(g_mtx); return g_models.count(m); }
bool llama_dp_is_ctx  (const llama_context * c) { std::lock_guard<std::mutex> lk(g_mtx); return g_ctxs.count(c); }
bool llama_dp_is_mem  (const llama_memory_t m)  { std::lock_guard<std::mutex> lk(g_mtx); return g_mems.count(m); }

// ---------------------------------------------------------------------------
// shared decode primitive: place tokens onto local replicas (by explicit per-token local replica
// + local seq id) and run them concurrently on the worker pool. used by the single-node path, by
// rank 0's local share, and by each worker rank. also records route_* for logits readback.
// ---------------------------------------------------------------------------

static int32_t dp_run_local(llama_dp_context * dp, int n_tokens,
                            const llama_token * tok, const llama_pos * pos,
                            const int * lr, const llama_pos * lseq, const int8_t * logits) {
    const int G = dp->n;
    for (int r = 0; r < G; ++r) {
        dp->scratch[r].n_tokens = 0;
    }
    dp->route_r.assign(n_tokens, 0);
    dp->route_li.assign(n_tokens, 0);
    dp->route_out.assign(n_tokens, 0);

    for (int i = 0; i < n_tokens; ++i) {
        const int r = lr[i];
        llama_batch & s = dp->scratch[r];
        const int k = s.n_tokens;
        s.token   [k]    = tok    ? tok[i]    : 0;
        s.pos     [k]    = pos    ? pos[i]    : 0;
        s.n_seq_id[k]    = 1;
        s.seq_id  [k][0] = lseq[i];
        s.logits  [k]    = logits ? logits[i] : 0;
        s.n_tokens++;

        dp->route_r  [i] = r;
        dp->route_li [i] = k;
        dp->route_out[i] = logits ? logits[i] : 0;
    }

    std::fill(dp->ret.begin(), dp->ret.end(), (int8_t) 0);
    dp->run_on_all([dp](int r) {
        if (dp->scratch[r].n_tokens > 0) {
            dp->ret[r] = (int8_t) dp->ctxs[r]->decode(dp->scratch[r]);
        }
    });
    for (int r = 0; r < G; ++r) {
        if (dp->ret[r] != 0) {
            return dp->ret[r];
        }
    }
    return 0;
}

// stop the pool, free contexts/scratch, unregister, delete. unregisters first so llama_free()
// on the lead handle (ctxs[0]) takes the stock path instead of re-entering this layer.
static void dp_teardown(llama_dp_context * dp) {
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        if (!dp->ctxs.empty()) {
            g_ctxs.erase(dp->ctxs[0]);
        }
        if (!dp->mems.empty()) {
            g_mems.erase(dp->mems[0]);
        }
    }
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
// MPI worker loop (rank > 0) + rank-0 cross-node decode
// ---------------------------------------------------------------------------

#ifdef LLAMA_DP_MPI
// rank > 0: serve rank 0's decode/sync/clear requests until shutdown, then finalize and exit.
// never returns to the benchmark's main().
static void dp_mpi_worker_loop(llama_dp_context * dp) {
    std::vector<llama_token> tok;
    std::vector<llama_pos>   pos, lseq;
    std::vector<int>         lr;
    std::vector<int8_t>      logits;

    for (;;) {
        int hdr[2] = {0, 0};
        MPI_Recv(hdr, 2, MPI_INT, 0, TAG_HDR, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        const int cmd = hdr[0];

        if (cmd == DP_MSG_SHUTDOWN) {
            break;
        }
        if (cmd == DP_MSG_SYNC) {
            dp->run_on_all([dp](int r) { dp->ctxs[r]->synchronize(); });
            int done = 1;
            MPI_Send(&done, 1, MPI_INT, 0, TAG_SYNCED, MPI_COMM_WORLD);
            continue;
        }
        if (cmd == DP_MSG_CLEAR) {
            const bool data = hdr[1] != 0;
            dp->run_on_all([dp, data](int r) { dp->mems[r]->clear(data); });
            continue; // fire-and-forget: FIFO ordering on TAG_HDR keeps it before the next decode
        }
        // DP_MSG_DECODE
        const int n = hdr[1];
        tok.resize(n); pos.resize(n); lr.resize(n); lseq.resize(n); logits.resize(n);
        if (n > 0) {
            MPI_Recv(tok.data(),    n, MPI_INT,  0, TAG_TOK,    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(pos.data(),    n, MPI_INT,  0, TAG_POS,    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(lr.data(),     n, MPI_INT,  0, TAG_LR,     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(lseq.data(),   n, MPI_INT,  0, TAG_LSEQ,   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(logits.data(), n, MPI_CHAR, 0, TAG_LOGITS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        int32_t status = dp_run_local(dp, n, tok.data(), pos.data(), lr.data(), lseq.data(), logits.data());
        MPI_Send(&status, 1, MPI_INT, 0, TAG_STATUS, MPI_COMM_WORLD);
    }

    dp_teardown(dp);
    if (g_mpi_we_init) {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) {
            MPI_Finalize();
        }
    }
    exit(0);
}

// rank 0: shard the batch across all cluster replicas. global replica gr = seq % R; owner node =
// gr / G; local replica = gr % G; local seq id = seq / R. local share runs on the pool; remote
// shares are shipped to the worker ranks, which decode concurrently.
static int32_t dp_decode_mpi(llama_dp_context * dp, llama_batch batch) {
    const int G    = dp->n;
    const int R    = dp_total_reps();
    const int size = dp_mpi_size();

    std::vector<std::vector<llama_token>> tok(size);
    std::vector<std::vector<llama_pos>>   pos(size), lseq(size);
    std::vector<std::vector<int>>         lr(size);
    std::vector<std::vector<int8_t>>      logits(size);

    for (int i = 0; i < batch.n_tokens; ++i) {
        const llama_seq_id seq = (batch.seq_id && batch.n_seq_id && batch.n_seq_id[i] > 0)
                                     ? batch.seq_id[i][0] : 0;
        const int       gr   = (int) (((seq % R) + R) % R);
        const int       node = gr / G;
        const int       lrep = gr % G;
        const llama_pos ls   = seq / R;
        tok   [node].push_back(batch.token  ? batch.token[i]  : 0);
        pos   [node].push_back(batch.pos    ? batch.pos[i]    : 0);
        lr    [node].push_back(lrep);
        lseq  [node].push_back(ls);
        logits[node].push_back(batch.logits ? batch.logits[i] : 0);
    }

    // dispatch remote shares first so they run concurrently with the local decode below.
    for (int g = 1; g < size; ++g) {
        int hdr[2] = { DP_MSG_DECODE, (int) tok[g].size() };
        MPI_Send(hdr, 2, MPI_INT, g, TAG_HDR, MPI_COMM_WORLD);
        const int n = hdr[1];
        if (n > 0) {
            MPI_Send(tok[g].data(),    n, MPI_INT,  g, TAG_TOK,    MPI_COMM_WORLD);
            MPI_Send(pos[g].data(),    n, MPI_INT,  g, TAG_POS,    MPI_COMM_WORLD);
            MPI_Send(lr[g].data(),     n, MPI_INT,  g, TAG_LR,     MPI_COMM_WORLD);
            MPI_Send(lseq[g].data(),   n, MPI_INT,  g, TAG_LSEQ,   MPI_COMM_WORLD);
            MPI_Send(logits[g].data(), n, MPI_CHAR, g, TAG_LOGITS, MPI_COMM_WORLD);
        }
    }

    int32_t status = dp_run_local(dp, (int) tok[0].size(), tok[0].data(), pos[0].data(),
                                  lr[0].data(), lseq[0].data(), logits[0].data());

    for (int g = 1; g < size; ++g) {
        int32_t rstatus = 0;
        MPI_Recv(&rstatus, 1, MPI_INT, g, TAG_STATUS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (status == 0 && rstatus != 0) {
            status = rstatus;
        }
    }
    return status;
}
#endif // LLAMA_DP_MPI

// ---------------------------------------------------------------------------
// model side
// ---------------------------------------------------------------------------

llama_model * llama_dp_model_load(const char * path, llama_model_params params) {
    const int n = params.n_data_parallel;

    dp_mpi_init(n); // no-op unless built with MPI; sets rank/size/total when launched under mpirun

    auto * dp = new llama_dp_model();
    dp->n = n;
    dp->replicas.reserve(n);

    for (int r = 0; r < n; ++r) {
        llama_model_params mp = params;
        mp.n_data_parallel = 0;                    // replica load takes the STOCK path (no recursion)
        mp.split_mode      = LLAMA_SPLIT_MODE_NONE; // one whole replica per device
        mp.main_gpu        = r;                     // pin replica r to (node-local) device r
        mp.devices         = nullptr;
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

    LLAMA_LOG_INFO("llama_dp: loaded %d data-parallel replicas (rank %d/%d)\n",
                   n, dp_mpi_rank(), dp_mpi_size());

    llama_model * handle = dp->replicas[0];
    { std::lock_guard<std::mutex> lk(g_mtx); g_models[handle] = dp; }
    return handle;
}

void llama_dp_model_free(llama_model * model) {
    llama_dp_model * dp = lookup(g_models, model);
    if (!dp) {
        return;
    }
    { std::lock_guard<std::mutex> lk(g_mtx); g_models.erase(model); }
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

    { std::lock_guard<std::mutex> lk(g_mtx); g_models.erase(model); } // avoid recursion in the loop
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
    { std::lock_guard<std::mutex> lk(g_mtx); g_models[model] = dpm; }

    if (!ok) {
        for (llama_context * c : dp->ctxs) {
            llama_free(c);
        }
        delete dp;
        return nullptr;
    }

    dp->n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(dpm->replicas[0]));
    dp->n_kv_max = (int32_t) llama_n_ctx(dp->ctxs[0]);

    dp->scratch.resize(n);
    for (int r = 0; r < n; ++r) {
        dp->scratch[r] = llama_batch_init(dp->n_kv_max, 0, 1);
    }
    dp->ret.assign(n, 0);

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

#ifdef LLAMA_DP_MPI
    if (dp_mpi_size() > 1) {
        MPI_Barrier(MPI_COMM_WORLD); // every rank has its contexts + pool ready
        if (dp_mpi_rank() != 0) {
            dp_mpi_worker_loop(dp);  // never returns (exits at shutdown)
        }
    }
#endif

    return handle;
}

int32_t llama_dp_decode(llama_context * ctx, llama_batch batch) {
    llama_dp_context * dp = lookup(g_ctxs, ctx);

#ifdef LLAMA_DP_MPI
    if (dp_mpi_size() > 1) {
        return dp_decode_mpi(dp, batch);
    }
#endif

    // single-node: replica = seq % G, local seq id = seq / G.
    const int G  = dp->n;
    const int nt = batch.n_tokens;
    std::vector<int>       lr(nt);
    std::vector<llama_pos> lseq(nt);
    for (int i = 0; i < nt; ++i) {
        const llama_seq_id seq = (batch.seq_id && batch.n_seq_id && batch.n_seq_id[i] > 0)
                                     ? batch.seq_id[i][0] : 0;
        lr[i]   = (int) (((seq % G) + G) % G);
        lseq[i] = seq / G;
    }
    return dp_run_local(dp, nt, batch.token, batch.pos, lr.data(), lseq.data(), batch.logits);
}

void llama_dp_synchronize(llama_context * ctx) {
    llama_dp_context * dp = lookup(g_ctxs, ctx);

#ifdef LLAMA_DP_MPI
    if (dp_mpi_size() > 1) {
        // cluster-wide sync so the caller's timing reflects every node, not just the local GPUs.
        for (int g = 1; g < dp_mpi_size(); ++g) {
            int hdr[2] = { DP_MSG_SYNC, 0 };
            MPI_Send(hdr, 2, MPI_INT, g, TAG_HDR, MPI_COMM_WORLD);
        }
        dp->run_on_all([dp](int r) { dp->ctxs[r]->synchronize(); });
        for (int g = 1; g < dp_mpi_size(); ++g) {
            int done = 0;
            MPI_Recv(&done, 1, MPI_INT, g, TAG_SYNCED, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        return;
    }
#endif

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

// perplexity logits readback is single-node only; across nodes the tokens live on other ranks.
static bool dp_logits_unsupported_mpi() {
#ifdef LLAMA_DP_MPI
    if (dp_mpi_size() > 1) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            LLAMA_LOG_ERROR("llama_dp: logits readback (perplexity) is not supported across MPI ranks; "
                            "run perplexity single-node or via the launcher\n");
        }
        return true;
    }
#endif
    return false;
}

float * llama_dp_get_logits_ith(llama_context * ctx, int32_t i) {
    if (dp_logits_unsupported_mpi()) {
        return nullptr;
    }
    llama_dp_context * dp = lookup(g_ctxs, ctx);
    if (i < 0 || i >= (int) dp->route_r.size()) {
        return nullptr;
    }
    return dp_logits_ith(dp->ctxs[dp->route_r[i]], dp->route_li[i]);
}

float * llama_dp_get_logits(llama_context * ctx) {
    if (dp_logits_unsupported_mpi()) {
        return nullptr;
    }
    llama_dp_context * dp = lookup(g_ctxs, ctx);
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

#ifdef LLAMA_DP_MPI
    if (dp_mpi_size() > 1) {
        int hdr[2] = { DP_MSG_SHUTDOWN, 0 };
        for (int g = 1; g < dp_mpi_size(); ++g) {
            MPI_Send(hdr, 2, MPI_INT, g, TAG_HDR, MPI_COMM_WORLD);
        }
    }
#endif

    dp_teardown(dp); // unregisters + stops pool + frees contexts

#ifdef LLAMA_DP_MPI
    if (dp_mpi_size() > 1 && g_mpi_we_init) {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) {
            MPI_Finalize();
        }
    }
#endif
}

// ---------------------------------------------------------------------------
// memory side — the handle is replica 0's real memory; fan ops to every replica.
// ---------------------------------------------------------------------------

void llama_dp_memory_clear(llama_memory_t mem, bool data) {
    llama_dp_context * dp = lookup(g_mems, mem);

#ifdef LLAMA_DP_MPI
    if (dp_mpi_size() > 1) {
        int hdr[2] = { DP_MSG_CLEAR, data ? 1 : 0 };
        for (int g = 1; g < dp_mpi_size(); ++g) {
            MPI_Send(hdr, 2, MPI_INT, g, TAG_HDR, MPI_COMM_WORLD);
        }
    }
#endif

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

    static bool warned = false;
    if (!warned) {
        warned = true;
        LLAMA_LOG_WARN("llama_dp: cross-replica seq_cp (seq %d -> %d) not supported; "
                       "is_pp_shared (-pps) is not handled under data parallelism\n",
                       seq_id_src, seq_id_dst);
    }
}
