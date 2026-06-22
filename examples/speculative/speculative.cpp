#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"
#include "orchestrator.h"

#ifdef LLAMA_ORCH_MPI
#include "cluster.h"
#else
struct cluster_link; // fwd decl: cluster mode compiled out, the pointer stays null
#endif

#include <algorithm>
#include <clocale>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <vector>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

struct seq_draft {
    bool active   = false;
    bool drafting = false;
    bool skip     = false;

    int i_batch_dft = 0;
    std::vector<int> i_batch_tgt;

    std::vector<llama_token> tokens;
    std::vector<std::vector<llama_token_data>> dists;

    struct common_sampler * smpl = nullptr;
};

// per-stream result of one speculative-decoding run (one replica in the DP case).
struct spec_stream_stats {
    int    n_input   = 0;
    int    n_predict = 0;
    int    n_drafted = 0;
    int    n_accept  = 0;
    double t_enc_s   = 0.0;
    double t_dec_s   = 0.0;
    bool   ok        = false;
};

// single-use barrier: forces the work-stealing pool to deal exactly one job to each of the N replicas
// (so replica r's job runs on target ctx r paired with draft ctx r) and aligns the timed start.
// mirrors batched-bench's bb_barrier (std::barrier is C++20; this tree targets C++17).
struct spec_barrier {
    std::mutex              mtx;
    std::condition_variable cv;
    int                     count;
    const int               target;
    explicit spec_barrier(int n) : count(0), target(n) {}
    void arrive_and_wait() {
        std::unique_lock<std::mutex> lk(mtx);
        if (++count >= target) {
            cv.notify_all();
        } else {
            cv.wait(lk, [&] { return count >= target; });
        }
    }
};

// runs ONE speculative-decoding stream on an already-loaded (target, draft) context pair; defined
// after main(). `print` gates the user-facing token stream + perf report (off for DP replicas).
static spec_stream_stats run_spec_stream(
    llama_context * ctx_tgt, llama_context * ctx_dft,
    const llama_model * model_tgt, const llama_model * model_dft,
    common_params params, bool print);  // params by value: common_sampler_init wants a mutable
                                        // sampling ref, and a per-call copy keeps DP replicas race-free

// true if at least one GPU backend device is present. when false everything runs on CPU and a
// main_gpu / split_mode=NONE override would be rejected by llama_prepare_model_devices, so the draft
// placement below is only applied when a GPU exists (mirrors orchestrator_make_pool's own guard).
static bool any_gpu_device() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            return true;
        }
    }
    return false;
}

// STRONG scaling (the default, real-work metric): slice the input corpus into M distinct,
// non-overlapping prompts of params.dp_chunk_chars bytes each and submit all M to the work-stealing
// pool; replicas pull prompts as they finish. Throughput is total generated tokens / wall-clock on
// DISTINCT work - not the same prompt duplicated. At n_rep==1 this is the single-GPU serial baseline.
static int run_dp_strong(const common_params & params, orchestrator_pool & pool,
                         std::vector<common_init_result_ptr> & drafts,
                         const llama_model * model_tgt, int n_rep, cluster_link * link) {
    const int           chunk  = params.dp_chunk_chars > 0 ? params.dp_chunk_chars : 4096;
    const std::string & corpus = params.prompt;

    // slice into non-overlapping chunk-byte shards (drop a trailing partial chunk; if the whole corpus
    // is smaller than one chunk, use it as a single prompt).
    std::vector<std::string> chunks;
    for (size_t off = 0; off + (size_t) chunk <= corpus.size(); off += (size_t) chunk) {
        chunks.emplace_back(corpus.substr(off, (size_t) chunk));
    }
    if (chunks.empty()) {
        chunks.push_back(corpus);
    }
    const int M_total = (int) chunks.size();

    // cross-node partition: each node runs a static, non-overlapping stripe of the corpus (chunk i ->
    // node i % n_nodes); there is NO cross-node work-stealing (intra-node stealing still balances the
    // local replicas). The corpus is identical on every node (same -f on the shared FS), so the stripes
    // tile the work with no inter-node traffic. Single-rank / non-MPI build -> n_nodes==1, rank==0 ->
    // every chunk is local and the path is byte-identical to the single-node run.
    int n_nodes = 1;
    int rank    = 0;
#ifdef LLAMA_ORCH_MPI
    const bool cluster_mode = link && link->size() > 1;
    if (cluster_mode) {
        n_nodes = link->size();
        rank    = link->rank();
    }
#else
    const bool cluster_mode = false;
    (void) link;
#endif

    std::vector<int> local_idx;                       // global chunk indices assigned to THIS node
    for (int i = 0; i < M_total; ++i) {
        if (i % n_nodes == rank) {
            local_idx.push_back(i);
        }
    }
    const int M = (int) local_idx.size();             // jobs on this node
    if (cluster_mode) {
        LOG_INF("%s: strong scaling (cluster): node %d/%d runs %d of %d distinct prompts across %d replicas\n",
                __func__, rank, n_nodes, M, M_total, n_rep);
    } else {
        LOG_INF("%s: strong scaling: %d distinct prompts (%d-byte shards) across %d replicas\n",
                __func__, M_total, chunk, n_rep);
    }

    std::vector<spec_stream_stats> job_stats(M);          // distinct index per job -> race-free
    std::vector<int>               rep_jobs(n_rep, 0);    // each index written only by its own worker
    std::vector<long long>         rep_tokens(n_rep, 0);

#ifdef LLAMA_ORCH_MPI
    // align the timed-region start across nodes: no node begins submitting until all have finished
    // loading + warming up (mirrors batched-bench's pre-sweep link->barrier()).
    if (cluster_mode) {
        link->barrier();
    }
#endif

    for (int j = 0; j < M; ++j) {
        const int i = local_idx[j];                   // global chunk index for this local job
        pool.submit([&job_stats, &drafts, &rep_jobs, &rep_tokens, model_tgt, &params, &chunks, i, j]
                    (llama_context * ctx_tgt, int rr) -> bool {
            common_params jp = params;        // per-job copy (concurrent read-only copy of params is safe)
            jp.prompt = chunks[i];
            job_stats[j] = run_spec_stream(ctx_tgt, drafts[rr]->context(),
                                           model_tgt, drafts[rr]->model(), jp, /*print=*/false);
            rep_jobs[rr]   += 1;
            rep_tokens[rr] += job_stats[j].n_predict;
            return job_stats[j].ok;
        });
    }
    const orchestrator_run_stats run = pool.drain();

    long long tot_predict = 0, tot_drafted = 0, tot_accept = 0;
    double    sum_job_time = 0.0;             // serial-equivalent work (enc+dec), measured under load
    size_t    ok_jobs = 0;
    for (int i = 0; i < M; ++i) {
        const spec_stream_stats & s = job_stats[i];
        if (s.ok) {
            ok_jobs++;
            tot_predict  += s.n_predict;
            tot_drafted  += s.n_drafted;
            tot_accept   += s.n_accept;
            sum_job_time += s.t_enc_s + s.t_dec_s;
        }
    }
    const double wall = run.seconds;

#ifdef LLAMA_ORCH_MPI
    if (cluster_mode) {
        // each node contributes one aggregate line; rank 0 reduces + prints. kilobytes, once per run.
        const std::string blob = string_format("%d %lld %lld %lld %.9g %d %d\n",
            n_rep, tot_predict, tot_drafted, tot_accept, wall, (int) ok_jobs, M);
        const std::vector<std::string> gathered = link->gather(blob);

        if (link->is_head()) {
            LOG("\n");
            LOG("%s: CLUSTER strong-scaling speculative decoding: %d nodes x %d replicas = %d (target+draft) pairs, %d distinct prompts\n",
                __func__, n_nodes, n_rep, n_nodes * n_rep, M_total);
            LOG("|%4s | %6s | %6s | %12s | %9s | %12s | %8s |\n",
                "NODE", "pairs", "jobs", "gen tokens", "wall s", "decode t/s", "accept%");
            LOG("|%4s-|-%6s-|-%6s-|-%12s-|-%9s-|-%12s-|-%8s-|\n",
                "----", "------", "------", "------------", "---------", "------------", "--------");

            double    cl_wall = 0.0;
            long long cl_predict = 0, cl_drafted = 0, cl_accept = 0;
            int       cl_pairs = 0, cl_jobs = 0, cl_ok = 0;
            bool      parse_ok = true;
            for (int nd = 0; nd < (int) gathered.size(); ++nd) {
                int       nrep = 0, okj = 0, mj = 0;
                long long pred = 0, draf = 0, acc = 0;
                double    w = 0.0;
                if (std::sscanf(gathered[nd].c_str(), "%d %lld %lld %lld %lf %d %d",
                                &nrep, &pred, &draf, &acc, &w, &okj, &mj) != 7) {
                    parse_ok = false;
                    break;
                }
                const double node_tps = w    > 0.0 ? (double) pred / w : 0.0;
                const double node_acc = draf > 0    ? 100.0 * (double) acc / (double) draf : 0.0;
                LOG("|%4d | %6d | %6d | %12lld | %9.3f | %12.2f | %7.2f%% |\n",
                    nd, nrep, okj, pred, w, node_tps, node_acc);
                cl_wall     = std::max(cl_wall, w);
                cl_predict += pred;
                cl_drafted += draf;
                cl_accept  += acc;
                cl_pairs   += nrep;
                cl_jobs    += mj;
                cl_ok      += okj;
            }
            if (!parse_ok) {
                LOG_ERR("%s: cluster reduce failed: a node aggregate blob did not parse\n", __func__);
                return 1;
            }
            const double cl_tps = cl_wall    > 0.0 ? (double) cl_predict / cl_wall : 0.0;
            const double cl_acc = cl_drafted > 0    ? 100.0 * (double) cl_accept / (double) cl_drafted : 0.0;
            LOG("|%4s | %6d | %6d | %12lld | %9.3f | %12.2f | %7.2f%% |\n",
                "CL", cl_pairs, cl_jobs, cl_predict, cl_wall, cl_tps, cl_acc);
            LOG("\n");
            LOG("%s: %d/%d prompts ok across cluster | %lld gen tokens | aggregate decode %.2f t/s (slowest-node wall %.3fs) | accept %.2f%%\n",
                __func__, cl_ok, cl_jobs, cl_predict, cl_tps, cl_wall, cl_acc);
            LOG("%s: real cluster speedup vs 1 GPU = compare this slowest-node wall to the '-dp 1 --dp-chunk-chars %d' single-GPU baseline wall (ideal = %dx)\n",
                __func__, chunk, n_nodes * n_rep);
        }
        return ok_jobs == (size_t) M ? 0 : 1;
    }
#endif

    // ---- node-local (single-node) report: unchanged from the pre-cluster path ----
    const double agg_tps = wall > 0.0 ? (double) tot_predict / wall : 0.0;
    const double accept  = tot_drafted > 0 ? 100.0 * (double) tot_accept / (double) tot_drafted : 0.0;
    const double balance = wall > 0.0 ? sum_job_time / wall : 0.0;  // ~n_rep when well balanced

    LOG("\n");
    LOG("%s: strong-scaling DP speculative decoding: %d distinct prompts across %d replicas\n",
        __func__, M, n_rep);
    LOG("|%4s | %6s | %12s |\n", "REP", "jobs", "gen tokens");
    LOG("|%4s-|-%6s-|-%12s-|\n", "----", "------", "------------");
    for (int r = 0; r < n_rep; ++r) {
        LOG("|%4d | %6d | %12lld |\n", r, rep_jobs[r], rep_tokens[r]);
    }
    LOG("\n");
    LOG("%s: %zu/%d prompts ok | %lld gen tokens in %.3fs wall | aggregate decode %.2f t/s | accept %.2f%%\n",
        __func__, ok_jobs, M, tot_predict, wall, agg_tps, accept);
    LOG("%s: load balance: sum(per-prompt enc+dec) %.3fs / wall %.3fs = %.2fx of %d replicas\n",
        __func__, sum_job_time, wall, balance, n_rep);
    LOG("%s: real speedup vs 1 GPU = compare this wall to the '-dp 1 --dp-chunk-chars %d' baseline wall\n",
        __func__, chunk);

    return ok_jobs == (size_t) M ? 0 : 1;
}

// Data-parallel speculative decoding entry: builds N (target+draft) replica pairs (one pair per GPU),
// then runs STRONG scaling (default, distinct corpus shards) or WEAK scaling (--dp-weak diagnostic,
// identical streams). The stock single-stream path (no -dp, no --dp-chunk-chars) never reaches here.
static int speculative_dp(common_params & params, cluster_link * link) {
    llama_backend_init();
    llama_numa_init(params.numa);

    // snapshot the draft placement before building the pool: orchestrator_make_pool builds the
    // TARGET replicas from params.model, so it must run while params.model is still the target.
    const auto    dft_mparams = params.speculative.draft.mparams;
    const int32_t dft_ngl     = params.speculative.draft.n_gpu_layers;

    // one target model replica per device (pinned), built + scheduled by the orchestrator.
    auto pool = orchestrator_make_pool(params, orchestrator_specs_from_params(params));
    if (!pool) {
        LOG_ERR("%s: failed to build the target replica pool\n", __func__);
        llama_backend_free();
        return 1;
    }
    const int n_rep = pool->size();

    // one draft model + context per replica, co-located on that replica's GPU (pin via main_gpu,
    // mirroring how the orchestrator pins single-device target replicas). on a CPU-only build there
    // is no device to pin to, so the override is skipped (the draft loads on CPU like the stock tool).
    const bool has_gpu = any_gpu_device();
    std::vector<common_init_result_ptr> drafts;
    drafts.reserve(n_rep);
    for (int r = 0; r < n_rep; ++r) {
        common_params dp = params;                  // copy; mutate only the draft model + placement
        dp.model        = dft_mparams;
        dp.n_gpu_layers = dft_ngl;
        if (has_gpu) {
            dp.devices.clear();                     // co-locate with target replica r via main_gpu
            dp.split_mode = LLAMA_SPLIT_MODE_NONE;
            dp.main_gpu   = pool->at(r).device;
        }

        auto init = common_init_from_params(dp);
        if (!init || !init->model() || !init->context()) {
            LOG_ERR("%s: replica %d: failed to load draft model '%s'\n",
                    __func__, r, dft_mparams.path.c_str());
            drafts.clear();
            pool.reset();
            llama_backend_free();
            return 1;
        }
        drafts.push_back(std::move(init));
    }

    LOG_INF("%s: built %d (target+draft) replica pair(s)\n", __func__, n_rep);

    const llama_model * model_tgt = pool->model();

    // STRONG scaling (default): shard the corpus into distinct prompts across replicas - the real-work
    // throughput metric. WEAK scaling (--dp-weak) below is the identical-stream diagnostic.
    if (!params.dp_weak) {
        const int rc = run_dp_strong(params, *pool, drafts, model_tgt, n_rep, link);
        drafts.clear();
        pool.reset();
        llama_backend_free();
        return rc;
    }

    // --- WEAK scaling (--dp-weak): every replica runs the SAME prompt. the barrier deals exactly one
    // job to each replica (1:1 pairing) and aligns the timed start. NOT a real-work throughput number.
#ifdef LLAMA_ORCH_MPI
    const bool cluster_mode = link && link->size() > 1;
    const int  n_nodes      = cluster_mode ? link->size() : 1;
#else
    const bool cluster_mode = false;
    (void) link;
#endif

    std::vector<spec_stream_stats> rep_stats(n_rep);
    auto                           barrier   = std::make_shared<spec_barrier>(n_rep);
#ifdef LLAMA_ORCH_MPI
    if (cluster_mode) {
        link->barrier();   // align the timed start across nodes before any replica runs
    }
#endif
    for (int r = 0; r < n_rep; ++r) {
        pool->submit([&rep_stats, &drafts, model_tgt, &params, barrier](llama_context * ctx_tgt, int rr) -> bool {
            barrier->arrive_and_wait();
            rep_stats[rr] = run_spec_stream(ctx_tgt, drafts[rr]->context(),
                                            model_tgt, drafts[rr]->model(),
                                            params, /*print=*/false);
            return rep_stats[rr].ok;
        });
    }
    const orchestrator_run_stats run = pool->drain();

    // per-replica decode tok/s + acceptance, then an ALL row carrying the weak-scaling totals. In
    // cluster mode the per-replica table is suppressed (it would interleave across ranks in the shared
    // log); rank 0 prints a per-NODE + CL table after the gather below.
    if (!cluster_mode) {
        LOG("\n");
        LOG("%s: weak-scaling DP speculative decoding: %d replicas, same prompt each\n", __func__, n_rep);
        LOG("|%4s | %9s | %9s | %9s | %8s | %12s |\n", "REP", "n_predict", "n_drafted", "n_accept", "accept%", "decode t/s");
        LOG("|%4s-|-%9s-|-%9s-|-%9s-|-%8s-|-%12s-|\n", "----", "---------", "---------", "---------", "--------", "------------");
    }

    int    tot_predict = 0, tot_drafted = 0, tot_accept = 0;
    double max_t_dec   = 0.0;
    size_t ok_reps     = 0;
    for (int r = 0; r < n_rep; ++r) {
        const spec_stream_stats & s = rep_stats[r];
        const double dec_tps = s.t_dec_s   > 0.0 ? s.n_predict / s.t_dec_s     : 0.0;
        const double acc_pct = s.n_drafted > 0   ? 100.0 * s.n_accept / s.n_drafted : 0.0;
        if (!cluster_mode) {
            char rep_label[8];
            snprintf(rep_label, sizeof(rep_label), "%d", r);
            LOG("|%4s | %9d | %9d | %9d | %7.2f%% | %12.2f |\n",
                rep_label, s.n_predict, s.n_drafted, s.n_accept, acc_pct, dec_tps);
        }
        if (s.ok) {
            ok_reps++;
            tot_predict += s.n_predict;
            tot_drafted += s.n_drafted;
            tot_accept  += s.n_accept;
            max_t_dec    = std::max(max_t_dec, s.t_dec_s);
        }
    }

#ifdef LLAMA_ORCH_MPI
    if (cluster_mode) {
        // aggregate decode throughput: total predicted tokens over the concurrent span. each node
        // contributes its weak aggregate; rank 0 prints the per-NODE + CL table.
        const std::string blob = string_format("%d %d %d %d %.9g %.9g\n",
            n_rep, tot_predict, tot_drafted, tot_accept, max_t_dec, run.seconds);
        const std::vector<std::string> gathered = link->gather(blob);

        if (link->is_head()) {
            LOG("\n");
            LOG("%s: CLUSTER weak-scaling DP speculative decoding: %d nodes x %d replicas, same prompt each (diagnostic)\n",
                __func__, n_nodes, n_rep);
            LOG("|%4s | %9s | %9s | %9s | %8s | %12s |\n", "NODE", "n_predict", "n_drafted", "n_accept", "accept%", "decode t/s");
            LOG("|%4s-|-%9s-|-%9s-|-%9s-|-%8s-|-%12s-|\n", "----", "---------", "---------", "---------", "--------", "------------");

            long long cl_predict = 0, cl_drafted = 0, cl_accept = 0;
            double    cl_max_dec = 0.0, cl_e2e = 0.0;
            int       cl_pairs = 0;
            bool      parse_ok = true;
            for (int nd = 0; nd < (int) gathered.size(); ++nd) {
                int    nrep = 0, pred = 0, draf = 0, acc = 0;
                double mdec = 0.0, e2e = 0.0;
                if (std::sscanf(gathered[nd].c_str(), "%d %d %d %d %lf %lf",
                                &nrep, &pred, &draf, &acc, &mdec, &e2e) != 6) {
                    parse_ok = false;
                    break;
                }
                const double node_tps = mdec > 0.0 ? (double) pred / mdec : 0.0;
                const double node_acc = draf > 0   ? 100.0 * (double) acc / (double) draf : 0.0;
                LOG("|%4d | %9d | %9d | %9d | %7.2f%% | %12.2f |\n",
                    nd, pred, draf, acc, node_acc, node_tps);
                cl_predict += pred;
                cl_drafted += draf;
                cl_accept  += acc;
                cl_max_dec  = std::max(cl_max_dec, mdec);
                cl_e2e      = std::max(cl_e2e, e2e);
                cl_pairs   += nrep;
            }
            if (!parse_ok) {
                LOG_ERR("%s: cluster reduce failed: a node aggregate blob did not parse\n", __func__);
            } else {
                const double cl_tps_max = cl_max_dec > 0.0 ? (double) cl_predict / cl_max_dec : 0.0;
                const double cl_tps_e2e = cl_e2e     > 0.0 ? (double) cl_predict / cl_e2e     : 0.0;
                const double cl_acc     = cl_drafted > 0   ? 100.0 * (double) cl_accept / (double) cl_drafted : 0.0;
                LOG("|%4s | %9lld | %9lld | %9lld | %7.2f%% | %12.2f |\n",
                    "CL", cl_predict, cl_drafted, cl_accept, cl_acc, cl_tps_max);
                LOG("\n");
                LOG("%s: %d (target+draft) pairs; cluster aggregate decode %.2f t/s (max-span) / %.2f t/s (e2e drain) | accept %.2f%%\n",
                    __func__, cl_pairs, cl_tps_max, cl_tps_e2e, cl_acc);
            }
        }
    } else
#endif
    {
        // aggregate decode throughput: total predicted tokens over the concurrent span. max_t_dec =
        // slowest replica's own decode time; run.seconds = pool drain() span (dispatch -> completion).
        const double agg_tps_max = max_t_dec   > 0.0 ? tot_predict / max_t_dec  : 0.0;
        const double agg_tps_e2e = run.seconds > 0.0 ? tot_predict / run.seconds : 0.0;
        const double agg_acc_pct = tot_drafted > 0   ? 100.0 * tot_accept / tot_drafted : 0.0;
        LOG("|%4s | %9d | %9d | %9d | %7.2f%% | %12.2f |\n",
            "ALL", tot_predict, tot_drafted, tot_accept, agg_acc_pct, agg_tps_max);
        LOG("\n");
        LOG("%s: %zu/%d replicas ok; aggregate decode %.2f t/s (max-span) / %.2f t/s (e2e drain %.3fs)\n",
            __func__, ok_reps, n_rep, agg_tps_max, agg_tps_e2e, run.seconds);
    }

    // teardown order: draft contexts first, then the pool (joins workers, frees target contexts).
    drafts.clear();
    pool.reset();
    llama_backend_free();
    return ok_reps == (size_t) n_rep ? 0 : 1;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // initialize the cross-node link first (MPI_Init may consume its own argv entries). A non-MPI
    // build or a single-rank launch leaves `link` benign (size()==1) -> no behaviour change.
    cluster_link * link = nullptr;
#ifdef LLAMA_ORCH_MPI
    std::unique_ptr<cluster_link> cluster = cluster_link::init(&argc, &argv);
    link = cluster.get();
#endif

    common_params params;

    // needed to get candidate probs even for temp <= 0.0
    params.sampling.n_probs = 128;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    if (params.speculative.draft.mparams.path.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // data-parallel path: N (target+draft) replicas run spec-decode streams concurrently. engages when
    // the orchestrator is active (-dp > 1) OR a strong-scaling shard size is given (--dp-chunk-chars > 0,
    // which also enables the n_rep==1 serial baseline). plain single-stream (no -dp, no chunk) falls
    // through to the unchanged path below.
#ifdef LLAMA_ORCH_MPI
    // cross-node aggregation lives in the data-parallel path; warn if launched multi-rank without a DP
    // flag (each rank would otherwise run an independent single-stream spec decode and print its own).
    if (link && link->size() > 1 && !(orchestrator_dp_active(params) || params.dp_chunk_chars > 0) && link->is_head()) {
        LOG_WRN("%s: launched on %d ranks but no DP path is active; cluster aggregation needs -dp or "
                "--dp-chunk-chars. Each rank will run independently.\n", __func__, link->size());
    }
#endif

    if (orchestrator_dp_active(params) || params.dp_chunk_chars > 0) {
        return speculative_dp(params, link);
    }

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;
    llama_model * model_dft = NULL;

    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;

    // load the target model
    auto llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt->model();
    ctx_tgt   = llama_init_tgt->context();

    // load the draft model
    params.devices = params.speculative.draft.devices;
    params.model = params.speculative.draft.mparams;
    params.n_gpu_layers = params.speculative.draft.n_gpu_layers;
    if (params.speculative.draft.cpuparams.n_threads > 0) {
        params.cpuparams.n_threads = params.speculative.draft.cpuparams.n_threads;
    }

    params.cpuparams_batch.n_threads = params.speculative.draft.cpuparams_batch.n_threads;
    params.tensor_buft_overrides     = params.speculative.draft.tensor_buft_overrides;

    auto llama_init_dft = common_init_from_params(params);

    model_dft = llama_init_dft->model();
    ctx_dft   = llama_init_dft->context();

    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_ERR("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_ERR("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return 1;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        LOG_ERR("%s: draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return 1;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        LOG_ERR("%s: draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return 1;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_ERR("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_ERR("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return 1;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_ERR("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_ERR("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return 1;
            }
        }
    }

    const spec_stream_stats st = run_spec_stream(ctx_tgt, ctx_dft, model_tgt, model_dft, params, /*print=*/true);

    llama_backend_free();

    LOG("\n\n");

    return st.ok ? 0 : 1;
}

// Definition of the single-stream runner forward-declared above. The body below is the original
// single-stream loop verbatim (only the user-facing prints are gated by `print`), so the single-stream
// path stays byte-identical; the DP path calls it once per replica with print=false.
static spec_stream_stats run_spec_stream(
        llama_context * ctx_tgt, llama_context * ctx_dft,
        const llama_model * model_tgt, const llama_model * model_dft,
        common_params params, bool print) {
    spec_stream_stats stats;

    const int   n_seq_dft     = params.n_parallel;
    const float p_draft_split = params.speculative.draft.p_split;

    std::default_random_engine rng(params.sampling.seed == LLAMA_DEFAULT_SEED ? std::random_device()() : params.sampling.seed);
    std::uniform_real_distribution<> u_dist;

    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);

    auto * mem_tgt = llama_get_memory(ctx_tgt);
    auto * mem_dft = llama_get_memory(ctx_dft);

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    const int max_context_size     = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return stats;
    }

    if (print) {
        LOG("\n\n");

        for (auto id : inp) {
            LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
        }
    }

    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us();

    // eval the prompt with both models
    llama_decode(ctx_tgt, llama_batch_get_one( inp.data(), n_input - 1));
    llama_decode(ctx_tgt, llama_batch_get_one(&inp.back(),           1));
    llama_decode(ctx_dft, llama_batch_get_one( inp.data(), n_input));

    const auto t_enc_end = ggml_time_us();

    // the 2 models should have the same vocab
    //GGML_ASSERT(n_vocab == llama_vocab_n_tokens(model_dft));

    // how many tokens to draft each time
    int n_draft = params.speculative.draft.n_max;

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    int n_past_tgt = inp.size();
    int n_past_dft = inp.size();

    // used to determine end of generation
    bool has_eos = false;

    // target model sampling context (reuse the llama_context's sampling instance)
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    // draft sequence data
    std::vector<seq_draft> drafts(n_seq_dft);

    for (int s = 0; s < n_seq_dft; ++s) {
        // allocate llama_sampler for each draft sequence
        drafts[s].smpl = common_sampler_init(model_dft, params.sampling);
    }

    llama_batch batch_dft = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, n_seq_dft);

    const auto t_dec_start = ggml_time_us();

    // sample from the last token of the prompt
    drafts[0].i_batch_tgt.resize(1);
    drafts[0].i_batch_tgt[0] = 0;

    while (true) {
        std::set<int> active_seqs = {};

        // print current draft sequences
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }

            active_seqs.insert(s);
            const auto & tokens = drafts[s].tokens;

            LOG_DBG("draft %d: %s\n", s, string_from(ctx_dft, tokens).c_str());
        }

        int i_dft  = 0;
        int s_keep = 0;

        llama_token token_id;
        std::string token_str;

        // loop until we fail to accept a drafted token or we run out of drafted tokens
        while (true) {

            // check if the target token matches any of the drafts
            // for stochastic sampling, attempt to match the token with the drafted tokens
            {
                bool accept = false;
                if (params.sampling.temp > 0) {
                    // stochastic verification
                    common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft], true);

                    auto & dist_tgt = *common_sampler_get_candidates(smpl, true);

                    float p_tgt = 0.0f;
                    float p_dft = 0.0f;

                    while (active_seqs.size() > 0) {
                        // randomly select a sequence to verify from active sequences
                        std::uniform_int_distribution<unsigned int> u_int_dist(0, active_seqs.size() - 1);
                        int s = *std::next(active_seqs.begin(), u_int_dist(rng));
                        if (i_dft >= (int) drafts[s].tokens.size()) {
                            drafts[s].active = false;
                            active_seqs.erase(s);
                            continue;
                        }
                        if (accept) {
                            // if we already accepted a token, we can skip the rest
                            if (drafts[s].tokens[i_dft] != drafts[s_keep].tokens[i_dft]) {
                                drafts[s].active = false;
                                active_seqs.erase(s);
                            }
                            continue;
                        }

                        LOG_DBG("verifying sequence #%d at pos #%d from %d active sequence(s)\n", s, i_dft, (int) active_seqs.size());
                        float r = u_dist(rng);
                        llama_token_data_array dist_dft = { drafts[s].dists[i_dft].data() , drafts[s].dists[i_dft].size(), LLAMA_TOKEN_NULL, true };

                        //GGML_ASSERT(dist_tgt.size <= dist_dft.size);

                        // acquire the token probabilities assigned by the draft and target models
                        for (size_t i = 0; i < dist_tgt.size; i++) {
                            if (dist_tgt.data[i].id == drafts[s].tokens[i_dft]) {
                                p_tgt = dist_tgt.data[i].p;
                                break;
                            }
                        }
                        for (size_t i = 0; i < dist_dft.size; i++) {
                            if (dist_dft.data[i].id == drafts[s].tokens[i_dft]) {
                                p_dft = dist_dft.data[i].p;
                                break;
                            }
                        }
                        LOG_DBG("r = %f, p_dft = %f, p_tgt = %f\n", r, p_dft, p_tgt);
                        if (r <= p_tgt / p_dft) {
                            s_keep = s;
                            accept = true;
                            token_id = drafts[s].tokens[i_dft];
                            token_str = common_token_to_piece(ctx_tgt, token_id);
                            common_sampler_accept(smpl, token_id, true);

                            LOG_DBG("draft token %d of sequence %d (%d, '%s') accepted\n", i_dft, s, token_id, token_str.c_str());
                            break;
                        } else {
                            LOG_DBG("draft token %d of sequence %d (%d, '%s') rejected\n", i_dft, s, drafts[s].tokens[i_dft], common_token_to_piece(ctx_tgt, drafts[s].tokens[i_dft]).c_str());
                            drafts[s].active = false;

                            // calculate residual probability
                            GGML_ASSERT(dist_tgt.sorted);
                            GGML_ASSERT(dist_dft.sorted);

                            // sort dist by id
                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.id < b.id;
                            });
                            std::sort(dist_dft.data, dist_dft.data + dist_dft.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.id < b.id;
                            });

                            float sum_probs = 0.0f;

                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                if (i < dist_dft.size) {
                                    dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p - dist_dft.data[i].p);
                                } else {
                                    dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p);
                                }

                                sum_probs += dist_tgt.data[i].p;
                            }

                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                dist_tgt.data[i].p /= sum_probs;
                            }

                            // sort dist_tgt by p desc
                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.p > b.p;
                            });
                        }

                        active_seqs.erase(s);
                        for (int i = 0; i < n_seq_dft; i++) {
                            if (i == s) {
                                continue;
                            }
                            if (drafts[i].active && drafts[i].tokens[i_dft] == drafts[s].tokens[i_dft]) {
                                // synchronize active status for sequences with the same drafted token
                                drafts[i].active = drafts[i].active && accept;
                                if (!drafts[i].active) {
                                    active_seqs.erase(s);
                                }
                            }
                        }
                    }

                    if (!accept) {
                        // all drafted tokens were rejected
                        // sample from the target model
                        LOG_DBG("all drafted tokens were rejected, sampling from residual distribution\n");
                        std::vector<float> probs(dist_tgt.size);
                        for (size_t i = 0; i < dist_tgt.size; ++i) {
                            probs[i] = dist_tgt.data[i].p;
                        }

                        std::discrete_distribution<> dist(probs.begin(), probs.end());

                        const int idx = dist(rng);

                        token_id = dist_tgt.data[idx].id;
                        common_sampler_accept(smpl, token_id, true);
                        token_str = common_token_to_piece(ctx_tgt, token_id);
                    }
                } else {
                    // greedy verification

                    // sample from the target model
                    LOG_DBG("sampling target: s_keep = %3d, i_dft = %3d, i_batch_tgt = %3d\n", s_keep, i_dft, drafts[s_keep].i_batch_tgt[i_dft]);
                    token_id = common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft]);

                    common_sampler_accept(smpl, token_id, true);

                    token_str = common_token_to_piece(ctx_tgt, token_id);

                    for (int s = 0; s < n_seq_dft; ++s) {
                        if (!drafts[s].active) {
                            continue;
                        }

                        if (i_dft < (int) drafts[s].tokens.size() && token_id == drafts[s].tokens[i_dft]) {
                            LOG_DBG("the sampled target token matches the %dth drafted token of sequence %d (%d, '%s') - accepted\n", i_dft, s, token_id, token_str.c_str());

                            s_keep = s;
                            accept = true;
                        } else {
                            drafts[s].active = false;
                        }
                    }
                }

                if (llama_vocab_is_eog(vocab_tgt, token_id)) {
                    has_eos = true;
                }
                ++n_predict;

                if (accept) {
                    ++n_accept;
                    ++n_past_tgt;
                    ++n_past_dft;
                    ++i_dft;
                    if (print) {
                    if (params.use_color) {
                        // Color token according to its origin sequence
                        LOG("\u001b[%dm%s\u001b[37m", (36 - s_keep % 6), token_str.c_str());
                    } else {
                        LOG("%s", token_str.c_str());
                    }
                    }
                    continue;
                } else {
                    if (print) {
                        LOG("%s", token_str.c_str());
                    }
                    break;
                }
            }
        }

        {
            LOG_DBG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", token_id, token_str.c_str());

            // TODO: simplify
            {
                LOG_DBG("keeping sequence %d, n_past_tgt = %d, n_past_dft = %d\n", s_keep, n_past_tgt, n_past_dft);

                llama_memory_seq_keep(mem_dft, s_keep);
                llama_memory_seq_cp  (mem_dft, s_keep, 0, -1, -1);
                llama_memory_seq_keep(mem_dft, 0);

                llama_memory_seq_rm  (mem_tgt, s_keep, n_past_tgt, -1);
                llama_memory_seq_keep(mem_tgt, s_keep);
                llama_memory_seq_cp  (mem_tgt, s_keep, 0, -1, -1);
                llama_memory_seq_keep(mem_tgt, 0);
            }

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].active = false;
                drafts[s].tokens.clear();
                drafts[s].i_batch_tgt.clear();
                drafts[s].dists.clear();
            }
            // note: will be erased after the speculation phase
            drafts[0].tokens.push_back(token_id);
            drafts[0].dists.push_back(std::vector<llama_token_data>());
            drafts[0].i_batch_tgt.push_back(0);

            common_batch_clear(batch_dft);
            common_batch_add  (batch_dft, token_id, n_past_dft, { 0 }, true);

            llama_memory_seq_rm(mem_dft, 0, n_past_dft, -1);
            // LOG_DBG("dft batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_dft, batch_dft).c_str());
            llama_decode(ctx_dft, batch_dft);

            ++n_past_dft;
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }

        if (drafts[0].smpl) {
            common_sampler_free(drafts[0].smpl);
        }
        drafts[0].smpl = common_sampler_clone(smpl);

        int n_seq_cur  = 1;
        int n_past_cur = n_past_dft;

        for (int s = 0; s < n_seq_dft; ++s) {
            drafts[s].active   = false;
            drafts[s].drafting = false;
        }
        drafts[0].active      = true;
        drafts[0].drafting    = true;
        drafts[0].i_batch_dft = 0;

        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, drafts[0].tokens[0], n_past_tgt, { 0 }, true);

        // sample n_draft tokens from the draft model using tree-based sampling
        for (int i = 0; i < n_draft; ++i) {
            batch_dft.n_tokens = 0;

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].skip = false;
            }

            for (int s = 0; s < n_seq_dft; ++s) {
                if (!drafts[s].drafting || drafts[s].skip) {
                    continue;
                }

                common_sampler_sample(drafts[s].smpl, ctx_dft, drafts[s].i_batch_dft, true);

                const auto * cur_p = common_sampler_get_candidates(drafts[s].smpl, true);

                for (int k = 0; k < std::min(n_seq_dft + 3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - draft candidate %3d for seq %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            k, s, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                std::vector<int> sa(1, s);

                // attempt to split the branch if the probability is high enough
                for (int f = 1; f < 8; ++f) {
                    if (n_seq_cur < n_seq_dft && cur_p->data[f].p > p_draft_split) {
                        LOG_DBG("splitting seq %3d into %3d\n", s, n_seq_cur);

                        llama_memory_seq_rm(mem_dft,    n_seq_cur, -1, -1);
                        llama_memory_seq_cp(mem_dft, s, n_seq_cur, -1, -1);

                        // all previous tokens from this branch are now also part of the new branch
                        for (int t = 0; t < batch_tgt.n_tokens; ++t) {
                            for (int p = 0; p < batch_tgt.n_seq_id[t]; ++p) {
                                if (batch_tgt.seq_id[t][p] == s) {
                                    batch_tgt.seq_id[t][batch_tgt.n_seq_id[t]] = n_seq_cur;
                                    batch_tgt.n_seq_id[t]++;
                                    break;
                                }
                            }
                        }

                        // copy the draft state
                        drafts[n_seq_cur].active   = true;
                        drafts[n_seq_cur].drafting = true;
                        drafts[n_seq_cur].skip     = true;

                        drafts[n_seq_cur].tokens      = drafts[s].tokens;
                        drafts[n_seq_cur].dists       = drafts[s].dists;
                        drafts[n_seq_cur].i_batch_dft = drafts[s].i_batch_dft;
                        drafts[n_seq_cur].i_batch_tgt = drafts[s].i_batch_tgt;

                        if (drafts[n_seq_cur].smpl) {
                            common_sampler_free(drafts[n_seq_cur].smpl);
                        }
                        drafts[n_seq_cur].smpl = common_sampler_clone(drafts[s].smpl);

                        sa.push_back(n_seq_cur);

                        n_seq_cur++;
                    } else {
                        break;
                    }
                }

                // add drafted token for each sequence
                for (int is = 0; is < (int) sa.size(); ++is) {
                    const llama_token id = cur_p->data[is].id;

                    const int s = sa[is];

                    common_sampler_accept(drafts[s].smpl, id, true);

                    drafts[s].tokens.push_back(id);
                    // save cur_p.data into drafts[s].dists
                    drafts[s].dists.push_back({cur_p->data, cur_p->data + cur_p->size});

                    // add unique drafted tokens to the target batch
                    drafts[s].i_batch_tgt.push_back(batch_tgt.n_tokens);

                    common_batch_add(batch_tgt, id, n_past_tgt + i + 1, { s }, true);

                    // add the token to the batch for batched decoding with the draft model
                    drafts[s].i_batch_dft = batch_dft.n_tokens;

                    common_batch_add(batch_dft, id, n_past_cur, { s }, true);

                    if (batch_tgt.n_tokens > n_draft) {
                        drafts[s].drafting = false;
                    }
                }
            }

            // no sequence is drafting anymore
            if (batch_dft.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            llama_decode(ctx_dft, batch_dft);
            ++n_past_cur;
            ++n_drafted;

            if (batch_tgt.n_tokens > n_draft) {
                break;
            }
        }

        // evaluate the target model on the drafted tokens
        {
            llama_memory_seq_keep(mem_tgt, 0);
            for (int s = 1; s < n_seq_dft; ++s) {
                llama_memory_seq_cp(mem_tgt, 0, s, -1, -1);
            }

            // LOG_DBG("target batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_tgt, batch_tgt).c_str());
            llama_decode(ctx_tgt, batch_tgt);
            ++n_past_tgt;
        }

        // the first token is always proposed by the target model before the speculation loop so we erase it here
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }

            drafts[s].tokens.erase(drafts[s].tokens.begin());
            drafts[s].dists.erase(drafts[s].dists.begin());
        }
    }

    auto t_dec_end = ggml_time_us();

    if (print) {
        LOG("\n\n");

        LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
        LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

        LOG_INF("\n");
        LOG_INF("n_draft   = %d\n", n_draft);
        LOG_INF("n_predict = %d\n", n_predict);
        LOG_INF("n_drafted = %d\n", n_drafted);
        LOG_INF("n_accept  = %d\n", n_accept);
        LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

        LOG_INF("\n");
        LOG_INF("draft:\n\n");
        // TODO: print sampling/grammar timings for all drafts
        llama_perf_context_print(ctx_dft);

        LOG_INF("\n");
        LOG_INF("target:\n\n");
        common_perf_print(ctx_tgt, smpl);
    }

    common_sampler_free(smpl);
    for (int s = 0; s < n_seq_dft; ++s) {
        common_sampler_free(drafts[s].smpl);
    }

    llama_batch_free(batch_dft);
    llama_batch_free(batch_tgt);

    stats.n_input   = n_input;
    stats.n_predict = n_predict;
    stats.n_drafted = n_drafted;
    stats.n_accept  = n_accept;
    stats.t_enc_s   = (t_enc_end - t_enc_start) / 1e6;
    stats.t_dec_s   = (t_dec_end - t_dec_start) / 1e6;
    stats.ok        = true;
    return stats;
}
