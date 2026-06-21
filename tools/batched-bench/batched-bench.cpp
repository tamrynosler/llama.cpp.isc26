#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "orchestrator.h"

#ifdef LLAMA_ORCH_MPI
#include "cluster.h"
#include <sstream>
#else
struct cluster_link; // fwd decl: cluster mode compiled out, the pointer stays null
#endif

#include <algorithm>
#include <clocale>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -c 2048 -b 2048 -ub 512 -npp 128,256,512 -ntg 128,256 -npl 1,2,4,8,16,32 [-pps]\n", argv[0]);
    LOG("\n");
}

// ---- data-parallel batched-bench (S10) ----
// weak scaling: each of N replicas runs the byte-identical single-context benchmark on its own
// device, concurrently; throughput is aggregated across replicas. a single resolved replica never
// reaches here, so the baseline path stays byte-for-byte unchanged.

// single-use barrier (std::barrier is C++20; this tree targets C++17). submitting exactly n_rep
// jobs that each arrive here makes the otherwise work-stealing pool hand one job to every replica:
// a worker that takes a job blocks here until all n_rep have, so none can grab a second. that gives
// a 1:1 job->replica mapping (and a synchronized start to the timed region).
struct bb_barrier {
    std::mutex              m;
    std::condition_variable cv;
    int                     waiting = 0;
    const int               target;
    explicit bb_barrier(int n) : target(n) {}
    void arrive_and_wait() {
        std::unique_lock<std::mutex> lk(m);
        if (++waiting >= target) {
            cv.notify_all();
        } else {
            cv.wait(lk, [this] { return waiting >= target; });
        }
    }
};

// mirrors the decode_helper lambda in the baseline path; kept separate so the baseline stays untouched.
static bool bb_decode_helper(llama_context * ctx, llama_batch & batch, int32_t n_batch, bool synchronize) {
    for (int32_t i = 0; i < batch.n_tokens; i += n_batch) {
        const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

        llama_batch batch_view = {
            n_tokens,
            batch.token    + i,
            nullptr,
            batch.pos      + i,
            batch.n_seq_id + i,
            batch.seq_id   + i,
            batch.logits   + i,
        };

        const int ret = llama_decode(ctx, batch_view);
        if (ret != 0) {
            LOG_ERR("failed to decode the batch, n_batch = %d, ret = %d\n", n_batch, ret);
            return false;
        }

        if (synchronize) {
            llama_synchronize(ctx);
        }
    }

    return true;
}

// print one human-readable table row. the rep label is the replica index ("0".."N-1") or "ALL"
// for the aggregate; the T_e2e/S_e2e (drain-span) columns are only filled on the aggregate row.
static void bb_print_row(const char * rep, int pp, int tg, int pl, int n_kv,
                         float t_pp, float s_pp, float t_tg, float s_tg, float t, float s,
                         bool has_e2e, float t_e2e, float s_e2e) {
    if (has_e2e) {
        LOG("|%4s | %6d | %6d | %4d | %6d | %8.3f | %8.2f | %8.3f | %8.2f | %8.3f | %8.2f | %8.3f | %9.2f |\n",
            rep, pp, tg, pl, n_kv, t_pp, s_pp, t_tg, s_tg, t, s, t_e2e, s_e2e);
    } else {
        LOG("|%4s | %6d | %6d | %4d | %6d | %8.3f | %8.2f | %8.3f | %8.2f | %8.3f | %8.2f | %8s | %9s |\n",
            rep, pp, tg, pl, n_kv, t_pp, s_pp, t_tg, s_tg, t, s, "", "");
    }
}

static int batched_bench_dp(common_params & params, cluster_link * link) {
    const bool is_pp_shared   = params.is_pp_shared;
    const bool is_tg_separate = params.is_tg_separate;

    const std::vector<int> n_pp = params.n_pp;
    const std::vector<int> n_tg = params.n_tg;
    const std::vector<int> n_pl = params.n_pl;

    // warn, don't override (matches the -sm/-mg convention): a data-parallel run wants weights on
    // GPU, but we never silently change a value the user may have set.
    if (params.n_gpu_layers == -1) {
        LOG_WRN("%s: --data-parallel set but -ngl is auto; pass '-ngl all' to keep weights on GPU "
                "(a CPU-only data-parallel run will be slow)\n", __func__);
    }

    // each replica context needs n_seq_max >= the largest parallel-sequence count in the sweep; the
    // pool sizes n_seq_max from params.n_parallel (see orchestrator.h), so set it before building.
    params.n_parallel = n_pl.empty() ? 1 : *std::max_element(n_pl.begin(), n_pl.end());

    // resolve replica placement from the --dp-* flags (precedence in orchestrator_specs_from_params).
    auto pool = orchestrator_make_pool(params, orchestrator_specs_from_params(params));
    if (!pool) {
        LOG_ERR("%s: failed to build the data-parallel replica pool\n", __func__);
        return 1;
    }
    const int n_rep = pool->size();

    // cross-node cluster mode: active only when launched on >1 MPI rank (one rank per node). each
    // rank runs the byte-identical node-local sweep; rank 0 aggregates per-config throughput across
    // nodes. single-rank (or non-MPI build) -> cluster_mode false -> output identical to single-node.
#ifdef LLAMA_ORCH_MPI
    const bool cluster_mode = link && link->size() > 1;
#else
    const bool cluster_mode = false;
    (void) link;
#endif

    const llama_model * model   = pool->model();
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);
    const int32_t       n_batch = params.n_batch;

    // run one copy of `body` on every replica, exactly once, concurrently. the barrier deals each
    // job to a distinct replica; drain() returns the batch span and provides the happens-before for
    // reading per-replica results the body wrote.
    auto scatter = [&](const std::function<bool(llama_context *, int)> & body) {
        auto barrier = std::make_shared<bb_barrier>(n_rep);
        for (int r = 0; r < n_rep; ++r) {
            pool->submit([barrier, &body](llama_context * ctx, int rr) -> bool {
                barrier->arrive_and_wait();
                return body(ctx, rr);
            });
        }
        return pool->drain();
    };

    // probe the runtime context size (identical across replicas); replica 0 is the sole writer and
    // the value is read after drain(), so it is race-free.
    int32_t n_kv_max = 0;
    scatter([&](llama_context * ctx, int rr) -> bool {
        if (rr == 0) {
            n_kv_max = llama_n_ctx(ctx);
        }
        return true;
    });

    // per-replica scratch: one batch + one RNG each, touched only by that replica's worker (race-free
    // without locking). a per-replica RNG replaces the baseline's std::rand(); a shared global rand()
    // called from N workers would be a data race.
    std::vector<llama_batch>      batches(n_rep);
    std::vector<std::minstd_rand> rngs(n_rep);
    for (int r = 0; r < n_rep; ++r) {
        batches[r] = llama_batch_init(n_kv_max, 0, 1);
        rngs[r].seed((unsigned) (r + 1));
    }

    // warm up every replica (mirrors the baseline 16-token warmup).
    scatter([&](llama_context * ctx, int rr) -> bool {
        llama_batch & batch = batches[rr];
        common_batch_clear(batch);
        for (int i = 0; i < 16; ++i) {
            common_batch_add(batch, (llama_token) (rngs[rr]() % n_vocab), i, { 0 }, false);
        }
        return bb_decode_helper(ctx, batch, n_batch, true);
    });

#ifdef LLAMA_ORCH_MPI
    // align the timed-region start across nodes: every node has finished loading + warming up before
    // any node begins the sweep (avoids one node racing ahead while another is still loading).
    if (cluster_mode) {
        link->barrier();
    }
#endif

    // per-config node-local aggregate, collected (not printed) in cluster mode; rank 0 reduces these
    // across nodes after the sweep. config order is deterministic + identical on every node, so the
    // i-th record lines up across ranks.
    struct cluster_record {
        int   pp, tg, pl, n_kv;        // config descriptors (identical across nodes)
        int   pp_tokens, tg_tokens;    // per-replica token counts for this config
        float max_t_pp, max_t_tg, t_e2e;
        int   n_rep, ok_reps;
    };
    std::vector<cluster_record> cluster_records;

    // the per-node table header is printed only on the node-local path; in cluster mode rank 0 prints
    // a single aggregated table after the reduce.
    if (!cluster_mode && !params.batched_bench_output_jsonl) {
        LOG("\n");
        LOG("%s: n_replicas = %d, n_kv_max = %d, n_batch = %d, n_ubatch = %d, flash_attn = %d, is_pp_shared = %d, is_tg_separate = %d, n_gpu_layers = %d\n",
            __func__, n_rep, n_kv_max, params.n_batch, params.n_ubatch, int(params.flash_attn_type), is_pp_shared, is_tg_separate, params.n_gpu_layers);
        LOG("\n");
        LOG("|%4s | %6s | %6s | %4s | %6s | %8s | %8s | %8s | %8s | %8s | %8s | %8s | %9s |\n",
            "REP", "PP", "TG", "B", "N_KV", "T_PP s", "S_PP t/s", "T_TG s", "S_TG t/s", "T s", "S t/s", "T_e2e s", "S_e2e t/s");
        LOG("|%4s-|-%6s-|-%6s-|-%4s-|-%6s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|-%9s-|\n",
            "----", "------", "------", "----", "------", "--------", "--------", "--------", "--------", "--------", "--------", "--------", "---------");
    }

    // one result slot per replica, written only by that replica's job (distinct slot per job).
    struct rep_result { float t_pp = 0.0f; float t_tg = 0.0f; bool ok = false; };

    for (        int i_pp = 0; i_pp < (int) n_pp.size(); ++i_pp) {
        for (    int i_tg = 0; i_tg < (int) n_tg.size(); ++i_tg) {
            for (int i_pl = 0; i_pl < (int) n_pl.size(); ++i_pl) {
                const int pp = n_pp[i_pp];
                const int tg = n_tg[i_tg];
                const int pl = n_pl[i_pl];

                const int n_ctx_req = is_pp_shared ? (params.kv_unified ? pp : pl*pp) + pl*tg : pl*(pp + tg);
                if (n_ctx_req > n_kv_max) {
                    continue;
                }

                std::vector<rep_result> results(n_rep);

                const orchestrator_run_stats st = scatter([&](llama_context * ctx, int rr) -> bool {
                    auto *        mem   = llama_get_memory(ctx);
                    llama_batch & batch = batches[rr];
                    const auto    rand_tok = [&]() -> llama_token { return (llama_token) (rngs[rr]() % n_vocab); };

                    // B1: per-shape warmup (untimed) so one-time GPU costs for this
                    // (pp,tg,pl) shape don't land in the timed measurement. Each replica
                    // warms its own context; KV is reset by the timed section below.
                    {
                        common_batch_clear(batch);
                        for (int j = 0; j < (is_pp_shared ? 1 : pl); ++j) {
                            for (int i = 0; i < pp; ++i) {
                                common_batch_add(batch, rand_tok(), i, { j }, i == pp - 1);
                            }
                        }
                        llama_memory_clear(mem, false);
                        if (!bb_decode_helper(ctx, batch, n_batch, true)) {
                            return false;
                        }
                        for (int i = 0; i < (tg < 2 ? tg : 2); ++i) {
                            common_batch_clear(batch);
                            for (int j = 0; j < pl; ++j) {
                                common_batch_add(batch, rand_tok(), pp + i, { j }, true);
                            }
                            if (!bb_decode_helper(ctx, batch, n_batch, true)) {
                                return false;
                            }
                        }
                    }

                    common_batch_clear(batch);
                    for (int j = 0; j < (is_pp_shared ? 1 : pl); ++j) {
                        for (int i = 0; i < pp; ++i) {
                            common_batch_add(batch, rand_tok(), i, { j }, i == pp - 1);
                        }
                    }

                    llama_memory_clear(mem, false);

                    const auto t_pp_start = ggml_time_us();
                    if (!bb_decode_helper(ctx, batch, n_batch, false)) {
                        return false;
                    }
                    llama_synchronize(ctx);
                    const auto t_pp_end = ggml_time_us();

                    if (is_pp_shared) {
                        for (int32_t i = 1; i < pl; ++i) {
                            llama_memory_seq_cp(mem, 0, i, -1, -1);
                        }

                        if (!params.kv_unified) {
                            // run one dummy token to apply the memory copy
                            common_batch_clear(batch);
                            common_batch_add(batch, rand_tok(), pp + 0, { 0 }, true);
                            if (!bb_decode_helper(ctx, batch, n_batch, true)) {
                                return false;
                            }
                            llama_memory_seq_rm(mem, 0, pp, -1);
                        }
                    }

                    const auto t_tg_start = ggml_time_us();
                    if (is_tg_separate) {
                        for (int j = 0; j < pl; ++j) {
                            for (int i = 0; i < tg; ++i) {
                                common_batch_clear(batch);
                                common_batch_add(batch, rand_tok(), pp + i, { j }, true);
                                if (!bb_decode_helper(ctx, batch, n_batch, true)) {
                                    return false;
                                }
                            }
                        }
                    } else {
                        for (int i = 0; i < tg; ++i) {
                            common_batch_clear(batch);
                            for (int j = 0; j < pl; ++j) {
                                common_batch_add(batch, rand_tok(), pp + i, { j }, true);
                            }
                            if (!bb_decode_helper(ctx, batch, n_batch, true)) {
                                return false;
                            }
                        }
                    }
                    const auto t_tg_end = ggml_time_us();

                    rep_result & out = results[rr];
                    out.t_pp = (t_pp_end - t_pp_start) / 1000000.0f;
                    out.t_tg = (t_tg_end - t_tg_start) / 1000000.0f;
                    out.ok   = true;
                    return true;
                });

                // tokens decoded per replica for this config (the baseline's per-row token counts)
                const int pp_tokens = is_pp_shared ? pp : pl*pp;
                const int tg_tokens = pl*tg;
                const int n_kv      = n_ctx_req;

                size_t ok_reps  = 0;
                float  max_t_pp = 0.0f;
                float  max_t_tg = 0.0f;
                for (int r = 0; r < n_rep; ++r) {
                    if (results[r].ok) {
                        ok_reps++;
                        max_t_pp = std::max(max_t_pp, results[r].t_pp);
                        max_t_tg = std::max(max_t_tg, results[r].t_tg);
                    }
                }
                const float t_agg = max_t_pp + max_t_tg; // max-based concurrent span (slowest lane per phase)
                const float t_e2e = (float) st.seconds;  // drain() span: first dispatch -> last completion

                if (cluster_mode) {
                    // collect this node's aggregate; rank 0 reduces across nodes after the sweep.
                    cluster_records.push_back({pp, tg, pl, n_kv, pp_tokens, tg_tokens,
                                               max_t_pp, max_t_tg, t_e2e, n_rep, (int) ok_reps});
                } else if (params.batched_bench_output_jsonl) {
                    std::string reps_json;
                    for (int r = 0; r < n_rep; ++r) {
                        const float t_pp = results[r].t_pp;
                        const float t_tg = results[r].t_tg;
                        const float t    = t_pp + t_tg;
                        reps_json += string_format(
                            "%s{\"rep\": %d, \"ok\": %s, \"t_pp\": %f, \"speed_pp\": %f, \"t_tg\": %f, \"speed_tg\": %f, \"t\": %f, \"speed\": %f}",
                            r == 0 ? "" : ", ", r, results[r].ok ? "true" : "false",
                            t_pp, t_pp > 0.0f ? pp_tokens / t_pp : 0.0f,
                            t_tg, t_tg > 0.0f ? tg_tokens / t_tg : 0.0f,
                            t,    t    > 0.0f ? (pp_tokens + tg_tokens) / t : 0.0f);
                    }
                    LOG("{\"n_replicas\": %d, \"n_kv_max\": %d, \"n_batch\": %d, \"n_ubatch\": %d, \"flash_attn\": %d, \"is_pp_shared\": %d, \"n_gpu_layers\": %d, "
                        "\"pp\": %d, \"tg\": %d, \"pl\": %d, \"n_kv\": %d, \"n_ok\": %zu, \"replicas\": [%s], "
                        "\"agg_t_pp\": %f, \"agg_speed_pp\": %f, \"agg_t_tg\": %f, \"agg_speed_tg\": %f, \"agg_t\": %f, \"agg_speed\": %f, "
                        "\"e2e_t\": %f, \"e2e_speed\": %f}\n",
                        n_rep, n_kv_max, params.n_batch, params.n_ubatch, int(params.flash_attn_type), is_pp_shared, params.n_gpu_layers,
                        pp, tg, pl, n_kv, ok_reps, reps_json.c_str(),
                        max_t_pp, max_t_pp > 0.0f ? n_rep*pp_tokens / max_t_pp : 0.0f,
                        max_t_tg, max_t_tg > 0.0f ? n_rep*tg_tokens / max_t_tg : 0.0f,
                        t_agg,    t_agg    > 0.0f ? n_rep*(pp_tokens + tg_tokens) / t_agg : 0.0f,
                        t_e2e,    t_e2e    > 0.0f ? n_rep*(pp_tokens + tg_tokens) / t_e2e : 0.0f);
                } else {
                    // per-replica rows: each is a single-GPU number, directly comparable to the baseline
                    for (int r = 0; r < n_rep; ++r) {
                        if (!results[r].ok) {
                            LOG_ERR("%s: replica %d decode failed for pp=%d tg=%d pl=%d\n", __func__, r, pp, tg, pl);
                            continue;
                        }
                        const float t_pp = results[r].t_pp;
                        const float t_tg = results[r].t_tg;
                        const float t    = t_pp + t_tg;
                        char rep_label[8];
                        snprintf(rep_label, sizeof(rep_label), "%d", r);
                        bb_print_row(rep_label, pp, tg, pl, n_kv,
                                     t_pp, t_pp > 0.0f ? pp_tokens / t_pp : 0.0f,
                                     t_tg, t_tg > 0.0f ? tg_tokens / t_tg : 0.0f,
                                     t,    t    > 0.0f ? (pp_tokens + tg_tokens) / t : 0.0f,
                                     false, 0.0f, 0.0f);
                    }
                    // aggregate row: max-based per-phase speeds + the end-to-end (drain-span) speed.
                    // only meaningful when every replica succeeded; a partial aggregate would mislead.
                    if (ok_reps == (size_t) n_rep) {
                        bb_print_row("ALL", pp, tg, pl, n_kv,
                                     max_t_pp, max_t_pp > 0.0f ? n_rep*pp_tokens / max_t_pp : 0.0f,
                                     max_t_tg, max_t_tg > 0.0f ? n_rep*tg_tokens / max_t_tg : 0.0f,
                                     t_agg,    t_agg    > 0.0f ? n_rep*(pp_tokens + tg_tokens) / t_agg : 0.0f,
                                     true, t_e2e, t_e2e > 0.0f ? n_rep*(pp_tokens + tg_tokens) / t_e2e : 0.0f);
                    } else {
                        LOG_ERR("%s: pp=%d tg=%d pl=%d: only %zu/%d replicas succeeded; no aggregate row\n",
                                __func__, pp, tg, pl, ok_reps, n_rep);
                    }
                }
            }
        }
    }

    for (int r = 0; r < n_rep; ++r) {
        llama_batch_free(batches[r]);
    }

#ifdef LLAMA_ORCH_MPI
    if (cluster_mode) {
        // each node serializes its per-config aggregate (one line per config, in the shared config
        // order) into a blob; gather() delivers all blobs to rank 0. kilobytes total, once per run.
        std::string blob;
        for (const cluster_record & r : cluster_records) {
            blob += string_format("%.9g %.9g %.9g %d %d\n", r.max_t_pp, r.max_t_tg, r.t_e2e, r.n_rep, r.ok_reps);
        }
        const std::vector<std::string> gathered = link->gather(blob);

        if (link->is_head()) {
            const int n_nodes = link->size();
            const int n_cfg   = (int) cluster_records.size();

            // parse each node's blob back into per-config timings, indexed [node][cfg].
            struct node_cfg { float max_t_pp, max_t_tg, t_e2e; int n_rep, ok_reps; };
            std::vector<std::vector<node_cfg>> per_node(n_nodes);
            bool parse_ok = true;
            for (int nd = 0; nd < n_nodes; ++nd) {
                std::stringstream ss(gathered[nd]);
                std::string line;
                while (std::getline(ss, line)) {
                    if (line.empty()) {
                        continue;
                    }
                    node_cfg nc{};
                    if (std::sscanf(line.c_str(), "%f %f %f %d %d",
                                    &nc.max_t_pp, &nc.max_t_tg, &nc.t_e2e, &nc.n_rep, &nc.ok_reps) != 5) {
                        parse_ok = false;
                        break;
                    }
                    per_node[nd].push_back(nc);
                }
                if ((int) per_node[nd].size() != n_cfg) {
                    parse_ok = false;
                }
            }
            if (!parse_ok) {
                LOG_ERR("%s: cluster reduce failed: node aggregate blobs did not parse to %d configs\n",
                        __func__, n_cfg);
                return 1;
            }

            if (!params.batched_bench_output_jsonl) {
                LOG("\n");
                LOG("%s: CLUSTER mode: n_nodes = %d, replicas/node = %d, total replicas = %d, n_kv_max = %d, n_batch = %d\n",
                    __func__, n_nodes, n_rep, n_nodes * n_rep, n_kv_max, params.n_batch);
                LOG("\n");
                LOG("|%4s | %6s | %6s | %4s | %6s | %8s | %8s | %8s | %8s | %8s | %8s | %8s | %9s |\n",
                    "NODE", "PP", "TG", "B", "N_KV", "T_PP s", "S_PP t/s", "T_TG s", "S_TG t/s", "T s", "S t/s", "T_e2e s", "S_e2e t/s");
                LOG("|%4s-|-%6s-|-%6s-|-%4s-|-%6s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|-%9s-|\n",
                    "----", "------", "------", "----", "------", "--------", "--------", "--------", "--------", "--------", "--------", "--------", "---------");
            }

            for (int c = 0; c < n_cfg; ++c) {
                const cluster_record & cr = cluster_records[c];
                const int   pp_tokens = cr.pp_tokens;
                const int   tg_tokens = cr.tg_tokens;
                const int   tot_tokens = pp_tokens + tg_tokens;

                // cluster totals: sum tokens across nodes, take the slowest node per phase (mirrors the
                // node-local ALL row, lifted one level: replica->node, n_rep->total replicas).
                int   tot_rep = 0, tot_ok = 0;
                float cl_t_pp = 0.0f, cl_t_tg = 0.0f, cl_t_e2e = 0.0f;
                for (int nd = 0; nd < n_nodes; ++nd) {
                    const node_cfg & nc = per_node[nd][c];
                    tot_rep += nc.n_rep;
                    tot_ok  += nc.ok_reps;
                    cl_t_pp  = std::max(cl_t_pp,  nc.max_t_pp);
                    cl_t_tg  = std::max(cl_t_tg,  nc.max_t_tg);
                    cl_t_e2e = std::max(cl_t_e2e, nc.t_e2e);
                }
                const float cl_t_agg = cl_t_pp + cl_t_tg;

                if (params.batched_bench_output_jsonl) {
                    std::string nodes_json;
                    for (int nd = 0; nd < n_nodes; ++nd) {
                        const node_cfg & nc = per_node[nd][c];
                        nodes_json += string_format(
                            "%s{\"node\": %d, \"n_rep\": %d, \"n_ok\": %d, \"t_pp\": %f, \"speed_pp\": %f, \"t_tg\": %f, \"speed_tg\": %f, \"t_e2e\": %f, \"speed_e2e\": %f}",
                            nd == 0 ? "" : ", ", nd, nc.n_rep, nc.ok_reps,
                            nc.max_t_pp, nc.max_t_pp > 0.0f ? nc.n_rep*pp_tokens / nc.max_t_pp : 0.0f,
                            nc.max_t_tg, nc.max_t_tg > 0.0f ? nc.n_rep*tg_tokens / nc.max_t_tg : 0.0f,
                            nc.t_e2e,    nc.t_e2e    > 0.0f ? nc.n_rep*tot_tokens / nc.t_e2e   : 0.0f);
                    }
                    LOG("{\"cluster\": true, \"n_nodes\": %d, \"replicas_per_node\": %d, \"total_replicas\": %d, \"n_ok\": %d, "
                        "\"pp\": %d, \"tg\": %d, \"pl\": %d, \"n_kv\": %d, \"nodes\": [%s], "
                        "\"cl_t_pp\": %f, \"cl_speed_pp\": %f, \"cl_t_tg\": %f, \"cl_speed_tg\": %f, \"cl_t\": %f, \"cl_speed\": %f, "
                        "\"cl_t_e2e\": %f, \"cl_speed_e2e\": %f}\n",
                        n_nodes, n_rep, tot_rep, tot_ok,
                        cr.pp, cr.tg, cr.pl, cr.n_kv, nodes_json.c_str(),
                        cl_t_pp,  cl_t_pp  > 0.0f ? tot_rep*pp_tokens / cl_t_pp   : 0.0f,
                        cl_t_tg,  cl_t_tg  > 0.0f ? tot_rep*tg_tokens / cl_t_tg   : 0.0f,
                        cl_t_agg, cl_t_agg > 0.0f ? tot_rep*tot_tokens / cl_t_agg : 0.0f,
                        cl_t_e2e, cl_t_e2e > 0.0f ? tot_rep*tot_tokens / cl_t_e2e : 0.0f);
                } else {
                    // one row per node (each == that node's single-node ALL row), then the CL total.
                    for (int nd = 0; nd < n_nodes; ++nd) {
                        const node_cfg & nc = per_node[nd][c];
                        char node_label[8];
                        snprintf(node_label, sizeof(node_label), "%d", nd);
                        bb_print_row(node_label, cr.pp, cr.tg, cr.pl, cr.n_kv,
                                     nc.max_t_pp, nc.max_t_pp > 0.0f ? nc.n_rep*pp_tokens / nc.max_t_pp : 0.0f,
                                     nc.max_t_tg, nc.max_t_tg > 0.0f ? nc.n_rep*tg_tokens / nc.max_t_tg : 0.0f,
                                     nc.max_t_pp + nc.max_t_tg,
                                     (nc.max_t_pp + nc.max_t_tg) > 0.0f ? nc.n_rep*tot_tokens / (nc.max_t_pp + nc.max_t_tg) : 0.0f,
                                     true, nc.t_e2e, nc.t_e2e > 0.0f ? nc.n_rep*tot_tokens / nc.t_e2e : 0.0f);
                    }
                    bb_print_row("CL", cr.pp, cr.tg, cr.pl, cr.n_kv,
                                 cl_t_pp,  cl_t_pp  > 0.0f ? tot_rep*pp_tokens / cl_t_pp   : 0.0f,
                                 cl_t_tg,  cl_t_tg  > 0.0f ? tot_rep*tg_tokens / cl_t_tg   : 0.0f,
                                 cl_t_agg, cl_t_agg > 0.0f ? tot_rep*tot_tokens / cl_t_agg : 0.0f,
                                 true, cl_t_e2e, cl_t_e2e > 0.0f ? tot_rep*tot_tokens / cl_t_e2e : 0.0f);
                }
            }
        }
    }
#endif

    return 0;
}

// satisfies -Wmissing-declarations
int llama_batched_bench(int argc, char ** argv);

int llama_batched_bench(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // initialize the cross-node link first (MPI_Init may consume its own argv entries). A non-MPI
    // build or a single-rank launch leaves `link` benign (size()==1) -> no behaviour change.
    cluster_link * link = nullptr;
#ifdef LLAMA_ORCH_MPI
    std::unique_ptr<cluster_link> cluster = cluster_link::init(&argc, &argv);
    link = cluster.get();
#endif

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_BENCH, print_usage)) {
        return 1;
    }

    int is_pp_shared   = params.is_pp_shared;
    int is_tg_separate = params.is_tg_separate;

    std::vector<int> n_pp = params.n_pp;
    std::vector<int> n_tg = params.n_tg;
    std::vector<int> n_pl = params.n_pl;

    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

    // data-parallel path (S10): n replicas run the benchmark concurrently, throughput aggregated.
    // active when the --dp-* flags resolve to a non-trivial placement (more than one replica, or a
    // single replica split across >1 GPU); a lone single-GPU replica falls through to the unchanged
    // single-context path below.
#ifdef LLAMA_ORCH_MPI
    // cross-node aggregation lives in the data-parallel path; warn if launched multi-rank without -dp
    // (each rank would otherwise run an independent single-context bench and print its own table).
    if (link && link->size() > 1 && !orchestrator_dp_active(params) && link->is_head()) {
        LOG_WRN("%s: launched on %d ranks but --data-parallel is not active; cluster aggregation needs "
                "-dp. Each rank will run independently.\n", __func__, link->size());
    }
#endif

    if (orchestrator_dp_active(params)) {
        const int rc = batched_bench_dp(params, link);
        llama_backend_free();
        return rc;
    }

    // initialize the model

    llama_model_params model_params = common_model_params_to_llama(params);

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    llama_context_params ctx_params = common_context_params_to_llama(params);

    // ensure enough sequences are available
    ctx_params.n_seq_max = n_pl.empty() ? 1 : *std::max_element(n_pl.begin(), n_pl.end());

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    const auto get_token_rand = [n_vocab]() -> llama_token {
        return std::rand() % n_vocab;
    };

    auto * mem = llama_get_memory(ctx);

    const int32_t n_kv_max = llama_n_ctx(ctx);

    llama_batch batch = llama_batch_init(n_kv_max, 0, 1);

    // decode in batches of ctx_params.n_batch tokens
    auto decode_helper = [](llama_context * ctx, llama_batch & batch, int32_t n_batch, bool synchronize) {
        for (int32_t i = 0; i < batch.n_tokens; i += n_batch) {
            const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
            };

            const int ret = llama_decode(ctx, batch_view);
            if (ret != 0) {
                LOG_ERR("failed to decode the batch, n_batch = %d, ret = %d\n", n_batch, ret);
                return false;
            }

            if (synchronize) {
                llama_synchronize(ctx);
            }
        }

        return true;
    };

    // warm up
    {
        for (int i = 0; i < 16; ++i) {
            common_batch_add(batch, get_token_rand(), i, { 0 }, false);
        }

        if (!decode_helper(ctx, batch, ctx_params.n_batch, true)) {
            LOG_ERR("%s: llama_decode() failed\n", __func__);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
    }

    if (!params.batched_bench_output_jsonl) {
        LOG("\n");
        LOG("%s: n_kv_max = %d, n_batch = %d, n_ubatch = %d, flash_attn = %d, is_pp_shared = %d, is_tg_separate = %d, n_gpu_layers = %d, n_threads = %u, n_threads_batch = %u\n", __func__, n_kv_max, params.n_batch, params.n_ubatch, int(params.flash_attn_type), is_pp_shared, is_tg_separate, params.n_gpu_layers, ctx_params.n_threads, ctx_params.n_threads_batch);
        LOG("\n");
        LOG("|%6s | %6s | %4s | %6s | %8s | %8s | %8s | %8s | %8s | %8s |\n", "PP", "TG", "B", "N_KV", "T_PP s", "S_PP t/s", "T_TG s", "S_TG t/s", "T s", "S t/s");
        LOG("|%6s-|-%6s-|-%4s-|-%6s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|\n", "------", "------", "----", "------", "--------", "--------", "--------", "--------", "--------", "--------");
    }

    for (        int i_pp = 0; i_pp < (int) n_pp.size(); ++i_pp) {
        for (    int i_tg = 0; i_tg < (int) n_tg.size(); ++i_tg) {
            for (int i_pl = 0; i_pl < (int) n_pl.size(); ++i_pl) {
                const int pp = n_pp[i_pp];
                const int tg = n_tg[i_tg];
                const int pl = n_pl[i_pl];

                const int n_ctx_req = is_pp_shared ? (params.kv_unified ? pp : pl*pp) + pl*tg : pl*(pp + tg);

                if (n_ctx_req > n_kv_max) {
                    continue;
                }

                // B1: per-shape warmup. Run one throwaway PP (+ a couple TG steps) at
                // THIS (pp,tg,pl) shape, untimed, so one-time GPU costs (CUDA graph
                // capture, cuBLASLt heuristics, workspace alloc) for the shape don't
                // land inside the first timed cell that uses it. KV is reset by the
                // timed section's llama_memory_clear below.
                {
                    common_batch_clear(batch);
                    for (int j = 0; j < (is_pp_shared ? 1 : pl); ++j) {
                        for (int i = 0; i < pp; ++i) {
                            common_batch_add(batch, get_token_rand(), i, { j }, i == pp - 1);
                        }
                    }
                    llama_memory_clear(mem, false);
                    if (!decode_helper(ctx, batch, ctx_params.n_batch, true)) {
                        LOG_ERR("%s: llama_decode() failed\n", __func__);
                        llama_free(ctx);
                        llama_model_free(model);
                        return 1;
                    }
                    for (int i = 0; i < (tg < 2 ? tg : 2); ++i) {
                        common_batch_clear(batch);
                        for (int j = 0; j < pl; ++j) {
                            common_batch_add(batch, get_token_rand(), pp + i, { j }, true);
                        }
                        if (!decode_helper(ctx, batch, ctx_params.n_batch, true)) {
                            LOG_ERR("%s: llama_decode() failed\n", __func__);
                            llama_free(ctx);
                            llama_model_free(model);
                            return 1;
                        }
                    }
                }

                common_batch_clear(batch);

                for (int j = 0; j < (is_pp_shared ? 1 : pl); ++j) {
                    for (int i = 0; i < pp; ++i) {
                        common_batch_add(batch, get_token_rand(), i, { j }, i == pp - 1);
                    }
                }

                llama_memory_clear(mem, false);

                const auto t_pp_start = ggml_time_us();

                if (!decode_helper(ctx, batch, ctx_params.n_batch, false)) {
                    LOG_ERR("%s: llama_decode() failed\n", __func__);
                    llama_free(ctx);
                    llama_model_free(model);
                    return 1;
                }

                llama_synchronize(ctx);

                const auto t_pp_end = ggml_time_us();

                if (is_pp_shared) {
                    for (int32_t i = 1; i < pl; ++i) {
                        llama_memory_seq_cp(mem, 0, i, -1, -1);
                    }

                    if (!params.kv_unified) {
                        // run one dummy token to apply the memory copy
                        common_batch_clear(batch);
                        common_batch_add(batch, get_token_rand(), pp + 0, { 0 }, true);
                        if (!decode_helper(ctx, batch, ctx_params.n_batch, true)) {
                            LOG_ERR("%s: llama_decode() failed\n", __func__);
                            llama_free(ctx);
                            llama_model_free(model);
                            return 1;
                        }
                        llama_memory_seq_rm(mem, 0, pp, -1);
                    }
                }

                const auto t_tg_start = ggml_time_us();

                if (is_tg_separate) {
                    // decode pattern:
                    // 0 0 0 ... 1 1 1 ... 2 2 2 ... 3 3 3 ...
                    for (int j = 0; j < pl; ++j) {
                        for (int i = 0; i < tg; ++i) {
                            common_batch_clear(batch);

                            common_batch_add(batch, get_token_rand(), pp + i, { j }, true);

                            if (!decode_helper(ctx, batch, ctx_params.n_batch, true)) {
                                LOG_ERR("%s: llama_decode() failed\n", __func__);
                                llama_free(ctx);
                                llama_model_free(model);
                                return 1;
                            }
                        }
                    }
                } else {
                    // decode pattern:
                    // 0123 0123 0123 ...
                    for (int i = 0; i < tg; ++i) {
                        common_batch_clear(batch);

                        for (int j = 0; j < pl; ++j) {
                            common_batch_add(batch, get_token_rand(), pp + i, { j }, true);
                        }

                        if (!decode_helper(ctx, batch, ctx_params.n_batch, true)) {
                            LOG_ERR("%s: llama_decode() failed\n", __func__);
                            llama_free(ctx);
                            llama_model_free(model);
                            return 1;
                        }
                    }
                }

                const auto t_tg_end = ggml_time_us();

                const int32_t n_kv = n_ctx_req;

                const float t_pp = (t_pp_end - t_pp_start) / 1000000.0f;
                const float t_tg = (t_tg_end - t_tg_start) / 1000000.0f;
                const float t    = t_pp + t_tg;

                const float speed_pp = is_pp_shared ? pp / t_pp : pl*pp / t_pp;
                const float speed_tg = pl*tg / t_tg;
                const float speed    = ((is_pp_shared ? pp : pl*pp) + pl*tg) / t;

                if(params.batched_bench_output_jsonl) {
                    LOG(
                        "{\"n_kv_max\": %d, \"n_batch\": %d, \"n_ubatch\": %d, \"flash_attn\": %d, \"is_pp_shared\": %d, \"n_gpu_layers\": %d, \"n_threads\": %u, \"n_threads_batch\": %u, "
                        "\"pp\": %d, \"tg\": %d, \"pl\": %d, \"n_kv\": %d, \"t_pp\": %f, \"speed_pp\": %f, \"t_tg\": %f, \"speed_tg\": %f, \"t\": %f, \"speed\": %f}\n",
                        n_kv_max, params.n_batch, params.n_ubatch, int(params.flash_attn_type), params.is_pp_shared, params.n_gpu_layers, ctx_params.n_threads, ctx_params.n_threads_batch,
                        pp, tg, pl, n_kv, t_pp, speed_pp, t_tg, speed_tg, t, speed
                    );
                } else {
                    LOG("|%6d | %6d | %4d | %6d | %8.3f | %8.2f | %8.3f | %8.2f | %8.3f | %8.2f |\n", pp, tg, pl, n_kv, t_pp, speed_pp, t_tg, speed_tg, t, speed);
                }
            }
        }
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_batch_free(batch);

    llama_free(ctx);
    llama_model_free(model);

    llama_backend_free();

    return 0;
}
