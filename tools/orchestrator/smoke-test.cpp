// S5 smoke test: load N replicas on CPU, check the accessors, tear down cleanly.
// not part of the real tools - built only with -DLLAMA_BUILD_TESTS=ON.
#include "orchestrator.h"

#include "common.h"
#include "llama.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// parse a comma-separated device list like "0,1,2,3" into {0,1,2,3}
static std::vector<int> parse_devices(const std::string & s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) {
            out.push_back(std::stoi(tok));
        }
    }
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <model.gguf> [devices] [n_gpu_layers]\n"
                "  devices:      comma-separated GPU indices (default \"0,0\" = 2 replicas on GPU 0)\n"
                "  n_gpu_layers: layers to offload per replica (default 99 = all; 0 = keep on CPU)\n",
                argv[0]);
        return 1;
    }

    common_init();        // set up logging so the orchestrator's messages show
    llama_backend_init(); // global backend init (matches every tool)

    common_params params;
    params.model.path   = argv[1];
    params.n_gpu_layers = (argc >= 4) ? atoi(argv[3]) : 99; // offload all by default
    params.n_parallel   = 4; // A1: build each context with 4 sequence slots (n_seq_max),
                             // so continuous batching can be added later without a rebuild.
                             // make_pool logs the resulting n_seq_max.

    std::vector<int> devices = parse_devices((argc >= 3) ? argv[2] : "0,0");
    const int n = (int) devices.size();
    if (n == 0) {
        fprintf(stderr, "[smoke] FAIL: empty device list\n");
        return 1;
    }

    printf("\n[smoke] building %d replica(s) from '%s' (n_gpu_layers=%d)\n",
           n, params.model.path.c_str(), params.n_gpu_layers);

    auto pool = orchestrator_make_pool(params, devices);
    if (!pool) {
        fprintf(stderr, "[smoke] FAIL: orchestrator_make_pool returned null\n");
        return 1;
    }

    bool ok = true;

    printf("[smoke] size() = %d (expected %d)\n", pool->size(), n);
    ok &= (pool->size() == n);

    for (int i = 0; i < pool->size(); ++i) {
        orchestrator_replica_info info = pool->at(i);
        printf("[smoke] at(%d): index=%d device=%d (expected device=%d)\n",
               i, info.index, info.device, devices[i]);
        ok &= (info.index == i);
        ok &= (info.device == devices[i]);
    }

    const llama_model * m = pool->model();
    if (m) {
        const llama_vocab * vocab = llama_model_get_vocab(m);
        printf("[smoke] model() ok, vocab size = %d\n", llama_vocab_n_tokens(vocab));
    } else {
        printf("[smoke] FAIL: model() returned null\n");
        ok = false;
    }

    // S6: serial submit/drain. K distinct-slot jobs all succeed; then one false
    // and one throwing job both register as failures.
    {
        const int K = 8;
        std::vector<int> results(K, -1);
        for (int i = 0; i < K; ++i) {
            pool->submit([&results, i](llama_context *, int) {
                results[i] = i; // distinct slot per job, no shared accumulator
                return true;
            });
        }
        orchestrator_run_stats st = pool->drain();
        printf("[smoke] drain: n_jobs=%zu n_failed=%zu ok=%d\n", st.n_jobs, st.n_failed, st.ok());
        ok &= (st.n_jobs == (size_t) K) && (st.n_failed == 0) && st.ok();
        for (int i = 0; i < K; ++i) {
            ok &= (results[i] == i);
        }

        pool->submit([](llama_context *, int) { return false; });
        pool->submit([](llama_context *, int) -> bool { throw std::runtime_error("boom"); });
        orchestrator_run_stats st2 = pool->drain();
        printf("[smoke] drain2: n_jobs=%zu n_failed=%zu\n", st2.n_jobs, st2.n_failed);
        ok &= (st2.n_jobs == 2) && (st2.n_failed == 2) && !st2.ok();
    }

    // S7: concurrent dispatch. K >> n_replicas lightweight jobs run across the
    // worker threads; verify the same accounting holds under real concurrency.
    {
        const int K = 64;
        std::vector<int> results(K, -1);
        std::vector<int> per_replica(pool->size(), 0); // worker r is the only writer of slot r
        for (int i = 0; i < K; ++i) {
            pool->submit([&results, &per_replica, i](llama_context *, int r) {
                // brief work so workers actually overlap - this is what gives the
                // TSan gate real concurrency to inspect, not a timing hack.
                std::this_thread::sleep_for(std::chrono::microseconds(200));
                results[i] = i;      // distinct slot per job
                per_replica[r]++;    // distinct slot per worker
                return true;
            });
        }
        orchestrator_run_stats st = pool->drain();
        printf("[smoke] concurrent drain: n_jobs=%zu n_failed=%zu seconds=%.6f\n",
               st.n_jobs, st.n_failed, st.seconds);
        // seconds >= 0 is the real invariant (a negative span would mean the clock
        // ran backwards); magnitude is proven by the real decode workloads later.
        ok &= (st.n_jobs == (size_t) K) && (st.n_failed == 0) && st.ok() && (st.seconds >= 0.0);
        for (int i = 0; i < K; ++i) {
            ok &= (results[i] == i); // every job ran exactly once
        }
        size_t total = 0;
        for (int r = 0; r < pool->size(); ++r) {
            printf("[smoke]   replica %d ran %d job(s)\n", r, per_replica[r]);
            total += per_replica[r];
        }
        ok &= (total == (size_t) K); // jobs distributed across replicas, none lost

        // C: per-job latency distribution is populated and monotone.
        printf("[smoke] latency ms: min=%.3f p50=%.3f p95=%.3f p99=%.3f max=%.3f mean=%.3f\n",
               st.lat_ms_min, st.lat_ms_p50, st.lat_ms_p95, st.lat_ms_p99, st.lat_ms_max, st.lat_ms_mean);
        ok &= (st.lat_ms_min <= st.lat_ms_p50) && (st.lat_ms_p50 <= st.lat_ms_p95) &&
              (st.lat_ms_p95 <= st.lat_ms_p99) && (st.lat_ms_p99 <= st.lat_ms_max) &&
              (st.lat_ms_mean > 0.0);
    }

    // B: best-effort timeout. arm a short deadline; a job that polls deadline_expired()
    // should see it flip and report failure. (The abort callback itself needs a real
    // llama_decode, so it is exercised on the cluster, not here.)
    {
        pool->set_timeout_ms(5);
        std::atomic<bool> saw_expiry{false};
        orchestrator_pool * pp = pool.get();
        pool->submit([pp, &saw_expiry](llama_context *, int r) {
            // spin until the deadline trips, capped at ~2 s so a bug can't hang the test.
            const auto t0 = std::chrono::steady_clock::now();
            while (!pp->deadline_expired(r)) {
                if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(2)) {
                    return true; // deadline never fired - assertion below will flag it
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            saw_expiry = true;
            return false; // an over-deadline job reports failure
        });
        orchestrator_run_stats st = pool->drain();
        printf("[smoke] timeout drain: n_jobs=%zu n_failed=%zu saw_expiry=%d\n",
               st.n_jobs, st.n_failed, (int) saw_expiry.load());
        ok &= (st.n_jobs == 1) && (st.n_failed == 1) && saw_expiry.load();
        pool->set_timeout_ms(0); // disarm for any later batches
    }

    // A2: spec-based factory + device-group metadata. the main pool was built via the
    // vector<int> convenience wrapper, which maps each index to a single-device spec, so
    // its at() metadata exercises the new fields. assert they round-trip. the true
    // multi-GPU split path (devices.size() >= 2) needs >= 2 GPUs and so is exercised on the
    // cluster, not here - a 1-GPU Mac correctly rejects a 2-device spec at load time.
    {
        bool meta_ok = true;
        for (int i = 0; i < pool->size(); ++i) {
            orchestrator_replica_info info = pool->at(i);
            meta_ok &= (info.devices.size() == 1) && (info.devices[0] == devices[i]) &&
                       (info.split_mode == LLAMA_SPLIT_MODE_NONE);
        }
        printf("[smoke] A2: single-device spec metadata round-trips: %d\n", (int) meta_ok);
        ok &= meta_ok;
    }

    printf("[smoke] tearing down pool...\n");
    pool.reset(); // free all replicas now; should be clean

    llama_backend_free();

    printf("[smoke] %s\n\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
