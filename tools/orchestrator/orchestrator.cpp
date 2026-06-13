#include "orchestrator.h"

#include "common.h"    // common_params, common_model_params_to_llama, common_context_params_to_llama
#include "llama-cpp.h" // llama_model_ptr, llama_context_ptr (RAII teardown)
#include "log.h"       // LOG_INF / LOG_ERR

#include <algorithm>          // std::sort for the latency percentiles
#include <atomic>             // per-replica deadline + pool-wide timeout
#include <condition_variable> // cv_work / cv_space / cv_done
#include <deque>              // job queue
#include <memory>             // std::unique_ptr / std::make_unique for the deadline atomics
#include <mutex>              // queue + scheduler guard
#include <string>             // device-group labels for logging
#include <thread>             // one worker per replica
#include <utility>            // std::move
#include <vector>             // latency samples + device-group handles

// private implementation

struct orchestrator_pool::impl {
    impl() = default;
    ~impl(); // stops + joins workers before any member (the contexts) is destroyed

    // one full model + its context per replica. declaration order matters: ctx
    // is declared after model so it is destroyed first (bottom-to-top), tearing
    // down the context before the model it borrows from - same ordering rule as
    // common_init_result::impl. `deadline` is declared before both so it is the LAST
    // of the three to be destroyed: the ctx holds an abort callback pointing at it, so
    // the atomic must outlive the ctx.
    struct replica {
        int               device = -1;                          // first/main device (== devices[0])
        std::vector<int>  devices;                              // every GPU this replica occupies
        llama_split_mode  split_mode = LLAMA_SPLIT_MODE_NONE;   // how the model is split across them

        // per-job deadline in us (0 = disarmed). heap atomic so its address stays stable
        // when the replicas vector grows; the abort callback captures &*deadline.
        std::unique_ptr<std::atomic<int64_t>> deadline;

        llama_model_ptr   model;       // owned; freed via llama_model_free
        llama_context_ptr ctx;         // owned; freed via llama_free, before model
    };

    std::vector<replica> replicas;

    // bounded FIFO job queue + the scheduler that drains it across the workers.
    static constexpr size_t         queue_cap = 1024;
    std::deque<orchestrator_job_fn> queue;
    std::mutex                      mtx;

    // three single-purpose condition variables on `mtx`:
    //   cv_work  - workers sleep here until a job arrives or stop is set
    //   cv_space - submit() sleeps here while the queue is full (backpressure)
    //   cv_done  - drain() sleeps here until the queue empties and nothing is in flight
    std::condition_variable cv_work;
    std::condition_variable cv_space;
    std::condition_variable cv_done;

    bool   stop       = false; // set once at teardown; tells workers to exit
    size_t n_inflight = 0;     // jobs popped by a worker but not yet finished

    // best-effort per-job timeout in us (0 = disabled). read by each worker when it arms
    // a job's deadline; atomic so set_timeout_ms() can change it without the queue lock.
    std::atomic<int64_t> timeout_us{0};

    // per-batch accounting, reset by each drain() (all guarded by mtx)
    size_t              n_jobs     = 0;
    size_t              n_failed   = 0;
    size_t              n_started  = 0;  // jobs dispatched this batch; the first stamps t_first_us
    int64_t             t_first_us = 0;  // ggml_time_us() at the batch's first dispatch
    int64_t             t_last_us  = 0;  // ggml_time_us() at the last completion so far
    std::vector<double> latencies;       // per-job service latency (ms) this batch, for percentiles

    // declared last so they are torn down first; ~impl() has already joined them
    // by then, so the std::thread destructors see non-joinable threads.
    std::vector<std::thread> workers;

    void worker_loop(int r); // body of worker r, bound to replicas[r].ctx
};

// registered on every replica context. returns true (abort the in-flight decode) once
// the armed deadline has passed. `data` points at that replica's atomic deadline (us);
// 0 means disarmed. polled between graph ops on CPU/Metal; CUDA does not poll it, so GPU
// decode jobs must also check orchestrator_pool::deadline_expired() between steps.
static bool orchestrator_abort_cb(void * data) {
    const auto * deadline = static_cast<const std::atomic<int64_t> *>(data);
    const int64_t d = deadline->load(std::memory_order_relaxed);
    return d != 0 && ggml_time_us() >= d;
}

// each worker owns one ctx for its whole life and pulls jobs off the shared
// queue. jobs run OUTSIDE the lock, so replicas execute concurrently.
void orchestrator_pool::impl::worker_loop(int r) {
    llama_context *       ctx      = replicas[r].ctx.get();
    std::atomic<int64_t> & deadline = *replicas[r].deadline; // stable for the worker's life
    for (;;) {
        orchestrator_job_fn job;
        int64_t             job_start_us = 0;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv_work.wait(lock, [&] { return stop || !queue.empty(); });
            if (stop && queue.empty()) {
                return; // shutdown: finish the queue first, then exit
            }
            job = std::move(queue.front());
            queue.pop_front();
            job_start_us = ggml_time_us(); // dispatch time of this job
            if (n_started++ == 0) {
                t_first_us = job_start_us; // first dispatch of this batch
            }
            n_inflight++;
            cv_space.notify_one(); // a queue slot just freed
        }
        // arm the watchdog for this job (0 timeout leaves the deadline disarmed). the
        // abort callback and deadline_expired() both read this atomic.
        const int64_t t = timeout_us.load(std::memory_order_relaxed);
        deadline.store(t > 0 ? job_start_us + t : 0, std::memory_order_relaxed);

        bool ok = false;
        try {
            ok = job(ctx, r);
        } catch (...) {
            ok = false; // a throwing job is a failure, never a process abort
        }
        deadline.store(0, std::memory_order_relaxed); // disarm before the next job

        {
            // re-locking here publishes the job's writes (e.g. results[i], done
            // outside the lock) to drain(): this unlock synchronizes-with the
            // lock drain() takes, so the writes happen-before drain() returns.
            std::lock_guard<std::mutex> lock(mtx);
            n_jobs++;
            if (!ok) {
                n_failed++;
            }
            const int64_t now = ggml_time_us();
            t_last_us = now; // last completion so far
            latencies.push_back((now - job_start_us) / 1e3); // service latency, ms
            n_inflight--;
            if (queue.empty() && n_inflight == 0) {
                cv_done.notify_all(); // batch finished; wake any waiting drain()
            }
        }
    }
}

// stop the workers and join them BEFORE any member is destroyed - in particular
// before the contexts a worker might still be using. running here (not after the
// members) also avoids std::thread's joinable-on-destroy std::terminate.
orchestrator_pool::impl::~impl() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        stop = true;
    }
    cv_work.notify_all();
    cv_space.notify_all(); // release any submit() parked on a full queue
    for (auto & t : workers) {
        if (t.joinable()) {
            t.join();
        }
    }
}

// construction / teardown

// private; constructed only via orchestrator_make_pool(). allocates the impl so
// every pool has valid internal state for its whole lifetime.
orchestrator_pool::orchestrator_pool() : pimpl(std::make_unique<impl>()) {}

// destroying pimpl runs impl::~impl(), which stops + joins the workers before
// the contexts they use are torn down.
orchestrator_pool::~orchestrator_pool() = default;

// queries

// number of replicas (== devices.size() passed to make_pool)
int orchestrator_pool::size() const {
    return (int) pimpl->replicas.size();
}

// metadata for replica i (index + device group + split mode); exposes no live pointers
orchestrator_replica_info orchestrator_pool::at(int i) const {
    // an out-of-range index is a caller bug, not a runtime condition - fail loud.
    GGML_ASSERT(i >= 0 && (size_t) i < pimpl->replicas.size());
    const impl::replica & rep = pimpl->replicas[i];
    orchestrator_replica_info info;
    info.index      = i;
    info.device     = rep.device;
    info.devices    = rep.devices;
    info.split_mode = rep.split_mode;
    return info;
}

// read-only model handle for metadata queries (n_vocab, ...); never a context.
// all replicas are identical copies, so replica 0's model speaks for all of them.
const llama_model * orchestrator_pool::model() const {
    if (pimpl->replicas.empty()) {
        return nullptr;
    }
    return pimpl->replicas[0].model.get();
}

// watchdog

// set the best-effort per-job timeout (0 disables). stored in us; takes effect on each
// worker's next job. lock-free: the workers read it with a relaxed atomic load.
void orchestrator_pool::set_timeout_ms(int64_t timeout_ms) {
    pimpl->timeout_us.store(timeout_ms > 0 ? timeout_ms * 1000 : 0, std::memory_order_relaxed);
}

// true once replica i's current job has passed its armed deadline (0 = disarmed). a long
// decode job polls this between llama_decode steps to stop on GPU, where the abort
// callback is not honoured.
bool orchestrator_pool::deadline_expired(int i) const {
    GGML_ASSERT(i >= 0 && (size_t) i < pimpl->replicas.size());
    const int64_t d = pimpl->replicas[i].deadline->load(std::memory_order_relaxed);
    return d != 0 && ggml_time_us() >= d;
}

// scheduling

// enqueue a job for the next free replica; blocks while the queue is full
void orchestrator_pool::submit(orchestrator_job_fn job) {
    std::unique_lock<std::mutex> lock(pimpl->mtx);
    GGML_ASSERT(!pimpl->stop); // no submits once teardown has begun
    pimpl->cv_space.wait(lock, [&] {
        return pimpl->queue.size() < impl::queue_cap || pimpl->stop;
    });
    // hardening: if we were woken by teardown (not free space), drop the job
    // rather than push into a dying pool. unreachable for a single-producer
    // caller (it owns the pool's lifetime); guards a concurrent destroy.
    if (pimpl->stop) {
        return;
    }
    pimpl->queue.push_back(std::move(job));
    pimpl->cv_work.notify_one();
}

// block until every job submitted since the last drain() has finished; report
// that batch's stats, then reset for the next batch.
orchestrator_run_stats orchestrator_pool::drain() {
    std::unique_lock<std::mutex> lock(pimpl->mtx);
    pimpl->cv_done.wait(lock, [&] {
        return pimpl->queue.empty() && pimpl->n_inflight == 0;
    });

    orchestrator_run_stats stats;
    stats.n_jobs   = pimpl->n_jobs;
    stats.n_failed = pimpl->n_failed;
    stats.seconds  = pimpl->n_jobs == 0 ? 0.0
                   : (pimpl->t_last_us - pimpl->t_first_us) / 1e6;

    // latency distribution over this batch. safe to sort in place: n_inflight == 0, so no
    // worker is touching `latencies`. nearest-rank percentiles on the sorted samples.
    std::vector<double> & lat = pimpl->latencies;
    if (!lat.empty()) {
        std::sort(lat.begin(), lat.end());
        const size_t n = lat.size();
        double sum = 0.0;
        for (double v : lat) {
            sum += v;
        }
        const auto pct = [&](double p) {
            size_t idx = (size_t) (p * (n - 1) + 0.5); // round to nearest rank
            if (idx >= n) {
                idx = n - 1;
            }
            return lat[idx];
        };
        stats.lat_ms_min  = lat.front();
        stats.lat_ms_max  = lat.back();
        stats.lat_ms_mean = sum / (double) n;
        stats.lat_ms_p50  = pct(0.50);
        stats.lat_ms_p95  = pct(0.95);
        stats.lat_ms_p99  = pct(0.99);
    }

    // safe to reset: no worker is active (n_inflight == 0) and the queue is empty.
    pimpl->n_jobs = pimpl->n_failed = pimpl->n_started = 0;
    pimpl->t_first_us = pimpl->t_last_us = 0;
    pimpl->latencies.clear();
    return stats;
}

// factory

// build one replica per spec; returns nullptr if any replica fails to load.
//
// this is the heart of the data-parallel idea: instead of slicing one model across GPUs,
// we load a SEPARATE, complete copy of the model per spec, and run them at the same time.
// a single-device spec is an ordinary single-GPU model (split_mode NONE); a multi-device
// spec is the capacity escape hatch - ONE replica split across that group via llama.cpp's
// native model split, for a model too big for one card. nothing about the math changes
// either way. all-or-nothing: if any replica fails to load we tear down the ones already
// built and return nullptr, so the caller never sees a half-built pool.
std::unique_ptr<orchestrator_pool> orchestrator_make_pool(
        common_params & params, const std::vector<orchestrator_replica_spec> & specs) {
    if (specs.empty()) {
        LOG_ERR("%s: no specs given; need at least one replica\n", __func__);
        return nullptr;
    }

    // make sure backends are registered before we enumerate / load - same call
    // the CLI device parsers make; harmless if already done.
    ggml_backend_load_all();

    // enumerate GPU devices once, for validation + placement. single-device specs place
    // via main_gpu (identical to `-sm none -mg`); multi-device specs pass an explicit
    // NULL-terminated device list to llama.
    std::vector<ggml_backend_dev_t> gpus;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpus.push_back(d);
        }
    }
    if (gpus.empty()) {
        LOG_WRN("%s: no GPU devices found - replicas will load on CPU "
                "(expected for the local CPU-only build / smoke test)\n", __func__);
    } else {
        LOG_INF("%s: %zu GPU device(s) available\n", __func__, gpus.size());
    }

    // private ctor + friend access: allocate the pool (and its impl) up front.
    std::unique_ptr<orchestrator_pool> pool(new orchestrator_pool());

    // context params are identical for every replica.
    const auto cparams = common_context_params_to_llama(params);

    // batching headroom: each context is built with n_seq_max sequence slots (from
    // params.n_parallel). surface it so the operator can see how much room exists for
    // future continuous batching, even though the scheduler runs one job at a time today.
    if (cparams.n_ctx == 0) {
        LOG_INF("%s: n_seq_max = %u per replica (n_ctx from model, split across the sequences)\n",
                __func__, cparams.n_seq_max);
    } else {
        LOG_INF("%s: n_seq_max = %u per replica (n_ctx = %u -> ~%u tokens/sequence)\n",
                __func__, cparams.n_seq_max, cparams.n_ctx,
                cparams.n_seq_max ? cparams.n_ctx / cparams.n_seq_max : cparams.n_ctx);
    }
    if (cparams.n_ctx > 0 && cparams.n_seq_max > cparams.n_ctx) {
        LOG_WRN("%s: n_ctx (%u) < n_seq_max (%u): each sequence gets < 1 token of context\n",
                __func__, cparams.n_ctx, cparams.n_seq_max);
    }

    // build the replicas in parallel - one thread per replica. each replica's model load +
    // context creation is independent (a distinct GPU, or CPU), and loading them serially was
    // the dominant fixed cost (nsys: a 4-step H2D staircase + serial cudaMallocHost, ~70% of a
    // short run). the threads write DISTINCT pre-sized slots, so no lock is needed around the
    // shared vector; everything else they read (gpus, cparams, params, specs) is read-only here.
    orchestrator_pool::impl * p = pool->pimpl.get();
    p->replicas.resize(specs.size());
    std::vector<char> ok(specs.size(), 0); // per-replica success (char avoids vector<bool>)

    // __func__ inside the lambda would resolve to its operator(); capture the real name.
    const char * const fn = __func__;

    auto build_one = [&](size_t r) {
        const orchestrator_replica_spec & spec = specs[r];
        if (spec.devices.empty()) {
            LOG_ERR("%s: [replica %zu] empty device list in spec\n", fn, r);
            return;
        }

        auto mparams = common_model_params_to_llama(params);

        // device handles for a multi-device split, NULL-terminated for llama. local to this
        // thread so it outlives the load call below (mparams.devices points into it).
        std::vector<ggml_backend_dev_t> group;
        llama_split_mode                split_mode = LLAMA_SPLIT_MODE_NONE;

        if (gpus.empty()) {
            // CPU-only build (local / TSan): can't pin or split. a multi-device spec is
            // meaningless here, so fall back to a single CPU replica.
            if (spec.devices.size() > 1) {
                LOG_WRN("%s: [replica %zu] %zu-device split requested but no GPUs found - "
                        "loading a single CPU replica instead\n",
                        fn, r, spec.devices.size());
            }
        } else {
            // validate every device index in the group before touching mparams.
            for (int dev : spec.devices) {
                if (dev < 0 || (size_t) dev >= gpus.size()) {
                    LOG_ERR("%s: [replica %zu] requested device %d but only %zu GPU(s) exist\n",
                            fn, r, dev, gpus.size());
                    return;
                }
            }
            if (spec.devices.size() == 1) {
                // single GPU: pin, numerically identical to `-sm none -mg dev`.
                mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
                mparams.main_gpu   = spec.devices[0];
                mparams.devices    = nullptr;
            } else {
                // 2+ GPUs: place ONE replica across the group via native model split.
                for (int dev : spec.devices) {
                    group.push_back(gpus[dev]);
                }
                group.push_back(nullptr); // NULL-terminated, as llama requires
                mparams.split_mode = spec.split_mode;
                mparams.devices    = group.data();
                mparams.main_gpu   = 0; // index into `devices`, not the global enumeration
                split_mode         = spec.split_mode;
            }
        }

        // human-readable device-group label for logging.
        std::string where;
        for (size_t k = 0; k < spec.devices.size(); ++k) {
            if (k) {
                where += ",";
            }
            where += gpus.empty() ? "CPU" : ggml_backend_dev_name(gpus[spec.devices[k]]);
        }

        LOG_INF("%s: [replica %zu/%zu] loading on device(s) {%s} from '%s'\n",
                fn, r + 1, specs.size(), where.c_str(), params.model.path.c_str());

        // wrap the model immediately so it is freed even if context creation fails.
        llama_model_ptr model(llama_model_load_from_file(params.model.path.c_str(), mparams));
        if (!model) {
            LOG_ERR("%s: [replica %zu] failed to LOAD MODEL ('%s')\n",
                    fn, r, params.model.path.c_str());
            return;
        }

        llama_context_ptr ctx(llama_init_from_model(model.get(), cparams));
        if (!ctx) {
            LOG_ERR("%s: [replica %zu] failed to CREATE CONTEXT (device(s) {%s})\n",
                    fn, r, where.c_str());
            return;
        }

        // watchdog wiring: a heap atomic deadline (stable address) plus an abort callback
        // that reads it. the callback fires between graph ops on CPU/Metal; the worker arms
        // and disarms the deadline around each job.
        auto deadline = std::make_unique<std::atomic<int64_t>>(0);
        llama_set_abort_callback(ctx.get(), orchestrator_abort_cb, deadline.get());

        // publish into this replica's own slot. std::move transfers the smart pointers (no
        // model copy); moving `deadline` keeps the heap atomic's address stable so the abort
        // callback's pointer stays valid.
        orchestrator_pool::impl::replica & rep = p->replicas[r];
        rep.device     = spec.devices[0];
        rep.devices    = spec.devices;
        rep.split_mode = split_mode;
        rep.deadline   = std::move(deadline);
        rep.model      = std::move(model);
        rep.ctx        = std::move(ctx);

        ok[r] = 1;
        LOG_INF("%s: [replica %zu/%zu] OK {%s}\n", fn, r + 1, specs.size(), where.c_str());
    };

    {
        std::vector<std::thread> loaders;
        loaders.reserve(specs.size());
        for (size_t r = 0; r < specs.size(); ++r) {
            loaders.emplace_back(build_one, r);
        }
        for (auto & t : loaders) {
            t.join();
        }
    }

    // all-or-nothing, same contract as the serial version: if any replica failed, discard the
    // whole pool (the ones that did build are freed as `pool` unwinds; no workers started yet).
    for (size_t r = 0; r < specs.size(); ++r) {
        if (!ok[r]) {
            LOG_ERR("%s: replica %zu failed to build - discarding the pool\n", __func__, r);
            return nullptr;
        }
    }

    // every replica built; only now start the workers (one per replica). starting after the
    // build means a pool that failed above never spawned a thread - nothing to join on error.
    for (int r = 0; r < (int) p->replicas.size(); ++r) {
        p->workers.emplace_back([p, r] { p->worker_loop(r); });
    }

    LOG_INF("%s: SUCCESS - built %zu replica(s)\n", __func__, p->replicas.size());
    return pool;
}

// convenience: one single-GPU replica per device index (the original API). a repeated
// index oversubscribes a GPU (e.g. {0,0,1,1}). maps each index to a single-device spec
// and forwards to the spec-based factory.
std::unique_ptr<orchestrator_pool> orchestrator_make_pool(
        common_params & params, const std::vector<int> & devices) {
    std::vector<orchestrator_replica_spec> specs;
    specs.reserve(devices.size());
    for (int d : devices) {
        orchestrator_replica_spec s;
        s.devices = { d };
        specs.push_back(std::move(s));
    }
    return orchestrator_make_pool(params, specs);
}

// resolve the --dp-* placement fields into specs (precedence documented in the header).
// pure arithmetic over the parsed flags; GPU enumeration/validation happens in make_pool.
std::vector<orchestrator_replica_spec> orchestrator_specs_from_params(const common_params & params) {
    std::vector<orchestrator_replica_spec> specs;

    // 1. explicit grouped placement (model-too-big split). most explicit -> wins.
    if (!params.dp_device_groups.empty()) {
        for (const auto & group : params.dp_device_groups) {
            orchestrator_replica_spec s;
            s.devices    = group;
            s.split_mode = group.size() > 1 ? params.dp_split_mode : LLAMA_SPLIT_MODE_NONE;
            specs.push_back(std::move(s));
        }
        return specs;
    }

    // 2. explicit flat placement (pinned / oversubscription).
    if (!params.dp_devices.empty()) {
        for (int dev : params.dp_devices) {
            orchestrator_replica_spec s;
            s.devices = { dev };
            specs.push_back(std::move(s));
        }
        return specs;
    }

    // 3. count-based placement: R replicas on each of GPUs 0..D-1.
    if (params.dp_num_devices > 0 || params.dp_replicas_per_device > 0) {
        const int n_dev = params.dp_num_devices > 0 ? params.dp_num_devices : params.n_data_parallel;
        const int per   = params.dp_replicas_per_device > 0 ? params.dp_replicas_per_device : 1;
        for (int d = 0; d < n_dev; ++d) {
            for (int r = 0; r < per; ++r) {
                orchestrator_replica_spec s;
                s.devices = { d };
                specs.push_back(std::move(s));
            }
        }
        return specs;
    }

    // 4. default: n_data_parallel pinned replicas on GPUs 0..N-1.
    for (int i = 0; i < params.n_data_parallel; ++i) {
        orchestrator_replica_spec s;
        s.devices = { i };
        specs.push_back(std::move(s));
    }
    return specs;
}

// route through the orchestrator when the placement is non-trivial: >1 replica, or a single replica
// spanning >1 GPU (a model-too-big split the baseline path would ignore). a lone single-GPU replica
// stays baseline, keeping --data-parallel 1 / no flags byte-identical no-ops.
bool orchestrator_dp_active(const common_params & params) {
    const auto specs = orchestrator_specs_from_params(params);
    if (specs.size() > 1) {
        return true;
    }
    return specs.size() == 1 && specs.front().devices.size() > 1;
}
