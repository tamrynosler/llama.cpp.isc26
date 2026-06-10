#include "orchestrator.h"

#include "common.h"    // common_params, common_model_params_to_llama, common_context_params_to_llama
#include "llama-cpp.h" // llama_model_ptr, llama_context_ptr (RAII teardown)
#include "log.h"       // LOG_INF / LOG_ERR

#include <condition_variable> // cv_work / cv_space / cv_done
#include <deque>              // job queue
#include <mutex>              // queue + scheduler guard
#include <thread>             // one worker per replica
#include <utility>            // std::move

// private implementation

struct orchestrator_pool::impl {
    impl() = default;
    ~impl(); // stops + joins workers before any member (the contexts) is destroyed

    // one full model + its context per replica. declaration order matters: ctx
    // is declared after model so it is destroyed first (bottom-to-top), tearing
    // down the context before the model it borrows from - same ordering rule as
    // common_init_result::impl.
    struct replica {
        int               device = -1; // GPU index this replica is pinned to
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

    // per-batch accounting, reset by each drain() (all guarded by mtx)
    size_t  n_jobs     = 0;
    size_t  n_failed   = 0;
    size_t  n_started  = 0;    // jobs dispatched this batch; the first stamps t_first_us
    int64_t t_first_us = 0;    // ggml_time_us() at the batch's first dispatch
    int64_t t_last_us  = 0;    // ggml_time_us() at the last completion so far

    // declared last so they are torn down first; ~impl() has already joined them
    // by then, so the std::thread destructors see non-joinable threads.
    std::vector<std::thread> workers;

    void worker_loop(int r); // body of worker r, bound to replicas[r].ctx
};

// each worker owns one ctx for its whole life and pulls jobs off the shared
// queue. jobs run OUTSIDE the lock, so replicas execute concurrently.
void orchestrator_pool::impl::worker_loop(int r) {
    llama_context * ctx = replicas[r].ctx.get();
    for (;;) {
        orchestrator_job_fn job;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv_work.wait(lock, [&] { return stop || !queue.empty(); });
            if (stop && queue.empty()) {
                return; // shutdown: finish the queue first, then exit
            }
            job = std::move(queue.front());
            queue.pop_front();
            if (n_started++ == 0) {
                t_first_us = ggml_time_us(); // first dispatch of this batch
            }
            n_inflight++;
            cv_space.notify_one(); // a queue slot just freed
        }
        bool ok = false;
        try {
            ok = job(ctx, r);
        } catch (...) {
            ok = false; // a throwing job is a failure, never a process abort
        }
        {
            // re-locking here publishes the job's writes (e.g. results[i], done
            // outside the lock) to drain(): this unlock synchronizes-with the
            // lock drain() takes, so the writes happen-before drain() returns.
            std::lock_guard<std::mutex> lock(mtx);
            n_jobs++;
            if (!ok) {
                n_failed++;
            }
            t_last_us = ggml_time_us(); // last completion so far
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

// metadata for replica i (index + pinned device); exposes no live pointers
orchestrator_replica_info orchestrator_pool::at(int i) const {
    // an out-of-range index is a caller bug, not a runtime condition - fail loud.
    GGML_ASSERT(i >= 0 && (size_t) i < pimpl->replicas.size());
    return { i, pimpl->replicas[i].device };
}

// read-only model handle for metadata queries (n_vocab, ...); never a context.
// all replicas are identical copies, so replica 0's model speaks for all of them.
const llama_model * orchestrator_pool::model() const {
    if (pimpl->replicas.empty()) {
        return nullptr;
    }
    return pimpl->replicas[0].model.get();
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

    // safe to reset: no worker is active (n_inflight == 0) and the queue is empty.
    pimpl->n_jobs = pimpl->n_failed = pimpl->n_started = 0;
    pimpl->t_first_us = pimpl->t_last_us = 0;
    return stats;
}

// factory

// build one replica per entry in `devices`, each pinned to devices[r] with
// split_mode = NONE; returns nullptr if any replica fails to load.
//
// this is the heart of the data-parallel idea: instead of slicing one model
// across GPUs (tensor/layer split), we load a SEPARATE, complete copy of the
// model onto each requested GPU. each copy is an ordinary single-GPU model, so
// nothing about the math changes - we just end up with N of them that can run
// at the same time. all-or-nothing: if any copy fails to load we tear down the
// ones already built and return nullptr, so the caller never sees a half-built
// pool.
std::unique_ptr<orchestrator_pool> orchestrator_make_pool(
        common_params & params, const std::vector<int> & devices) {
    if (devices.empty()) {
        LOG_ERR("%s: no devices given; need at least one replica\n", __func__);
        return nullptr;
    }

    // make sure backends are registered before we enumerate / load - same call
    // the CLI device parsers make; harmless if already done.
    ggml_backend_load_all();

    // enumerate GPU devices once, for validation + debug logging only. placement
    // itself is done via main_gpu below, identical to `-sm none -mg`.
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

    // build the replicas one at a time; `r` is the replica number, `dev` is the
    // GPU it should live on. two things can go wrong per replica - loading the
    // model file, and opening a context on it - and we report them separately.
    for (size_t r = 0; r < devices.size(); ++r) {
        const int dev = devices[r];

        // validate the requested GPU index when GPUs are present.
        if (!gpus.empty() && (dev < 0 || (size_t) dev >= gpus.size())) {
            LOG_ERR("%s: [replica %zu] requested device %d but only %zu GPU(s) exist\n",
                    __func__, r, dev, gpus.size());
            return nullptr;
        }
        const char * dev_name = gpus.empty() ? "CPU" : ggml_backend_dev_name(gpus[dev]);

        // start from the model params every tool would use, then pin this replica
        // to one device: split_mode NONE + main_gpu = dev is numerically identical
        // to a normal `-sm none -mg dev` run. devices is cleared so main_gpu selects
        // the dev-th GPU from the full enumeration. n_gpu_layers is left as the user
        // set it - like every other tool, you still need -ngl to offload to the GPU.
        // when there are no GPUs (CPU-only build / TSan), pinning is meaningless and
        // main_gpu = dev would be rejected (main_gpu must index an existing device),
        // so leave the CPU defaults: every replica runs on CPU.
        auto mparams = common_model_params_to_llama(params);
        if (!gpus.empty()) {
            mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
            mparams.main_gpu   = dev;
            mparams.devices    = nullptr;
        }

        LOG_INF("%s: [replica %zu/%zu] loading on device %d (%s) from '%s'\n",
                __func__, r + 1, devices.size(), dev, dev_name, params.model.path.c_str());

        // wrap the model immediately so it is freed even if context creation fails.
        llama_model_ptr model(llama_model_load_from_file(params.model.path.c_str(), mparams));
        if (!model) {
            LOG_ERR("%s: [replica %zu] failed to LOAD MODEL (device %d, '%s')\n",
                    __func__, r, dev, params.model.path.c_str());
            return nullptr; // already-built replicas freed as `pool` unwinds
        }

        llama_context_ptr ctx(llama_init_from_model(model.get(), cparams));
        if (!ctx) {
            LOG_ERR("%s: [replica %zu] failed to CREATE CONTEXT (device %d, '%s')\n",
                    __func__, r, dev, dev_name);
            return nullptr; // this model + prior replicas freed on return
        }

        // hand ownership of this loaded model + context to the pool. std::move
        // transfers the smart pointers (no copying of the model), so from here
        // the pool is responsible for freeing them at shutdown.
        orchestrator_pool::impl::replica rep;
        rep.device = dev;
        rep.model  = std::move(model);
        rep.ctx    = std::move(ctx);
        pool->pimpl->replicas.push_back(std::move(rep));

        LOG_INF("%s: [replica %zu/%zu] OK (device %d, %s)\n",
                __func__, r + 1, devices.size(), dev, dev_name);
    }

    // every replica loaded; only now start the workers. starting after the build
    // loop means a pool that failed partway (returned nullptr above) never spawned
    // a thread, so there is nothing to join on that error path.
    orchestrator_pool::impl * p = pool->pimpl.get();
    for (int r = 0; r < (int) p->replicas.size(); ++r) {
        p->workers.emplace_back([p, r] { p->worker_loop(r); });
    }

    LOG_INF("%s: SUCCESS - built %zu replica(s)\n", __func__, p->replicas.size());
    return pool;
}
