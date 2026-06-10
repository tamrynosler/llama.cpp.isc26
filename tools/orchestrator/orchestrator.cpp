#include "orchestrator.h"

#include "common.h"    // common_params, common_model_params_to_llama, common_context_params_to_llama
#include "llama-cpp.h" // llama_model_ptr, llama_context_ptr (RAII teardown)
#include "log.h"       // LOG_INF / LOG_ERR

#include <utility>     // std::move

// note: threading headers (<thread>, <mutex>, <condition_variable>, <deque>,
// <chrono>, <atomic>) are added alongside the scheduler members in S6/S7.

// private implementation

struct orchestrator_pool::impl {
    impl()  = default;
    ~impl() = default;

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

    // scheduler state (declared as each piece is implemented):
    // S6: bounded job queue + bookkeeping for serial dispatch.
    // S7: worker threads, mutex, condition_variable, shutdown flag, failure
    //     tally, and per-batch wall-clock timing.
};

// construction / teardown

// private; constructed only via orchestrator_make_pool(). allocates the impl so
// every pool has valid internal state for its whole lifetime.
orchestrator_pool::orchestrator_pool() : pimpl(std::make_unique<impl>()) {}

orchestrator_pool::~orchestrator_pool() = default;
// TODO(S7): once worker threads exist, stop the queue and join all workers here
//           BEFORE pimpl (and thus the contexts) are destroyed.

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

// enqueue a job to run on the next free replica; blocks if the queue is full
void orchestrator_pool::submit(orchestrator_job_fn job) {
    // TODO(S6): push job onto the bounded queue (serial, single-worker dispatch
    //           first); S7 promotes this to concurrent worker threads.
    (void) job;
}

// wait for all jobs submitted since the last drain() to finish; return stats
orchestrator_run_stats orchestrator_pool::drain() {
    // TODO(S6): block until the queue drains and all in-flight jobs complete.
    //           S7 adds the wall-clock span and the failure tally.
    return {};
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
        auto mparams = common_model_params_to_llama(params);
        mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
        mparams.main_gpu   = dev;
        mparams.devices    = nullptr;

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

    LOG_INF("%s: SUCCESS - built %zu replica(s)\n", __func__, pool->pimpl->replicas.size());
    return pool;
}
