// tools/orchestrator/orchestrator.h
#pragma once

#include "llama.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

struct common_params; // defined in common/common.h

// metadata about one replica; carries no context pointer on purpose - the live
// llama_context is owned by a worker thread and must not escape the pool
struct orchestrator_replica_info {
    int index  = -1; // 0 .. size()-1
    int device = -1; // GPU index this replica is pinned to
};

// runs on one replica's ctx; return true on success, false on failure (e.g.
// llama_decode returned nonzero). a throw is caught and counted as a failure,
// it never terminates the process.
// the pool serializes access to each ctx, so no locking is needed around ctx.
// that covers ctx ONLY - state the closure captures is shared across replicas
// and must be made race-free by the caller (write a distinct slot per job, not
// a shared accumulator).
using orchestrator_job_fn = std::function<bool(llama_context * ctx, int replica_index)>;

// stats for one drain(): wall-clock span plus a failure tally
struct orchestrator_run_stats {
    double seconds  = 0.0; // span from first dispatch to last completion
    size_t n_jobs   = 0;   // jobs executed in this batch
    size_t n_failed = 0;   // jobs that returned false or threw
    bool ok() const { return n_failed == 0; }
};

// N full model replicas + a thread-safe scheduler: one worker thread per replica
// pulls jobs off a shared bounded queue and runs them on its own ctx.
// every replica is a full, independent copy that must fit entirely on its device
// - this scales throughput, not capacity. a model too large for one device still
// belongs to -sm layer/row, not here.
struct orchestrator_pool {
    int                       size() const; // number of replicas (== devices.size())
    orchestrator_replica_info at(int i) const; // metadata only, no live pointers

    // read-only model handle for metadata (n_vocab, n_embd, ...). all replicas are
    // identical and the model is immutable after load, so this is safe to read
    // while jobs run. does not expose any context.
    const llama_model * model() const;

    // enqueue a job; blocks if the bounded queue is full. runs later on the next
    // free replica. call only from a non-worker thread (not from inside a job) -
    // a job that submits can deadlock against a full queue.
    void submit(orchestrator_job_fn job);

    // block until every job submitted since the previous drain() (or since
    // construction) has finished; return that batch's stats. reusable: the clock
    // resets at the first dispatch of each batch.
    // stats are coherent only for one producer that submits a whole batch, drains
    // it fully, then submits the next. overlapping submit() with drain() (or from
    // multiple threads) conflates batches into one span and count.
    orchestrator_run_stats drain();

    ~orchestrator_pool();
    orchestrator_pool(const orchestrator_pool &)             = delete;
    orchestrator_pool & operator=(const orchestrator_pool &) = delete;

private:
    orchestrator_pool(); // built only via orchestrator_make_pool()
    struct impl;
    std::unique_ptr<impl> pimpl;
    friend std::unique_ptr<orchestrator_pool> orchestrator_make_pool(
        common_params & params, const std::vector<int> & devices);
};

// build one replica per entry in `devices` (replica r pinned to devices[r]),
// reusing common_model_params_to_llama() / common_context_params_to_llama() then
// overriding split_mode = NONE and main_gpu = devices[r]. `devices` may repeat an
// index to oversubscribe a GPU (e.g. {0,0,1,1}). returns nullptr if any replica
// fails to load, after freeing any already built.
std::unique_ptr<orchestrator_pool> orchestrator_make_pool(
    common_params & params, const std::vector<int> & devices);
