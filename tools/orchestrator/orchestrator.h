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
    int              index  = -1; // 0 .. size()-1
    int              device = -1; // first/main device of this replica (== devices[0])
    std::vector<int> devices;     // every GPU this replica occupies (1 = pinned, 2+ = model split)
    llama_split_mode split_mode = LLAMA_SPLIT_MODE_NONE; // how the model is split across `devices`
};

// placement for one replica. a single device pins the replica to that GPU (split_mode
// NONE, the default behaviour). two or more devices place ONE replica across that group
// using llama.cpp's native model split - a capacity escape hatch for a model too large
// for a single GPU. this does NOT change throughput scaling (that still comes from
// running multiple replicas); we only configure the existing split, no kernel/ggml work.
struct orchestrator_replica_spec {
    std::vector<int> devices;                              // 1 = pinned; 2+ = split group
    llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER;  // consulted only when devices.size() > 1
};

// runs on one replica's ctx; return true on success, false on failure (e.g.
// llama_decode returned nonzero). a throw is caught and counted as a failure,
// it never terminates the process.
// the pool serializes access to each ctx, so no locking is needed around ctx.
// that covers ctx ONLY - state the closure captures is shared across replicas
// and must be made race-free by the caller (write a distinct slot per job, not
// a shared accumulator).
using orchestrator_job_fn = std::function<bool(llama_context * ctx, int replica_index)>;

// stats for one drain(): wall-clock span, a failure tally, and the per-job latency
// distribution for that batch.
struct orchestrator_run_stats {
    double seconds  = 0.0; // span from first dispatch to last completion
    size_t n_jobs   = 0;   // jobs executed in this batch
    size_t n_failed = 0;   // jobs that returned false or threw
    bool ok() const { return n_failed == 0; }

    // per-job SERVICE latency (dispatch -> completion, excludes queue wait), milliseconds.
    // all zero when n_jobs == 0. token-level throughput (pp/tg tokens/sec) is intentionally
    // NOT here - the pool stays token-agnostic; the bench harness aggregates that from each
    // job's own llama_perf_context().
    double lat_ms_min  = 0.0;
    double lat_ms_max  = 0.0;
    double lat_ms_mean = 0.0;
    double lat_ms_p50  = 0.0;
    double lat_ms_p95  = 0.0;
    double lat_ms_p99  = 0.0;
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

    // best-effort per-job timeout in milliseconds; 0 disables it (the default). when set,
    // each job is armed with a deadline of dispatch + timeout. an over-deadline llama_decode
    // is aborted via the context's abort callback - but only on the CPU/Metal backends;
    // CUDA does not poll the callback, so a GPU decode job must additionally check
    // deadline_expired() between steps. a truly wedged driver call cannot be force-killed,
    // so drain() can still block on a hung worker. callable any time; takes effect on the
    // next job each worker picks up.
    void set_timeout_ms(int64_t timeout_ms);

    // true once the job currently running on replica `i` has passed its deadline. a long
    // decode job polls this between llama_decode steps to stop generation on GPU, where the
    // abort callback is not honoured. false when no timeout is set or none has elapsed.
    bool deadline_expired(int i) const;

    ~orchestrator_pool();
    orchestrator_pool(const orchestrator_pool &)             = delete;
    orchestrator_pool & operator=(const orchestrator_pool &) = delete;

private:
    orchestrator_pool(); // built only via orchestrator_make_pool()
    struct impl;
    std::unique_ptr<impl> pimpl;
    friend std::unique_ptr<orchestrator_pool> orchestrator_make_pool(
        common_params & params, const std::vector<orchestrator_replica_spec> & specs);
};

// build one replica per spec, reusing common_model_params_to_llama() /
// common_context_params_to_llama(). a single-device spec pins that replica (split_mode
// NONE, main_gpu = device); a multi-device spec splits one replica across the group via
// `split_mode`. all-or-nothing: if any replica fails to load, the ones already built are
// freed and nullptr is returned.
//
// `params.n_parallel` sets each replica context's n_seq_max - how many sequences it can
// hold. set it now even though the scheduler runs one closure-job at a time, so continuous
// batching can be added later without rebuilding contexts.
std::unique_ptr<orchestrator_pool> orchestrator_make_pool(
    common_params & params, const std::vector<orchestrator_replica_spec> & specs);

// convenience: one single-GPU replica per device index (the original API). a repeated
// index oversubscribes a GPU (e.g. {0,0,1,1}). maps each index to a single-device spec
// and forwards to the spec-based factory above.
std::unique_ptr<orchestrator_pool> orchestrator_make_pool(
    common_params & params, const std::vector<int> & devices);

// resolve replica placement from the data-parallel CLI fields into specs, applying the
// documented precedence (most explicit wins). this is the single point that interprets
// the --dp-* flags, so every tool builds its pool the same way:
//   1. dp_device_groups (grouped --dp-devices, e.g. 0+1,2+3) -> one replica per group;
//      a multi-GPU group is split via dp_split_mode (the model-too-big escape hatch).
//   2. dp_devices (flat --dp-devices, e.g. 0,0,1,1) -> one pinned replica per entry
//      (repeats oversubscribe a GPU).
//   3. dp_num_devices / dp_replicas_per_device -> R replicas on each of GPUs 0..D-1
//      (count-based oversubscription; R defaults to 1).
//   4. otherwise n_data_parallel pinned replicas on GPUs 0..N-1 (the default).
// does not enumerate or validate GPUs (orchestrator_make_pool does that); pure placement arithmetic.
std::vector<orchestrator_replica_spec> orchestrator_specs_from_params(const common_params & params);

// whether this run should go through the orchestrator instead of the tool's baseline single-context
// path. true when the --dp-* flags resolve to more than one replica, OR to a single replica that
// spans multiple GPUs (a model-too-big split, which the baseline path would not apply). a lone
// single-GPU replica is the trivial case and stays on the baseline path, so --data-parallel 1 and
// no flags remain byte-identical no-ops.
bool orchestrator_dp_active(const common_params & params);
