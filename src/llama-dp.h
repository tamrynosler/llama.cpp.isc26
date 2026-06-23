// src/llama-dp.h
//
// Data-parallel proxy layer (orchestrator2). Lets a single, UNMODIFIED benchmark binary
// (llama-batched-bench) drive N independent model replicas across N devices.
//
// How it works: when llama_model_params.n_data_parallel > 1, the public factory functions
// return *proxy* handles instead of real ones:
//   - llama_model_load_from_file  -> a proxy llama_model* wrapping N real replica models
//                                    (replica r pinned to device r)
//   - llama_init_from_model       -> a proxy llama_context* wrapping one real context per replica
// Each subsequent C-API call on a proxy handle is intercepted by a one-line guard in the
// llama_* wrapper and routed here. The hot path, llama_decode(), shards the batch's sequences
// across the replicas (sequence j -> replica j % N) and runs the per-replica decodes
// concurrently, one worker thread per replica. The benchmark's own throughput numbers then
// reflect the aggregate, with batched-bench.cpp byte-identical to upstream.
//
// This layer uses ONLY the public llama.h C API to manage replicas, so it never reaches into
// llama_context/llama_model internals. Recursion is avoided because every per-replica call is
// made with n_data_parallel == 0 (the stock path).

#pragma once

#include "llama.h"

// True when data parallelism is requested for this load (n_data_parallel > 1).
bool llama_dp_enabled(const struct llama_model_params & params);

//
// model side
//

// Build a proxy model: load `params.n_data_parallel` real replicas of `path`, replica r pinned
// to device r (split_mode NONE, main_gpu = r). Returns a proxy llama_model* (registered) or
// nullptr on any replica failure (already-loaded replicas are freed first).
struct llama_model * llama_dp_model_load(const char * path, struct llama_model_params params);

bool                 llama_dp_is_model (const struct llama_model * model);
const struct llama_vocab * llama_dp_model_vocab(const struct llama_model * model); // replica 0's vocab
void                 llama_dp_model_free(struct llama_model * model);

//
// context side
//

// Build a proxy context: one real context per replica model, plus the worker pool + scratch.
struct llama_context * llama_dp_init(struct llama_model * dp_model, struct llama_context_params params);

bool           llama_dp_is_ctx     (const struct llama_context * ctx);
int32_t        llama_dp_decode      (struct llama_context * ctx, struct llama_batch batch);
void           llama_dp_synchronize (struct llama_context * ctx);
uint32_t       llama_dp_n_ctx       (const struct llama_context * ctx);
uint32_t       llama_dp_n_seq_max   (const struct llama_context * ctx);
llama_memory_t llama_dp_get_memory  (struct llama_context * ctx);
const struct llama_model * llama_dp_get_model(const struct llama_context * ctx);
void           llama_dp_perf_print  (const struct llama_context * ctx);
void           llama_dp_free        (struct llama_context * ctx);

// logits readback (perplexity). a token decoded on replica r is read back from r; because each
// sequence lives wholly on one replica and stays contiguous, the per-sequence contiguous read in
// perplexity's process_logits() is preserved.
float *        llama_dp_get_logits    (struct llama_context * ctx);
float *        llama_dp_get_logits_ith(struct llama_context * ctx, int32_t i);

//
// memory side (the memory handle returned by llama_dp_get_memory is the proxy context itself)
//

bool llama_dp_is_mem      (const llama_memory_t mem);
void llama_dp_memory_clear (llama_memory_t mem, bool data);
bool llama_dp_memory_seq_rm(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1);
void llama_dp_memory_seq_cp(llama_memory_t mem, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1);
