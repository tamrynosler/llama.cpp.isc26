// src/llama-dp.h
//
// Data-parallel proxy layer (orchestrator2). Lets a single, UNMODIFIED benchmark binary
// (llama-batched-bench, llama-perplexity) drive N independent model replicas across N devices.
//
// Handle model: the factory functions return replica 0's *real* llama_model / llama_context as
// the handle, registered in a side map that links it to the other N-1 replicas. This means every
// un-intercepted introspection/config call (llama_model_n_layer, the -fit device probe,
// llama_set_warmup, ...) runs natively on a real object - no crashes - while only the data-path
// functions are intercepted and fanned/sharded across replicas:
//   - llama_model_load_from_file -> loads N replicas (replica r pinned to device r), returns replica 0
//   - llama_init_from_model      -> builds one context per replica, returns replica 0's context
//   - llama_decode               -> shards the batch's sequences (seq j -> replica j % N), runs the
//                                   per-replica decodes concurrently (one worker thread per replica)
//   - llama_get_logits[_ith]     -> reads a token's logits back from the replica that produced it
//   - llama_synchronize / free   -> fan across replicas
//   - llama_memory_clear/seq_*   -> fan across replicas
// At n_data_parallel <= 1 nothing is registered and every guard is false -> the stock path.
//
// This layer uses only the public llama.h C API (+ llama-memory.h to fan memory ops without
// re-entering the C wrappers). Recursion through the intercepted factories/free is avoided by
// unregistering the lead handle around the per-replica build/teardown.

#pragma once

#include "llama.h"

// True when data parallelism is requested for this load (n_data_parallel > 1).
bool llama_dp_enabled(const struct llama_model_params & params);

//
// model side
//

// Load N replicas of `path` (replica r pinned to device r). Returns replica 0's real model
// (registered), or nullptr on any replica failure (already-loaded replicas are freed first).
struct llama_model * llama_dp_model_load(const char * path, struct llama_model_params params);
bool                 llama_dp_is_model (const struct llama_model * model);
void                 llama_dp_model_free(struct llama_model * model);

//
// context side
//

// Build one real context per replica; returns replica 0's real context (registered).
struct llama_context * llama_dp_init(struct llama_model * model, struct llama_context_params params);
bool     llama_dp_is_ctx     (const struct llama_context * ctx);
int32_t  llama_dp_decode      (struct llama_context * ctx, struct llama_batch batch);
void     llama_dp_synchronize (struct llama_context * ctx);
void     llama_dp_free        (struct llama_context * ctx);

// logits readback (perplexity). a token decoded on replica r is read back from r; because each
// sequence lives wholly on one replica and stays contiguous, the per-sequence contiguous read in
// perplexity's process_logits() is preserved.
float *  llama_dp_get_logits    (struct llama_context * ctx);
float *  llama_dp_get_logits_ith(struct llama_context * ctx, int32_t i);

//
// memory side (the handle is replica 0's real memory; ops fan to every replica)
//

bool llama_dp_is_mem      (const llama_memory_t mem);
void llama_dp_memory_clear (llama_memory_t mem, bool data);
bool llama_dp_memory_seq_rm(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1);
void llama_dp_memory_seq_cp(llama_memory_t mem, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1);
