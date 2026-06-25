# Data-Parallel Inference for llama.cpp — Implementation Report

## Goal and constraint

The ISC task is to maximise inference throughput (tokens/second) on a multi-GPU, multi-node
H100 cluster without degrading perplexity by more than 10%. A hard rule from the judges shapes
the whole design: the benchmark binaries — `llama-batched-bench` and `llama-perplexity` — **must
not be modified**. Only the supporting code they depend on may change.

Our solution adds **data parallelism entirely inside the llama library**, transparently to the
unmodified benchmarks. Throughput scales with the number of GPUs; perplexity is unchanged.

## How it works

llama.cpp benchmarks talk to the engine through a small C API and only ever hold *opaque
handles* (`llama_model *`, `llama_context *`). We exploit this: when the user passes
`--data-parallel N`, the library quietly returns a handle that is actually **replica 0 of N
independent model copies**, one pinned to each GPU. A side registry links that lead handle to its
sibling replicas.

Every ordinary call still works on the real replica-0 object — so model introspection, warm-up,
and configuration behave exactly as upstream. Only the *data-path* calls are intercepted:

- **`llama_decode`** — the batch's parallel sequences are split across the replicas (sequence *j*
  → replica *j % N*) and decoded **concurrently**, one worker thread per replica. Each sequence
  lives entirely on one GPU, so its key/value cache stays put across the generation loop.
- **`llama_get_logits[_ith]`** — a token's logits are read back from the replica that produced it
  (this is what lets the *unmodified* `llama-perplexity` run correctly).
- **`llama_memory_*` and `llama_synchronize`** — fanned out to every replica.

The benchmark never knows: it builds one batch, calls one `llama_decode`, and reads one set of
results, while the library spreads the work across all the GPUs. Its own reported throughput
therefore reflects the speed-up directly — no custom output, no edited measurement.

## Multi-node

Two complementary mechanisms extend this across the three nodes, both leaving the binaries stock:

1. **Launcher fan-out.** A SLURM script runs one stock `batched-bench` per node (each doing the
   library data parallelism across its local GPUs); a small Python aggregator sums the per-node
   JSON output. The nodes never communicate during inference (weak scaling).
2. **In-source MPI** (optional, `-DLLAMA_DP_MPI=ON`). Launched one rank per node, the library
   makes rank 0 the driver and the other ranks invisible MPI compute workers — they are caught
   during context creation and never run the benchmark's `main`. Rank 0 shards each decode across
   *all* cluster GPUs (local via threads, remote via MPI) so a **single stock process prints one
   table with cluster-wide throughput**.

## Correctness

Each sequence is processed end-to-end on one full, byte-identical replica, so per-sequence outputs
are identical to single-GPU execution. We verified this: upstream llama.cpp, our fork single-GPU,
and our fork with `--data-parallel` all produce perplexity **28.91** on the test model — a 0.0%
change, well inside the 10% gate. Data parallelism is pure throughput; it changes no numerics.

## Results

- **Single node (4× H100):** ~3.9× prefill, ~3.7× decode vs one GPU.
- **Cluster (3 nodes, 12× H100), launcher:** 2.96–3.00× node scaling, ~24,300 tok/s aggregate.
- **Cluster, in-source MPI:** one unified run across all 12 GPUs in a single table.

## Code footprint

The change to llama.cpp's own existing files is tiny — about **64 lines across six non-benchmark
files** (one-line interception guards in the context/model wrappers, a new `n_data_parallel`
parameter field, a flag bridge, and build wiring). All real logic lives in **two new
self-contained files** (`src/llama-dp.{h,cpp}`, ~750 lines) plus run scripts. Critically,
`batched-bench.cpp` and `perplexity.cpp` are **byte-for-byte identical to upstream** (verified by
diff).

In one sentence: data parallelism was implemented as a **transparent layer beneath the unmodified
benchmarks**, so the scored binaries stay stock while the engine spreads their work across every
GPU and node.
