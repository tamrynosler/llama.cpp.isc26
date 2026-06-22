# Orchestrator Slurm test scripts

Benchmark / verification jobs for the data-parallel orchestrator, run on one
or more nodes of the target cluster (4x H100 per node). Logs land in
`results/<job>-<jobid>/`.

## Scripts

| Script | Repo it targets | What it runs |
|--------|-----------------|--------------|
| `baseline-original.slurm` | original upstream `~/llama.cpp` | 3 models, 1 GPU, default perplexity. The reference numbers. |
| `dp-orchestrator.slurm`   | this fork `~/llama.cpp.isc26` | per model: 1 GPU default, 1 GPU matched (`-b $CTX`), 4 GPUs `--data-parallel 4`. |
| `dp-batched-bench.slurm`  | this fork `~/llama.cpp.isc26` | per model: 1 GPU baseline vs 4 GPUs `--data-parallel 4` batched-bench. The GPU-bound throughput showcase; compare the DP `ALL` row against the 1-GPU row. No input dataset (random tokens). |
| `dp-multinode-srun.slurm` | this fork `~/llama.cpp.isc26` | **multi-node (prototype).** `srun` fans the single-node DP sweep to every node; `aggregate-multinode.py` sums per-node throughput offline. RPC-free, no in-binary coordination. Job 453 = 2.00x on 2 nodes. |
| `dp-batched-bench-cluster.slurm` | this fork `~/llama.cpp.isc26` | **multi-node (C++-native).** Same workload as the prototype, but coordination + aggregation happen **in-binary** via the MPI `cluster_link` (`srun --mpi=pmix`, one rank per node). No python. Output carries per-`NODE` rows + a `CL` cluster row. Requires the `-DLLAMA_ORCH_MPI=ON` build. |
| `dp-speculative.slurm` | this fork `~/llama.cpp.isc26` | **speculative decoding + DP.** per (target,draft) pair + draft length: 1 GPU single spec stream vs 4 GPUs `--data-parallel 4` (one target+draft pair per GPU). Combines the spec speedup with the replica speedup; compare the DP `ALL`-row decode t/s vs the 1-GPU `decoded … t/s`, acceptance% unchanged. Single-node, no MPI. |

Fill in the `MODELS`, repo paths, and `INPUT` variables at the top of each
script (use the **same 3 models** in both). Build the relevant repo with
`-DGGML_CUDA=ON` first.

## Cluster specifics

- **No GRES.** Every node has 4 GPUs; we pick them with `CUDA_VISIBLE_DEVICES`.
  The scripts deliberately do **not** use `#SBATCH --gres`.
- Modules via lmod: `ml gcc cuda` (pin versions if your site requires, e.g.
  `ml gcc/15.2 cuda/13.1`). The C++-native cluster script also needs MPI:
  `ml gcc hpcx cuda` (HPC-X = Open MPI + UCX) and a build with
  `-DLLAMA_ORCH_MPI=ON`. Launch it with `srun --mpi=pmix` (fall back to `pmi2`).
- `-ngl 99` is required for GPU offload. The orchestrator **warns** if `-ngl` is
  left at auto but does **not** silently force it, so the scripts set it explicitly.

## Input text

Standard wikitext-2 raw (same file llama.cpp's own PPL docs use):

```
wget https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
unzip wikitext-2-raw-v1.zip   # -> wikitext-2-raw/wiki.test.raw
```

Point `INPUT` at `wiki.test.raw`.

## Reading the results

- **Legality (correctness):** the `Final estimate: PPL` from `*.1gpu-matched.log`
  must equal `*.4gpu-dp.log` exactly. Both use `n_seq = 1`, so the per-token NLL
  sums match bit-for-bit at the printed precision.
- **Why a separate "matched" run?** Default single-replica perplexity packs
  `n_seq = n_batch / n_ctx` chunks into one decode (a big batch); the S9
  orchestrator path uses `n_seq = 1` per replica (one chunk per decode). Because
  llama.cpp results are batch-size dependent (GEMM reduction order), the *default*
  1-GPU PPL differs slightly from the DP PPL. That is expected, not a bug —
  comparing at matched granularity (`-b $CTX`) removes it. Raising the
  per-replica `n_seq` knob later would also make DP match the default baseline.
- **Throughput (the actual goal):** compare wall-clock (`Elapsed (wall ...)` or
  the `chunks across N replicas in Y s` line) of `*.4gpu-dp.log` vs
  `*.1gpu-default.log`. We expect throughput to scale toward 4x with replica count.
