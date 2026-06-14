# Data-Parallel Orchestrator — Usage Manual

A guide to running and tuning the **data-parallel orchestrator** added to this `llama.cpp` fork.
You don't need to read any source — everything you need to test and tune is here.

---

## 1. What it does (in one minute)

The orchestrator runs **N independent full copies (replicas) of a model across multiple GPUs at once**
and feeds them work concurrently. It scales **throughput** — total tokens/sec across the node — not
the speed of a single request.

- **One replica per GPU** (default) → ~N× the throughput of one GPU.
- A replica can also **span multiple GPUs** (for a model too big for one card) or you can put
  **multiple replicas on one GPU** (for a small model with memory to spare).
- It works **within one node** and **across nodes** (multi-node, via MPI).

It plugs into three existing tools — turn it on with one flag, `-dp N`:

| Tool | What you test with it |
|------|------------------------|
| `llama-perplexity`   | **Correctness** — PPL is identical with and without the orchestrator |
| `llama-batched-bench`| **Throughput** — the headline near-linear scaling story |
| `llama-speculative`  | **Speculative decoding + scaling** — spec speedup × replica speedup |

> Without `-dp` (or `-dp 1`), every tool behaves **exactly** as stock llama.cpp. The orchestrator is
> off by default.

---

## 2. Build

All commands run from inside the repo (`cd ~/llama.cpp.isc26`).

**Single-node (covers perplexity, batched-bench, speculative, splits, oversubscription):**
```bash
ml gcc cuda
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

**Multi-node (adds the cross-node `--cluster` mode — needs MPI):**
```bash
ml gcc cuda hpcx                       # ORDER MATTERS: cuda before hpcx
cmake -B build-mpi -DGGML_CUDA=ON -DLLAMA_ORCH_MPI=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-mpi -j$(nproc)
```

> No GRES on this cluster — pick GPUs with `CUDA_VISIBLE_DEVICES`, never `--gres`.

---

## 3. The flags (quick reference)

| Flag | Meaning |
|------|---------|
| `-dp N`, `--data-parallel N` | Number of replicas. `1` (default) = orchestrator **off**. |
| `--dp-devices i,j,k,...` | Explicit GPU placement, one replica per entry. **Repeat** an index to oversubscribe (`0,0,1,1`). **Join with `+`** for a multi-GPU replica (`0+1,2+3`). |
| `--dp-split {layer,row,tensor}` | How a multi-GPU replica (a `+` group) is split across its GPUs. Default `layer`. |
| `--dp-num-devices D` | Count-based placement: use GPUs `0..D-1` (instead of listing them). |
| `--dp-replicas-per-device R` | Put `R` replicas on each selected GPU (oversubscription by count). |
| `--dp-weak` | *(speculative only)* identical-stream diagnostic — see §6.2. |
| `--dp-chunk-chars N` | *(speculative only)* shard size for strong scaling — see §6.1. |
| `-ngl 99` | **Always pass this.** Offload all layers to GPU. The orchestrator *warns* if you forget but won't force it (a CPU-only DP run is slow). |

Set GPUs with the environment, e.g. `export CUDA_VISIBLE_DEVICES=0,1,2,3`.

---

## 4. Common setup (placeholders used below)

```bash
export REPO=$HOME/llama.cpp.isc26
export BIN=$REPO/build/bin                 # use $REPO/build-mpi/bin for multi-node
export MODELS=/shared/llms/models
export WIKI=/shared/llms/datasets/wikitext-2-raw/wiki.test.raw
```

---

## 5. Throughput & correctness — the two core tools

### 5.1 `llama-batched-bench` — throughput (the main showcase)

Runs a prompt/generation sweep concurrently on every replica and reports an aggregate.

```bash
# (A) 1-GPU baseline
export CUDA_VISIBLE_DEVICES=0
$BIN/llama-batched-bench -m $MODELS/<model>.gguf \
    -c 65536 -b 4096 -ub 2048 -ngl 99 \
    -npp 512 -ntg 128 -npl 1,2,4,8,16,32,64

# (B) 4-GPU data-parallel (one replica per GPU)
export CUDA_VISIBLE_DEVICES=0,1,2,3
$BIN/llama-batched-bench -m $MODELS/<model>.gguf \
    -c 65536 -b 4096 -ub 2048 -ngl 99 \
    -npp 512 -ntg 128 -npl 1,2,4,8,16,32,64 \
    --data-parallel 4 --dp-devices 0,1,2,3
```

**What to read:** each replica prints its own row; the **`ALL` row** is the aggregate. Compare its
`S t/s` (and `S_e2e t/s`) to the 1-GPU `S t/s` — it should approach **4×**. Add `--output-format jsonl`
for machine-readable rows.

| Knob | Effect |
|------|--------|
| `-npl` (parallel sequences) | The main throughput lever — higher = more concurrent decode per replica. |
| `-b` / `-ub` | Bigger logical/physical batch → faster prefill (helps small/mid models). |
| `-npp` / `-ntg` | Prompt / generation token counts per sequence. |

### 5.2 `llama-perplexity` — correctness check

Proves the orchestrator changes throughput, **not** the numbers.

```bash
# 1-GPU
export CUDA_VISIBLE_DEVICES=0
$BIN/llama-perplexity -m $MODELS/<model>.gguf -f $WIKI -c 512 -b 512 -ngl 99

# 4-GPU data-parallel (same flags + -dp)
export CUDA_VISIBLE_DEVICES=0,1,2,3
$BIN/llama-perplexity -m $MODELS/<model>.gguf -f $WIKI -c 512 -b 512 -ngl 99 \
    --data-parallel 4 --dp-devices 0,1,2,3
```

**What to read:** the `Final estimate: PPL = …` line must be **identical** between the two runs.
Use the **same `-b`** on both (the DP path decodes one chunk per step). The DP run finishes faster.

---

## 6. Speculative decoding + scaling — `llama-speculative`

Each replica holds a **(target + draft) pair on one GPU** and runs its own speculative-decode stream.
The speculative speedup and the replica speedup **compound**.

```bash
export TGT=$MODELS/Qwen2.5-72B-Instruct-Q4_K_M.gguf
export DFT=$MODELS/qwen2.5-0.5b-instruct-q4_k_m.gguf   # draft must share the target's tokenizer
```

### 6.1 Strong scaling (DEFAULT) — distinct work, the real metric

Slices a corpus into **distinct prompts** and spreads them across replicas (work-stealing). This is
the real offline-batch number: *how much faster N GPUs chew through the same dataset.*

```bash
# corpus to slice (e.g. first 64 KB of wikitext -> 16 x 4 KB prompts)
head -c 65536 $WIKI > /tmp/corpus.txt

# (A) 1-GPU serial baseline over all the prompts  (--dp-chunk-chars engages the batch at -dp 1)
export CUDA_VISIBLE_DEVICES=0
$BIN/llama-speculative -m $TGT -md $DFT -ngl 99 -ngld 99 \
    -c 8192 -b 2048 -ub 512 -n 256 --spec-draft-n-max 8 --spec-draft-n-min 2 \
    --dp-chunk-chars 4096 --data-parallel 1 -f /tmp/corpus.txt

# (B) 4-GPU work-stealing over the SAME prompts
export CUDA_VISIBLE_DEVICES=0,1,2,3
$BIN/llama-speculative -m $TGT -md $DFT -ngl 99 -ngld 99 \
    -c 8192 -b 2048 -ub 512 -n 256 --spec-draft-n-max 8 --spec-draft-n-min 2 \
    --dp-chunk-chars 4096 --data-parallel 4 --dp-devices 0,1,2,3 -f /tmp/corpus.txt
```

**What to read:** the strong run prints per-replica `jobs` (the work balance) and
`aggregate decode … t/s`. **Real speedup = baseline wall ÷ parallel wall.** More prompts (bigger
corpus) → finer work-stealing → closer to N×.

| Knob | Effect |
|------|--------|
| `--spec-draft-n-max` | Tokens the draft proposes per step. Sweet spot varies by pair (try 4 / 8 / 16). |
| `--dp-chunk-chars` | Bytes per shard (prompt). Smaller = more, shorter prompts = better balance. |
| corpus size (`head -c`) | More bytes → more prompts → better work-stealing, longer run. |

### 6.2 Weak scaling (DIAGNOSTIC) — `--dp-weak`

Every replica runs the **identical** stream (same prompt). This measures scaling *efficiency* (best
case ~N×), **not** real throughput — use it to check the orchestrator's overhead, not to report
throughput on real data.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
$BIN/llama-speculative -m $TGT -md $DFT -ngl 99 -ngld 99 \
    -c 8192 -b 2048 -ub 512 -n 256 --spec-draft-n-max 8 --spec-draft-n-min 2 \
    -s 42 --dp-weak --data-parallel 4 --dp-devices 0,1,2,3 -f /tmp/corpus.txt
```

Pass a fixed `-s SEED` so every replica runs the same deterministic stream → each replica's
`accept%` matches the 1-GPU baseline and the `ALL` row is a clean ~N×.

---

## 7. GPU placement recipes

Works with **any** of the three tools — just swap the placement flags.

**One replica per GPU (default, max throughput on models that fit one card):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3   ... -dp 4 --dp-devices 0,1,2,3
# or simply: -dp 4   (defaults to GPUs 0..N-1)
```

**Oversubscription (small model, spare memory → more replicas than GPUs):**
```bash
# 8 replicas on 4 GPUs (2 each):
CUDA_VISIBLE_DEVICES=0,1,2,3   ... -dp 8 --dp-devices 0,0,1,1,2,2,3,3
# equivalently, by count:
CUDA_VISIBLE_DEVICES=0,1,2,3   ... --dp-num-devices 4 --dp-replicas-per-device 2
```
Each replica needs weights + KV cache to fit; back off if you hit OOM (see §10).

**Multi-GPU replica (model too big for one card):**
```bash
# 2 replicas, each spanning 2 GPUs:
CUDA_VISIBLE_DEVICES=0,1,2,3   ... -dp 2 --dp-devices 0+1,2+3 --dp-split layer
# try --dp-split row or tensor to compare
```

---

## 8. Multi-node (cross-node throughput)

Uses the **MPI build** (`build-mpi`). One rank per node, each running its own replica pool; the head
node prints per-`NODE` rows plus a combined `CL` (cluster) row. No code differences — same workload,
more nodes.

The ready-made script is easiest:
```bash
ml gcc cuda hpcx
sbatch tools/orchestrator/slurm/dp-batched-bench-cluster.slurm   # edit nodelist/models inside
```

Manual launch (2 nodes, 4 GPUs each = 8 GPUs):
```bash
srun --partition=workers --nodes=2 --nodelist=<nodeA>,<nodeB> --ntasks-per-node=1 \
     --mpi=pmix --export=ALL --kill-on-bad-exit=1 \
     $REPO/build-mpi/bin/llama-batched-bench -m $MODELS/<model>.gguf \
     -c 65536 -b 4096 -ub 2048 -ngl 99 -npp 512 -ntg 128 -npl 1,2,4,8,16,32,64 \
     --data-parallel 4 --dp-devices 0,1,2,3
```
> If `--mpi=pmix` errors, try `--mpi=pmi2`. The `CL` row's throughput should be ~(#nodes)× a single node.

---

## 9. Ready-made Slurm scripts

Pre-built harnesses in `tools/orchestrator/slurm/` — open one, fill in the model paths / nodelist at
the top, then `sbatch` it. Good starting points:

| Script | What it runs |
|--------|--------------|
| `dp-batched-bench.slurm` | 1 GPU vs 4 GPU `-dp 4` throughput sweep (the showcase). |
| `dp-orchestrator.slurm` | Perplexity: 1 GPU vs 4 GPU `-dp 4` (correctness + speed). |
| `dp-speculative.slurm` | Speculative **strong** scaling: 1-GPU serial vs 4-GPU, prints real speedup. |
| `dp-oversubscribe.slurm` | Multiple replicas per GPU sweep (small model). |
| `dp-split.slurm` | Multi-GPU split replica (model too big for one card). |
| `dp-batched-bench-cluster.slurm` | **Multi-node** batched-bench (MPI, `srun --mpi=pmix`). |
| `baseline-original.slurm` | Stock single-GPU reference numbers. |

Logs land in `results/<job>-<jobid>/` and the job's `*.log`.

---

## 10. Reading the output

- **`ALL` row** (batched-bench) — the aggregate across replicas. `S t/s` = max-based throughput,
  `S_e2e t/s` = end-to-end (wall-clock) throughput. Both should approach N× the 1-GPU `S t/s`.
- **`CL` row** (multi-node) — the cluster total; per-`NODE` rows above it should each match a
  single-node run.
- **`REAL STRONG SPEEDUP`** (speculative strong) — baseline-wall ÷ parallel-wall on the same prompts.
- **`Final estimate: PPL`** (perplexity) — must match between 1-GPU and DP runs.

---

## 11. Troubleshooting

| Symptom | Fix |
|---------|-----|
| Slow / "no usable GPU" / runs on CPU | Add `-ngl 99` (and check `CUDA_VISIBLE_DEVICES`). |
| Out of memory with `-dp` | Fewer replicas, fewer replicas-per-GPU, or use a multi-GPU split replica (`--dp-devices 0+1,...`). Each replica needs its own weights + KV. |
| Multi-node: `Cannot load module hpcx` | Load `cuda` **before** `hpcx`: `ml gcc cuda hpcx`. |
| Multi-node: rank can't find `libcudart` / hangs | Launch with `srun --export=ALL --kill-on-bad-exit=1`. |
| Multi-node: `--mpi=pmix` rejected | Use `--mpi=pmi2`. |
| `srun: Requested node configuration not available` | The node may be in a non-default partition — add `--partition=workers`. |
| Speculative: "draft model … must match target" | Target and draft must share the same tokenizer/vocab (e.g. both Qwen2.5). |

---

## 12. Scope (what it does **not** do)

- It scales **throughput**, not single-request **latency** — one request is not faster.
- It is for **offline / batch** tools (perplexity, batched-bench, speculative). It is **not** wired
  into `llama-cli` or `llama-server` (the server already has its own concurrent-request handling).
- Multi-GPU **split** is a capacity escape hatch (fit a big model); throughput still comes from
  running **more replicas**, not from splitting.

---

*Questions on a specific run? Capture the full `*.log` and the exact command — that's everything
needed to diagnose.*
