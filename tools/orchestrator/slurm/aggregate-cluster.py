#!/usr/bin/env python3
"""Aggregate multi-node data-parallel batched-bench JSONL into a full-cluster throughput table.

Each node ran the stock llama-batched-bench with --data-parallel N across its local GPUs
(node-local library DP, no cross-node communication during inference). Per-config CLUSTER
throughput is therefore the SUM of the per-node throughput; node-scaling = cluster / fastest-node
(ideal = number of nodes; below that exposes a lagging node).

Usage: aggregate-cluster.py <results-dir>
Reads <results-dir>/node-*.jsonl (one JSON object per (pp,tg,pl) config per node, from the stock
batched-bench --output-format jsonl).
"""
import glob
import json
import os
import sys
from collections import defaultdict


def main():
    if len(sys.argv) != 2:
        print("usage: aggregate-cluster.py <results-dir>", file=sys.stderr)
        return 1

    results_dir = sys.argv[1]
    files = sorted(glob.glob(os.path.join(results_dir, "node-*.jsonl")))
    if not files:
        print(f"no node-*.jsonl files in {results_dir}", file=sys.stderr)
        return 1

    # rows[(pp,tg,pl)] accumulates the per-node throughput for that config.
    rows = defaultdict(lambda: {
        "n_kv": 0, "nodes": 0,
        "speed_pp": 0.0, "speed_tg": 0.0, "speed": 0.0,
        "per_node_speed": [],
    })
    order = []

    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "pp" not in o or "tg" not in o or "pl" not in o:
                    continue
                key = (o["pp"], o["tg"], o["pl"])
                if key not in rows:
                    order.append(key)
                r = rows[key]
                r["n_kv"]      = o.get("n_kv", r["n_kv"])
                r["nodes"]    += 1
                r["speed_pp"] += o.get("speed_pp", 0.0)
                r["speed_tg"] += o.get("speed_tg", 0.0)
                r["speed"]    += o.get("speed", 0.0)
                r["per_node_speed"].append(o.get("speed", 0.0))

    n_nodes = len(files)
    print(f"CLUSTER aggregate over {n_nodes} node(s)")
    print()
    hdr = (f"|{'PP':>6} |{'TG':>6} |{'B':>4} |{'N_KV':>7} |"
           f"{'S_PP t/s':>11} |{'S_TG t/s':>11} |{'S t/s (sum)':>13} |{'nodes':>6} |{'scaling':>8} |")
    sep = "|" + "-" * (len(hdr) - 2) + "|"
    print(hdr)
    print(sep)

    for key in order:
        pp, tg, pl = key
        r = rows[key]
        best = max(r["per_node_speed"]) if r["per_node_speed"] else 0.0
        scaling = (r["speed"] / best) if best > 0 else 0.0
        print(f"|{pp:>6} |{tg:>6} |{pl:>4} |{r['n_kv']:>7} |"
              f"{r['speed_pp']:>11.2f} |{r['speed_tg']:>11.2f} |{r['speed']:>13.2f} |"
              f"{r['nodes']:>6} |{scaling:>7.2f}x |")

    return 0


if __name__ == "__main__":
    sys.exit(main())
