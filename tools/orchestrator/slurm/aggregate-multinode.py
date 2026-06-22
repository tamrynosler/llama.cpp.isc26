#!/usr/bin/env python3
"""Aggregate multi-node data-parallel batched-bench JSONL into a full-cluster throughput table.

Each node ran the single-node DP sweep independently (dp-multinode-srun.slurm), so per-config
cluster throughput is the SUM of per-node throughput, and node-scaling = cluster / fastest-node
(ideal = number of nodes; a value below that exposes a lagging node).

Usage: aggregate-multinode.py <results-dir>
Reads <results-dir>/<model>.node-<host>.jsonl files (one DP-aggregate JSON object per config).
Uses each object's "e2e_speed" (end-to-end drain-span aggregate tok/s) as the per-node number.
"""
import glob
import json
import math
import os
import sys
from collections import defaultdict


def main():
    if len(sys.argv) != 2:
        print("usage: aggregate-multinode.py <results-dir>", file=sys.stderr)
        return 1
    results_dir = sys.argv[1]
    files = sorted(glob.glob(os.path.join(results_dir, "*.node-*.jsonl")))
    if not files:
        print(f"no *.node-*.jsonl files in {results_dir}", file=sys.stderr)
        return 1

    # rows[model][(pp, tg, pl)][node] = e2e_speed
    rows = defaultdict(lambda: defaultdict(dict))
    nodes = set()
    for path in files:
        base = os.path.basename(path)
        model, sep, rest = base.partition(".node-")
        if not sep:
            continue
        node = rest[:-len(".jsonl")] if rest.endswith(".jsonl") else rest
        nodes.add(node)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (o["pp"], o["tg"], o["pl"])
                rows[model][key][node] = float(o.get("e2e_speed", 0.0))

    node_list = sorted(nodes)
    n_nodes = len(node_list)
    print(f"nodes ({n_nodes}): {', '.join(node_list)}")
    print(f"metric: e2e aggregate throughput (tok/s); cluster = sum over nodes; "
          f"ideal node-scaling = {n_nodes}.0x (cluster / fastest node)\n")

    for model in sorted(rows):
        print(f"=== {model} ===")
        header = (f"| {'pp':>5} | {'tg':>4} | {'pl':>3} | "
                  + " | ".join(f"{n:>12}" for n in node_list)
                  + f" | {'cluster t/s':>12} | {'scaling':>8} |")
        print(header)
        print("|" + "-" * (len(header) - 2) + "|")
        for key in sorted(rows[model]):
            pp, tg, pl = key
            vals = [rows[model][key].get(n, math.nan) for n in node_list]
            present = [v for v in vals if not math.isnan(v)]
            total = sum(present)
            scaling = total / max(present) if present else 0.0
            cells = " | ".join(f"{v:12.1f}" if not math.isnan(v) else f"{'-':>12}" for v in vals)
            print(f"| {pp:>5} | {tg:>4} | {pl:>3} | {cells} | {total:12.1f} | {scaling:7.2f}x |")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
