#!/usr/bin/env bash
# Per-matmul-shape breakdown of the decode GEMV from an nsys capture.
#
# nsys's name-aggregated view collapses all mul_mat_vec_q launches into one row. This decomposes
# them by gridX (= output rows, since mmvq uses rows_per_cuda_block=1 at batch=1), which maps each
# bucket to a specific matmul (q/k/v/o proj, FFN gate-up/down, LM head). With per-shape time we
# estimate achieved bandwidth and see which matmuls are bandwidth-bound vs launch/occupancy-bound.
#
# Usage:
#   decode-temp/nsys-shape-breakdown.sh <capture.nsys-rep | capture.sqlite>
# Examples:
#   decode-temp/nsys-shape-breakdown.sh decode-profile-1197/dense_d0_graphs_on.nsys-rep
#   decode-temp/nsys-shape-breakdown.sh decode-profile-1197/dense_d8192_graphs_on.sqlite

set -uo pipefail

IN="${1:?usage: nsys-shape-breakdown.sh <capture.nsys-rep|.sqlite>}"

# Resolve to a .sqlite (generate from .nsys-rep if needed; nsys stats creates it next to the rep).
case "$IN" in
    *.sqlite)   SQLITE="$IN" ;;
    *.nsys-rep) SQLITE="${IN%.nsys-rep}.sqlite"
                [ -f "$SQLITE" ] || nsys stats "$IN" >/dev/null 2>&1 || true ;;
    *)          echo "ERROR: expected a .nsys-rep or .sqlite file" >&2; exit 1 ;;
esac
[ -f "$SQLITE" ] || { echo "ERROR: no sqlite ($SQLITE). Run: nsys stats $IN" >&2; exit 1; }
command -v sqlite3 >/dev/null 2>&1 || { echo "ERROR: sqlite3 not on PATH" >&2; exit 1; }

echo "# GEMV (mul_mat_vec_q) by output shape -- $SQLITE"
echo "# gridX = output rows; n = launches; total_ms over the capture; avg_ns per launch"
sqlite3 -header -column "$SQLITE" "
SELECT
  CASE WHEN d.value LIKE '%(ggml_type)14%' THEN 'Q6_K'
       WHEN d.value LIKE '%(ggml_type)2%'  THEN 'Q4_0'
       ELSE 'other' END               AS qtype,
  k.gridX, k.blockX, k.blockY,
  COUNT(*)                            AS n,
  ROUND(SUM(k.end-k.start)/1e6, 2)    AS total_ms,
  ROUND(AVG(k.end-k.start), 0)        AS avg_ns
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds d ON d.id = k.demangledName
WHERE d.value LIKE '%mul_mat_vec_q%'
GROUP BY qtype, k.gridX, k.blockX, k.blockY
ORDER BY total_ms DESC;"

echo ""
echo "# Flash-attention kernels by shape (decode attention cost; grows with KV depth)"
sqlite3 -header -column "$SQLITE" "
SELECT s.value AS kernel, COUNT(*) AS n, ROUND(SUM(k.end-k.start)/1e6,2) AS total_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON s.id = k.shortName
WHERE s.value LIKE '%flash_attn%'
GROUP BY s.value ORDER BY total_ms DESC;"
