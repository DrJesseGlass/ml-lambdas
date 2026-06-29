#!/usr/bin/env bash
# Multi-thread PREFILL stall profile on N1: settle whether candle's remaining
# multi-thread prefill scaling loss is Amdahl (the serial activation quant) or
# genuine memory-bandwidth contention on the parallel GEMM.
#
# Runs the lane=row prefill path (plain Q4_K + CANDLE_MATMUL_PACKED_Q4K=1) at
# THREADS counts, A/B'ing CANDLE_PREFILL_PARQUANT=1 (parallel activation quant)
# vs =0 (serial). For each combo: wall-clock prefill t/s (-> scaling efficiency)
# plus user-space perf counters (IPC, backend-stall%, and L2-refills-per-million-
# instructions - the contention signal).
#
# READ THE RESULT LIKE THIS:
#   - PARQUANT=1 scales ~linearly AND L2/Mi + IPC stay ~flat across threads
#       => the serial quant WAS the bottleneck; parquant fixes it, done.
#   - PARQUANT=1 STILL scales poorly, L2/Mi RISES + IPC FALLS with threads
#       => the GEMM is memory-bandwidth bound (cores thrash the shared SLC/DRAM,
#          like the old decode finding); needs cache-aware tiling, not quant.
#
# Counters are user-space (:u). perf counts the whole (taskset-pinned) process,
# so the ratios (IPC, stall%, L2/Mi) are warmup-insensitive; t/s is the bench's
# own median over measured reps. Prefill-isolated: tg=1.
#
# Usage (on the N1 box):
#   ./bench/perf_mt_prefill.sh                       # PP="512" THREADS="1 2 4"
#   PP="512 2048" THREADS="1 2 4 6" ./bench/perf_mt_prefill.sh
#
# NOTE: no `set -e` on purpose - perf/bench calls are wrapped and tolerated; a
# stray non-zero (e.g. `read` with no trailing newline) must NOT abort the sweep.
set -uo pipefail
M=${M:-$HOME/ml-lambdas/qwen-lambda/models}
B=${B:-$HOME/candle/target/release/examples/quantized-qwen3-bench}
G=${G:-$M/Qwen3-0.6B-Q4_K_M.gguf}   # PLAIN Q4_K (the runtime-repack lanerow path)
PP=${PP:-512}                       # may be a list, e.g. "512 2048"
THREADS=${THREADS:-"1 2 4"}
REPS=${REPS:-3}
EV=cycles:u,instructions:u,stalled-cycles-backend:u,stalled-cycles-frontend:u,l1d_cache_refill:u,l2d_cache_refill:u

[ -x "$B" ] || { echo "ERROR: no bench binary at $B (build with RUSTFLAGS=-C target-cpu=native)"; exit 1; }
[ -f "$G" ] || { echo "ERROR: no model at $G"; exit 1; }
command -v perf >/dev/null || { echo "ERROR: perf not found (install linux-perf / perf)"; exit 1; }
grep -qw asimddp /proc/cpuinfo || echo "WARN: no dotprod in /proc/cpuinfo - lanerow path won't engage"

# ---- perf preflight: prove perf can actually count user events, else say why ----
pf=$(perf stat -x, -e instructions:u -- true 2>&1 | awk -F, '/instructions/{print $1}')
if ! [[ "${pf:-}" =~ ^[0-9] ]]; then
  echo "ERROR: perf cannot count user-space events. Raw probe output:"
  perf stat -x, -e instructions:u -- true 2>&1 | sed 's/^/    /'
  echo "  Likely kernel.perf_event_paranoid is too high. Fix on the box with:"
  echo "    sudo sysctl kernel.perf_event_paranoid=-1"
  echo "  (or =1) then re-run. Aborting."
  exit 1
fi
echo "perf preflight ok (instructions:u counted)."

tmpc=$(mktemp); tmpj=$(mktemp)
trap 'rm -f "$tmpc" "$tmpj"' EXIT

# Echo one space-separated row (WITH trailing newline): instr ipc bstall l2 l2mi pps
measure() { # threads parquant pp
  local t=$1 pq=$2 pp=$3
  env RAYON_NUM_THREADS="$t" CANDLE_NUM_THREADS="$t" \
      CANDLE_QMATMUL_PREFILL_THREADS="$t" CANDLE_QMATMUL_DECODE_THREADS="$t" \
      CANDLE_MATMUL_PACKED_Q4K=1 CANDLE_PREFILL_LANEROW=1 CANDLE_PREFILL_PARQUANT="$pq" \
      CANDLE_KV_PREALLOC=512 \
      perf stat -x, -e "$EV" taskset -c "0-$((t-1))" \
        "$B" --model "$G" --pp "$pp" --tg 1 --reps "$REPS" --warmup 1 --json \
        >"$tmpj" 2>"$tmpc" || true
  local pps
  pps=$(sed -nE 's/.*"pp_tok_s_median":([0-9.]+).*/\1/p' "$tmpj")
  awk -F, -v pps="${pps:-nan}" '
    /[^_]cycles:u/{c=$1} /instructions:u/{i=$1}
    /stalled-cycles-backend/{sb=$1} /l1d_cache_refill/{l1=$1} /l2d_cache_refill/{l2=$1}
    END{ printf "%.4g %.3f %.1f %.4g %.2f %s\n", i, (c>0?i/c:0), (c>0?100*sb/c:0),
         l2, (i>0?1e6*l2/i:0), pps }' "$tmpc"
}

echo "model=$(basename "$G")  pp=[$PP]  tg=1  reps=$REPS  host=$(hostname)"
printf '%-4s %-4s %-6s %-10s %-6s %-9s %-10s %-11s %-9s %-8s\n' \
  thr pq pp instr IPC bstall% L2refill L2/Minstr pp_t/s scaling
printf '%s\n' "-------------------------------------------------------------------------------------------"

first_thr=$(awk '{print $1}' <<<"$THREADS")
for pq in 1 0; do
  tag=$([ "$pq" = 1 ] && echo par || echo ser)
  for pp in $PP; do
    base=""
    for t in $THREADS; do
      row=$(measure "$t" "$pq" "$pp")
      read -r instr ipc bst l2 l2mi pps <<<"$row" || true
      [ "$t" = "$first_thr" ] && base="$pps"
      scal=$(awk -v a="${pps:-nan}" -v b="${base:-nan}" \
             'BEGIN{ if (b+0>0) printf "%.2fx", a/b; else printf "-" }')
      printf '%-4s %-4s %-6s %-10s %-6s %-9s %-10s %-11s %-9s %-8s\n' \
        "$t" "$tag" "$pp" "$instr" "$ipc" "$bst" "$l2" "$l2mi" "$pps" "$scal"
    done
  done
done
echo
echo "scaling = pp_t/s at N threads / pp_t/s at the first THREADS value, same (pq,pp)."
echo "Watch L2/Minstr across threads on the 'par' rows: flat=not bandwidth-bound, rising=contention."
