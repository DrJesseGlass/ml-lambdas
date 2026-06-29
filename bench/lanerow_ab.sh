#!/usr/bin/env bash
# N1 A/B: our lane=row Q4_K prefill kernel vs UPSTREAM #3643's BlockQ4Kx8 kernel,
# SAME binary, toggled by CANDLE_PREFILL_LANEROW. This is the head-to-head the
# coexist design asks for: on the q6k-packed branch the matmul dispatch
# (quantized/mod.rs) routes Q4_K prefill (m>=4, n%8==0) to lane=row when
# CANDLE_PREFILL_LANEROW=1 (default) and FALLS THROUGH to #3643 when =0.
#
#   =0  -> upstream #3643 (pack_to_q4kx8 + vec_dot_8_q4k_q8k), the candle main path
#   =1  -> ours (lane=row 8x4 SDOT, gemm_q4kx8_q8k_lanerow)
#
# Clean wall-clock prefill t/s (warmup=steady state, median). Decode t/s printed as
# a sanity check: the toggle only touches m>=4, so decode (m=1) MUST be identical
# both ways (both use #3643). Run on the N1 (Graviton2) box. Usage:
#   THREADS="1 2 4 6" PP="512 2048" ./bench/lanerow_ab.sh
set -euo pipefail
B=${B:-$HOME/candle/target/release/examples/quantized-qwen3-bench}
G=${G:-$HOME/ml-lambdas/qwen-lambda/models/Qwen3-0.6B-Q4_K_M.gguf}
THREADS=${THREADS:-"1 2 4 6"}
PP=${PP:-"512 2048"}
REPS=${REPS:-3}

[ -x "$B" ] || { echo "no bench binary at $B"; exit 1; }
[ -f "$G" ] || { echo "no model at $G"; exit 1; }

run() { # threads pp lanerow  -> prints the bench JSON line
  local t=$1 pp=$2 lr=$3
  env RAYON_NUM_THREADS="$t" CANDLE_NUM_THREADS="$t" CANDLE_KV_PREALLOC=512 \
      CANDLE_PREFILL_LANEROW="$lr" \
      taskset -c "0-$((t-1))" "$B" --model "$G" --pp "$pp" --tg 8 --reps "$REPS" --warmup 1 --json
}
jget() { sed -nE "s/.*\"$2\":([0-9.]+).*/\1/p" <<<"$1"; }

echo "model=$(basename "$G")  reps=$REPS  host=$(hostname)  ($(nproc) vCPU)"
echo "  =0 upstream#3643   =1 ours(lane=row)"
printf '%-4s %-6s %-14s %-14s %-9s %-15s\n' thr pp '#3643_pp' 'lanerow_pp' 'ours/3643' 'decode 3643/ours'
printf '%s\n' "-----------------------------------------------------------------------------"
for t in $THREADS; do
  for pp in $PP; do
    u=$(run "$t" "$pp" 0); l=$(run "$t" "$pp" 1)
    upp=$(jget "$u" pp_tok_s_median); lpp=$(jget "$l" pp_tok_s_median)
    utg=$(jget "$u" tg_tok_s_median); ltg=$(jget "$l" tg_tok_s_median)
    spd=$(awk -v a="$lpp" -v b="$upp" 'BEGIN{printf (b>0)?"%.3fx":"-", a/b}')
    printf '%-4s %-6s %-14s %-14s %-9s %s/%s\n' "$t" "$pp" "$upp" "$lpp" "$spd" "$utg" "$ltg"
  done
done
