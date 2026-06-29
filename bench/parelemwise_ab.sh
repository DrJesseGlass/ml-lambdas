#!/usr/bin/env bash
# N1 A/B: parallel elementwise ops ON vs OFF, SAME binary, lanerow prefill path.
# Tests whether threading the serial elementwise ops (SwiGLU mul, SiLU, residual
# adds) recovers multi-thread prefill scaling. Decode t/s printed too (elementwise
# is in decode as well, so it may move - unlike the lanerow toggle).
#
# Both arms run lanerow (CANDLE_PREFILL_LANEROW=1) + parquant on; only
# CANDLE_PAR_ELEMWISE differs. Plain Q4_K + CANDLE_MATMUL_PACKED_Q4K=1.
# Run on the N1 box. Usage: THREADS="1 2 4" PP=512 ./bench/parelemwise_ab.sh
set -uo pipefail
B=${B:-$HOME/candle/target/release/examples/quantized-qwen3-bench}
G=${G:-$HOME/ml-lambdas/qwen-lambda/models/Qwen3-0.6B-Q4_K_M.gguf}
THREADS=${THREADS:-"1 2 4"}
PP=${PP:-512}
REPS=${REPS:-3}

[ -x "$B" ] || { echo "no bench binary at $B"; exit 1; }
[ -f "$G" ] || { echo "no model at $G"; exit 1; }

run() { # threads pp parelem -> bench JSON line
  local t=$1 pp=$2 pe=$3
  env RAYON_NUM_THREADS="$t" CANDLE_NUM_THREADS="$t" \
      CANDLE_QMATMUL_PREFILL_THREADS="$t" CANDLE_QMATMUL_DECODE_THREADS="$t" \
      CANDLE_MATMUL_PACKED_Q4K=1 CANDLE_PREFILL_LANEROW=1 CANDLE_PREFILL_PARQUANT=1 \
      CANDLE_PAR_ELEMWISE="$pe" CANDLE_KV_PREALLOC=512 \
      taskset -c "0-$((t-1))" "$B" --model "$G" --pp "$pp" --tg 8 --reps "$REPS" --warmup 1 --json
}
jget() { sed -nE "s/.*\"$2\":([0-9.]+).*/\1/p" <<<"$1"; }

echo "model=$(basename "$G")  pp=$PP  reps=$REPS  host=$(hostname)"
printf '%-4s %-12s %-12s %-9s %-12s %-12s %-9s\n' \
  thr pp_off pp_on pp_spdup tg_off tg_on tg_spdup
printf '%s\n' "----------------------------------------------------------------------------"
base_off=""; base_on=""
for t in $THREADS; do
  off=$(run "$t" "$PP" 0); on=$(run "$t" "$PP" 1)
  po=$(jget "$off" pp_tok_s_median); pn=$(jget "$on" pp_tok_s_median)
  to=$(jget "$off" tg_tok_s_median); tn=$(jget "$on" tg_tok_s_median)
  ps=$(awk -v a="$pn" -v b="$po" 'BEGIN{printf (b>0)?"%.3fx":"-", a/b}')
  ts=$(awk -v a="$tn" -v b="$to" 'BEGIN{printf (b>0)?"%.3fx":"-", a/b}')
  printf '%-4s %-12s %-12s %-9s %-12s %-12s %-9s\n' "$t" "$po" "$pn" "$ps" "$to" "$tn" "$ts"
done
echo
echo "pp_off/on = prefill t/s with CANDLE_PAR_ELEMWISE=0/1; spdup = on/off."
echo "Watch pp_spdup grow with threads (recovers multi-thread scaling) and whether tg moves."
