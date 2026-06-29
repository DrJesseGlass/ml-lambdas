#!/usr/bin/env bash
# N1 A/B: decode (GEMV) software prefetch distance sweep. Decode is memory-LATENCY
# bound on N1; CANDLE_DECODE_PREFETCH=<bytes> pulls the weight stream ahead into L1
# (0 = off). Expect a win at 1-2 threads (latency-bound) and little/none at 4
# (bandwidth-saturated - prefetch hides latency, not bandwidth).
#
# Decode-heavy: short prefill (pp=8), long gen (tg=256), more reps (decode is noisy).
# Lanerow+parquant+par-elemwise all on; only CANDLE_DECODE_PREFETCH varies. Plain
# Q4_K + CANDLE_MATMUL_PACKED_Q4K=1 so decode runs the prefetched gemm_q4kx_q8k MR=1.
# Run on the N1 box. Usage: THREADS="1 2 4" DISTS="0 256 512 1024" ./bench/decode_prefetch_ab.sh
set -uo pipefail
B=${B:-$HOME/candle/target/release/examples/quantized-qwen3-bench}
G=${G:-$HOME/ml-lambdas/qwen-lambda/models/Qwen3-0.6B-Q4_K_M.gguf}
THREADS=${THREADS:-"1 2 4"}
DISTS=${DISTS:-"0 256 512 1024"}
TG=${TG:-256}
REPS=${REPS:-5}

[ -x "$B" ] || { echo "no bench binary at $B"; exit 1; }
[ -f "$G" ] || { echo "no model at $G"; exit 1; }

tg_of() { # threads dist -> tg_tok_s_median
  local t=$1 d=$2
  local out
  out=$(env RAYON_NUM_THREADS="$t" CANDLE_NUM_THREADS="$t" \
      CANDLE_QMATMUL_DECODE_THREADS="$t" CANDLE_QMATMUL_PREFILL_THREADS="$t" \
      CANDLE_MATMUL_PACKED_Q4K=1 CANDLE_PREFILL_LANEROW=1 CANDLE_PREFILL_PARQUANT=1 \
      CANDLE_PAR_ELEMWISE=1 CANDLE_DECODE_PREFETCH="$d" CANDLE_KV_PREALLOC=512 \
      taskset -c "0-$((t-1))" "$B" --model "$G" --pp 8 --tg "$TG" --reps "$REPS" --warmup 1 --json)
  sed -nE 's/.*"tg_tok_s_median":([0-9.]+).*/\1/p' <<<"$out"
}

echo "model=$(basename "$G")  tg=$TG  reps=$REPS  host=$(hostname)"
printf '%-4s' thr; for d in $DISTS; do printf ' %-9s' "pf=$d"; done; printf ' %-8s\n' best
printf '%s\n' "----------------------------------------------------------------------"
for t in $THREADS; do
  printf '%-4s' "$t"
  base=""; best_d=""; best_v=0
  for d in $DISTS; do
    v=$(tg_of "$t" "$d")
    [ "$d" = "$(echo "$DISTS" | awk '{print $1}')" ] && base="$v"
    sp=$(awk -v a="$v" -v b="$base" 'BEGIN{printf (b>0)?"%.0f(%.2fx)":"%s", a, (b>0?a/b:0)}')
    printf ' %-9s' "$sp"
    awk -v a="$v" -v b="$best_v" 'BEGIN{exit !(a>b)}' && { best_v="$v"; best_d="$d"; }
  done
  printf ' pf=%-5s\n' "$best_d"
done
echo
echo "cell = tg_tok_s (ratio vs pf=0). 'best' = prefetch distance with highest tg t/s."
echo "Expect a >1.0x win at 1-2 threads; ~flat at 4 (bandwidth-saturated)."
