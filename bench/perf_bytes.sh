#!/usr/bin/env bash
# BYTES-MOVED per token, candle vs llama, on N1 - the right metric for the memory-bound
# parts (decode). Memory traffic = cache-line refills * 64B (N1 line = 64B):
#   l1d_cache_refill*64 = bytes pulled into L1 (from L2+)   -> total near-core traffic
#   l2d_cache_refill*64 = bytes pulled into L2 (from L3/DRAM) -> FAR-memory / bandwidth traffic
# Reps-difference isolates the MARGINAL bytes of one token (cancels the ~446MB model load +
# fixed overhead, which otherwise swamps everything):
#   bytes/decode-tok  = [refill(tg=HI) - refill(tg=LO)] * 64 / (HI-LO)   @ fixed pp
#   bytes/prefill-tok = [refill(pp=HI) - refill(pp=LO)] * 64 / (HI-LO)   @ fixed tiny tg
# If candle moves MORE bytes/token than llama for the SAME weights, it is an access-pattern
# problem (cache thrash / redundant loads), not an algorithmic one - that is a candle lever.
set -euo pipefail
M=${M:-$HOME/ml-lambdas/qwen-lambda/models}
CB=${CB:-$HOME/candle/target/release/examples/quantized-qwen3-bench}
LB=${LB:-$HOME/llama.cpp/build/bin/llama-bench}
CMODEL=${CMODEL:-$M/Qwen3-0.6B-Q6packed.gguf}
LMODEL=${LMODEL:-$M/Qwen3-0.6B-Q4out.gguf}
EV=l1d_cache_refill:u,l2d_cache_refill:u
CENV=(CANDLE_QMATMUL_DECODE_THREADS=1 CANDLE_QMATMUL_PREFILL_THREADS=1 CANDLE_KV_PREALLOC=512)
LINE=64
TG_LO=64; TG_HI=320; PP_LO=128; PP_HI=512; PREF_TG=1

# echo "l1refills l2refills" for a run
refills() { perf stat -x, -e "$EV" "$@" >/dev/null 2>/tmp/pb.csv || true
  awk -F, '/l1d_cache_refill/{a=$1}/l2d_cache_refill/{b=$1}END{printf "%s %s",a,b}' /tmp/pb.csv; }
cand() { env "${CENV[@]}" taskset -c 0 "$CB" --model "$CMODEL" --pp "$1" --tg "$2" --reps 1 --warmup 0 --json; }
llam() { taskset -c 0 "$LB" -m "$LMODEL" -p "$1" -n "$2" -r 1 --no-warmup -o json; }

emit() { # label l1lo l2lo l1hi l2hi denom
  awk -v L="$1" -v a0="$2" -v b0="$3" -v a1="$4" -v b1="$5" -v d="$6" -v line="$LINE" \
   'BEGIN{printf "%-16s L1-bytes/tok=%10.0f  FARmem-bytes/tok=%10.0f\n", L, line*(a1-a0)/d, line*(b1-b0)/d}'
}

echo "## DECODE bytes/token (pp=$PP_LO fixed; tg $TG_LO->$TG_HI)"
read a0 b0 <<<"$(refills env "${CENV[@]}" taskset -c 0 "$CB" --model "$CMODEL" --pp $PP_LO --tg $TG_LO --reps 1 --warmup 0 --json)"
read a1 b1 <<<"$(refills env "${CENV[@]}" taskset -c 0 "$CB" --model "$CMODEL" --pp $PP_LO --tg $TG_HI --reps 1 --warmup 0 --json)"
emit "candle decode" "$a0" "$b0" "$a1" "$b1" "$((TG_HI-TG_LO))"
read a0 b0 <<<"$(refills taskset -c 0 "$LB" -m "$LMODEL" -p $PP_LO -n $TG_LO -r 1 --no-warmup -o json)"
read a1 b1 <<<"$(refills taskset -c 0 "$LB" -m "$LMODEL" -p $PP_LO -n $TG_HI -r 1 --no-warmup -o json)"
emit "llama decode" "$a0" "$b0" "$a1" "$b1" "$((TG_HI-TG_LO))"

echo "## PREFILL bytes/token (tg=$PREF_TG fixed; pp $PP_LO->$PP_HI)"
read a0 b0 <<<"$(refills env "${CENV[@]}" taskset -c 0 "$CB" --model "$CMODEL" --pp $PP_LO --tg $PREF_TG --reps 1 --warmup 0 --json)"
read a1 b1 <<<"$(refills env "${CENV[@]}" taskset -c 0 "$CB" --model "$CMODEL" --pp $PP_HI --tg $PREF_TG --reps 1 --warmup 0 --json)"
emit "candle prefill" "$a0" "$b0" "$a1" "$b1" "$((PP_HI-PP_LO))"
read a0 b0 <<<"$(refills taskset -c 0 "$LB" -m "$LMODEL" -p $PP_LO -n $PREF_TG -r 1 --no-warmup -o json)"
read a1 b1 <<<"$(refills taskset -c 0 "$LB" -m "$LMODEL" -p $PP_HI -n $PREF_TG -r 1 --no-warmup -o json)"
emit "llama prefill" "$a0" "$b0" "$a1" "$b1" "$((PP_HI-PP_LO))"
