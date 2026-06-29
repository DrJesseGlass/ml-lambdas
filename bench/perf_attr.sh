#!/usr/bin/env bash
# Per-phase instruction / IPC / L1-miss attribution for candle vs llama.cpp, using
# the reps-difference method to cancel model-load + fixed overhead and isolate the
# marginal cost of (a) a decode token and (b) a prefill token.
#
#   decode token cost  = [run(tg=TG_HI) - run(tg=TG_LO)] / (TG_HI - TG_LO)   @ fixed pp
#   prefill token cost = [run(pp=PP_HI) - run(pp=PP_LO)] / (PP_HI - PP_LO)   @ fixed tg
#
# Counts are USER-SPACE (:u) - perf_event_paranoid=2 on the box. 1 thread, pinned.
set -euo pipefail

M=${M:-$HOME/ml-lambdas/qwen-lambda/models}
CB=${CB:-$HOME/candle/target/release/examples/quantized-qwen3-bench}
LB=${LB:-$HOME/llama.cpp/build/bin/llama-bench}
CMODEL=${CMODEL:-$M/Qwen3-0.6B-Q6packed.gguf}
LMODEL=${LMODEL:-$M/Qwen3-0.6B-Q4out.gguf}
# N1 PMU: generic L1-dcache-* map to 0; use the ARM event names.
EV=instructions:u,cycles:u,l1d_cache:u,l1d_cache_refill:u
PP_LO=${PP_LO:-128}; PP_HI=${PP_HI:-256}
TG_LO=${TG_LO:-64};  TG_HI=${TG_HI:-320}
PREF_TG=${PREF_TG:-1}   # tiny fixed decode so the prefill delta isn't contaminated by KV growth

CENV=(CANDLE_QMATMUL_DECODE_THREADS=1 CANDLE_QMATMUL_PREFILL_THREADS=1 CANDLE_KV_PREALLOC=512)

# pstat <cmd...> : run under perf stat, echo "instr cycles l1loads l1miss" (user-space).
pstat() {
  perf stat -x, -e "$EV" "$@" >/dev/null 2>/tmp/ps.csv || true
  awk -F, '/instructions:u/{i=$1}/cycles:u/{c=$1}/l1d_cache:u/{l=$1}/l1d_cache_refill:u/{m=$1}
           END{printf "%s %s %s %s", i, c, l, m}' /tmp/ps.csv
}
emit() { # label i0 c0 l0 m0 i1 c1 l1 m1 denom
  awk -v L="$1" -v i0="$2" -v c0="$3" -v l0="$4" -v m0="$5" -v i1="$6" -v c1="$7" -v l1="$8" -v m1="$9" -v d="${10}" \
    'BEGIN{di=(i1-i0)/d; dc=(c1-c0)/d; dl=(l1-l0)/d; dm=(m1-m0)/d;
      printf "%-16s instr/tok=%12.0f  IPC=%.2f  L1ld/tok=%11.0f  L1miss/tok=%10.0f  miss%%=%.2f%%\n",
        L, di, (dc>0?di/dc:0), dl, dm, (dl>0?100*dm/dl:0)}'
}

echo "## decode per-token (pp=$PP_LO fixed; tg $TG_LO->$TG_HI, delta=$((TG_HI-TG_LO)))"
read i0 c0 l0 m0 <<<"$(pstat env "${CENV[@]}" taskset -c 0 "$CB" --model "$CMODEL" --pp "$PP_LO" --tg "$TG_LO" --reps 1 --warmup 0 --json)"
read i1 c1 l1 m1 <<<"$(pstat env "${CENV[@]}" taskset -c 0 "$CB" --model "$CMODEL" --pp "$PP_LO" --tg "$TG_HI" --reps 1 --warmup 0 --json)"
emit "candle decode" "$i0" "$c0" "$l0" "$m0" "$i1" "$c1" "$l1" "$m1" "$((TG_HI-TG_LO))"
read i0 c0 l0 m0 <<<"$(pstat taskset -c 0 "$LB" -m "$LMODEL" -p "$PP_LO" -n "$TG_LO" -r 1 -o json)"
read i1 c1 l1 m1 <<<"$(pstat taskset -c 0 "$LB" -m "$LMODEL" -p "$PP_LO" -n "$TG_HI" -r 1 -o json)"
emit "llama decode" "$i0" "$c0" "$l0" "$m0" "$i1" "$c1" "$l1" "$m1" "$((TG_HI-TG_LO))"

echo
echo "## prefill per-token (tg=$PREF_TG fixed; pp $PP_LO->$PP_HI, delta=$((PP_HI-PP_LO)))"
read i0 c0 l0 m0 <<<"$(pstat env "${CENV[@]}" taskset -c 0 "$CB" --model "$CMODEL" --pp "$PP_LO" --tg "$PREF_TG" --reps 1 --warmup 0 --json)"
read i1 c1 l1 m1 <<<"$(pstat env "${CENV[@]}" taskset -c 0 "$CB" --model "$CMODEL" --pp "$PP_HI" --tg "$PREF_TG" --reps 1 --warmup 0 --json)"
emit "candle prefill" "$i0" "$c0" "$l0" "$m0" "$i1" "$c1" "$l1" "$m1" "$((PP_HI-PP_LO))"
read i0 c0 l0 m0 <<<"$(pstat taskset -c 0 "$LB" -m "$LMODEL" -p "$PP_LO" -n "$PREF_TG" -r 1 -o json)"
read i1 c1 l1 m1 <<<"$(pstat taskset -c 0 "$LB" -m "$LMODEL" -p "$PP_HI" -n "$PREF_TG" -r 1 -o json)"
emit "llama prefill" "$i0" "$c0" "$l0" "$m0" "$i1" "$c1" "$l1" "$m1" "$((PP_HI-PP_LO))"
