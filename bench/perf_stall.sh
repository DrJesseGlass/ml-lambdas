#!/usr/bin/env bash
# Localize candle's IPC gap on N1: is decode memory-bound or dependency-bound?
#   high backend-stall% + high L2 refill  => memory/bandwidth bound (instr cuts won't help speed)
#   high backend-stall% + low  L2 refill  => compute dependency-chain latency (break the chains)
#   high frontend-stall%                   => i-cache/branch (unlikely for these tight kernels)
# User-space counters (:u). 1 thread, pinned. Runs decode-heavy and prefill-heavy.
set -euo pipefail
M=${M:-$HOME/ml-lambdas/qwen-lambda/models}
B=${B:-$HOME/candle/target/release/examples/quantized-qwen3-bench}
G=${G:-$M/Qwen3-0.6B-Q6packed.gguf}
CE=(CANDLE_QMATMUL_DECODE_THREADS=1 CANDLE_QMATMUL_PREFILL_THREADS=1 CANDLE_KV_PREALLOC=512 CANDLE_VEC_SOFTMAX_EXP=1)
EV=cycles:u,instructions:u,stalled-cycles-backend:u,stalled-cycles-frontend:u,l1d_cache_refill:u,l2d_cache_refill:u

run() { # label pp tg
  echo "## $1 (pp=$2 tg=$3)"
  env "${CE[@]}" perf stat -x, -e "$EV" taskset -c 0 "$B" --model "$G" --pp "$2" --tg "$3" \
    --reps 5 --warmup 0 --json >/dev/null 2>/tmp/ps_$1.csv || true
  awk -F, '
    /[^_]cycles:u/{c=$1} /instructions:u/{i=$1}
    /stalled-cycles-backend/{sb=$1} /stalled-cycles-frontend/{sf=$1}
    /l1d_cache_refill/{l1=$1} /l2d_cache_refill/{l2=$1}
    END{
      printf "  cycles=%.3g instr=%.3g IPC=%.2f\n", c, i, i/c;
      printf "  backend-stall=%.1f%%  frontend-stall=%.1f%%\n", 100*sb/c, 100*sf/c;
      printf "  L1d-refill=%.3g  L2d-refill=%.3g  (L2/L1=%.1f%%)\n", l1, l2, (l1>0?100*l2/l1:0);
    }' /tmp/ps_$1.csv
}

run decode 8 256
run prefill 512 1
