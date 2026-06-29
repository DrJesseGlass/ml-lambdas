#!/usr/bin/env bash
# Thorough per-COMPONENT instruction breakdown for ONE engine at one (pp,tg).
#
#   perf stat  -> total user instructions for the run
#   perf record -e instructions:u -> instruction-weighted self% per symbol
#   classify symbols -> components -> absolute instr = total * self%
#
# Reusable: re-run for candle after each tweak to watch a component shrink. llama is
# measured ONCE and frozen as the reference (its kernels won't change). Single (pp,tg),
# 1 thread, pinned. NB: includes one model-load (labeled "load") - keep pp*tg modest so
# load is a known, separable slice rather than dominating.
#
# Usage: bash instr_breakdown.sh <candle|llama> [pp] [tg]   (writes /tmp/ib_<engine>.txt)
set -euo pipefail
ENGINE=${1:?candle|llama}; PP=${2:-128}; TG=${3:-64}
M=$HOME/ml-lambdas/qwen-lambda/models
CB=$HOME/candle/target/release/examples/quantized-qwen3-bench
LB=$HOME/llama.cpp/build/bin/llama-bench
DATA=/tmp/ib_$ENGINE.data
OUT=/tmp/ib_$ENGINE.txt

if [ "$ENGINE" = candle ]; then
  CMD=(env CANDLE_QMATMUL_DECODE_THREADS=1 CANDLE_QMATMUL_PREFILL_THREADS=1 CANDLE_KV_PREALLOC=512 \
       taskset -c 0 "$CB" --model "$M/Qwen3-0.6B-Q6packed.gguf" --pp "$PP" --tg "$TG" --reps 1 --warmup 0 --json)
else
  # --no-warmup so perf counts EXACTLY one prefill + one decode pass (the candle bench
  # uses --warmup 0 --reps 1); without it llama's warmup doubles the prefill counts.
  CMD=(taskset -c 0 "$LB" -m "$M/Qwen3-0.6B-Q4out.gguf" -p "$PP" -n "$TG" -r 1 --no-warmup -o json)
fi

perf stat -x, -e instructions:u,cycles:u "${CMD[@]}" >/dev/null 2>/tmp/ib_stat.csv || true
TOT=$(awk -F, '/instructions:u/{print $1}' /tmp/ib_stat.csv)
CYC=$(awk -F, '/cycles:u/{print $1}' /tmp/ib_stat.csv)
perf record -o "$DATA" -e instructions:u -- "${CMD[@]}" >/dev/null 2>/dev/null
perf report -i "$DATA" --stdio -g none --no-children 2>/dev/null | grep -E '\[[.k]\]' > /tmp/ib_rep.txt

awk -v tot="$TOT" -v cyc="$CYC" -v eng="$ENGINE" -v pp="$PP" -v tg="$TG" '
function comp(s){
  # load-time repack/convert FIRST (so a repack symbol is never read as matmul)
  if (s ~ /repack_q|_to_q4_K_8|_to_q6_K_8|fp16_to_fp32_row|gguf|mmap|from_gguf|vocab|::load|_IO_fread/) return "load";
  # matmul: candle packed kernels + ggml 8x4 gemm/gemv (q4_K/q6_K) + generic mul_mat
  if (s ~ /matmul_q4kx8|matmul_q6kx8|gemm_q4kx|gemm_q6kx|gemm_q4_K|gemm_q6_K|gemv_q4_K|gemv_q6_K|_8x4_q8|vec_dot|mul_mat/) return "matmul";
  if (s ~ /quantize_mat_q8|quantize_row_q8|from_float/) return "act_quant";
  if (s ~ /flash|causal|soft_max|softmax/) return "attention";
  if (s ~ /rope/) return "rope";
  if (s ~ /rms_norm|_norm/) return "norm";
  if (s ~ /silu|swiglu|unary_map|binary_map|vec_silu|vec_mul|vec_add|vec_scale|fp32_to_fp16/) return "elementwise";
  if (s ~ /memcpy|memset|malloc|_int_free|_int_malloc|cfree/) return "memory";
  if (s ~ /kv_cache|write_kv/) return "kv_cache";
  if (s ~ /expf|cosf|sinf|cos32|sin32/) return "transcendental";
  if (s ~ /sample|argmax|logits|top_k|top_p/) return "sampling";
  return "other";
}
{
  pct=$1; sub(/%$/,"",pct);
  i=index($0,"] "); sym=substr($0,i+2);
  c=comp(sym); agg[c]+=pct;
}
END{
  printf "# engine=%s pp=%s tg=%s  total_instr=%.0f  total_cycles=%.0f  IPC=%.2f\n",
    eng, pp, tg, tot, cyc, (cyc>0?tot/cyc:0);
  printf "%-15s %16s\n", "component", "instructions";
  n=0; for (c in agg){ keys[n++]=c }
  for (a=0;a<n;a++) for (b=a+1;b<n;b++) if (agg[keys[b]]>agg[keys[a]]){t=keys[a];keys[a]=keys[b];keys[b]=t}
  for (a=0;a<n;a++){ c=keys[a]; printf "%-15s %16.0f\n", c, tot*agg[c]/100 }
}' /tmp/ib_rep.txt | tee "$OUT"
