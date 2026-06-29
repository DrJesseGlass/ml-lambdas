#!/usr/bin/env bash
# N1 LEAVE-ONE-OUT A/B (runs ON the Graviton2 box). Per-PR speed profile: the
# deployed model-core stack with EVERY optimization on (baseline) vs the same
# binary with ONE optimization's toggle off. The delta = that PR's marginal
# contribution to the deployed stack, measured identically (one binary, one
# q4-lm_head gguf, same pp/tg) so the per-PR numbers are directly comparable.
#
# Pinned to commit b7fc384b - the pre-toggle-removal model-core, which still has
# every CANDLE_* toggle - so the A/B is possible (the current branch tip removed
# them). f16 KV (#3665) is only measurable wired into the model, hence the
# leave-one-out framing rather than per-branch standalone.
#
#   no_par_elemwise CANDLE_PAR_ELEMWISE=0    -> #3664 par-elemwise
#   no_f16kv        CANDLE_F16_KV=0          -> #3665 flash-kv-decode (f16 cache)
#   no_lanerow      CANDLE_PREFILL_LANEROW=0 -> #3666 q6k-packed (Q4 lane=row prefill)
#   no_parquant     CANDLE_PREFILL_PARQUANT=0-> #3666 q6k-packed (parallel act-quant)
#   no_rope         CANDLE_ROPE_FUSED=0      -> #3667 quantized-qwen3-rope
set -uo pipefail
CD=${CD:-$HOME/candle}
MD=${MD:-$HOME/ml-lambdas/qwen-lambda/models}
Q4KM=$MD/Qwen3-0.6B-Q4_K_M.gguf
Q6MC=$MD/Qwen3-0.6B-Q6packed-mc.gguf
PIN=${PIN:-b7fc384b}
PP=${PP:-512}; TG=${TG:-256}; REPS=${REPS:-3}; THREADS=${THREADS:-"1 6"}
BIN=$CD/target/release/examples/quantized-qwen3-bench
PACKER=$CD/target/release/examples/gguf-requant
jget() { sed -nE "s/.*\"$2\":([0-9.]+).*/\1/p" <<<"$1"; }
declare -A PPV TGV
# no_* = leave-one-out (turn an on-by-default opt OFF). vec_expf = turn-ON the
# default-off NEON poly expf experiment (#3665), measured against the allon baseline.
names=(allon no_par_elemwise no_f16kv no_lanerow no_parquant no_rope vec_expf)
declare -A ENVOF=(
  [allon]=""
  [no_par_elemwise]="CANDLE_PAR_ELEMWISE=0"
  [no_f16kv]="CANDLE_F16_KV=0"
  [no_lanerow]="CANDLE_PREFILL_LANEROW=0"
  [no_parquant]="CANDLE_PREFILL_PARQUANT=0"
  [no_rope]="CANDLE_ROPE_FUSED=0"
  [vec_expf]="CANDLE_VEC_SOFTMAX_EXP=1"
)

echo "=== LEAVE-ONE-OUT  host=$(hostname) $(nproc)vCPU  pp=$PP tg=$TG reps=$REPS  threads=$THREADS  pin=$PIN ==="
[ -f "$Q4KM" ] || { echo "NO Q4_K_M at $Q4KM - abort"; exit 1; }
echo "## building pinned model-core $PIN (bench + gguf-requant, native + f16-attn-dot) ..."
( cd "$CD" && git checkout -f -q "$PIN" && RUSTFLAGS="-C target-cpu=native" \
    cargo build --release --example quantized-qwen3-bench --example gguf-requant \
    --features candle-nn/f16-attn-dot ) || { echo "  BUILD FAILED"; exit 1; }
echo "  built $(cd "$CD" && git rev-parse --short HEAD)"
echo "## baking $Q6MC (q4 lm_head + Q6Kx8 residuals) ..."
rm -f "$Q6MC"; "$PACKER" --input "$Q4KM" --output "$Q6MC" --pack --tensors token_embd,output --dtype q4k | tail -1

run() { # name
  local nm=$1 env=${ENVOF[$1]} t out
  for t in $THREADS; do
    out=$(env $env RAYON_NUM_THREADS=$t CANDLE_NUM_THREADS=$t CANDLE_KV_PREALLOC=512 \
        taskset -c "0-$((t-1))" "$BIN" --model "$Q6MC" --pp "$PP" --tg "$TG" \
        --reps "$REPS" --warmup 1 --json)
    PPV[$nm,$t]=$(jget "$out" pp_tok_s_median)
    TGV[$nm,$t]=$(jget "$out" tg_tok_s_median)
    printf '   %-16s t=%s  pp=%-8s tg=%-8s\n' "$nm" "$t" "${PPV[$nm,$t]}" "${TGV[$nm,$t]}"
  done
}
for nm in "${names[@]}"; do echo "## $nm  (${ENVOF[$nm]:-all on / baseline})"; run "$nm"; done

g() { awk -v a="${1:-0}" -v b="${2:-0}" 'BEGIN{printf (b>0)?"%+.1f%%":"-",(a/b-1)*100}'; }
echo
echo "=== per-PR contribution to the deployed stack (baseline allon vs that toggle OFF) ==="
echo "    positive % = how much the optimization adds at that config"
printf '%-16s %-6s %-26s %-26s\n' "PR toggle off" thr "PREFILL pp tok/s (gain)" "DECODE tg tok/s (gain)"
printf '%s\n' "--------------------------------------------------------------------------------"
for t in $THREADS; do
  for nm in no_par_elemwise no_f16kv no_lanerow no_parquant no_rope; do
    printf '%-16s %-6s %s->%s %-10s %s->%s %-10s\n' "$nm" "$t" \
      "${PPV[$nm,$t]:-?}" "${PPV[allon,$t]:-?}" "($(g "${PPV[allon,$t]}" "${PPV[$nm,$t]}"))" \
      "${TGV[$nm,$t]:-?}" "${TGV[allon,$t]:-?}" "($(g "${TGV[allon,$t]}" "${TGV[$nm,$t]}"))"
  done
  echo
done

echo "=== #3665 vectorized expf, opt-in (CANDLE_VEC_SOFTMAX_EXP=1 vs baseline; positive = faster) ==="
for t in $THREADS; do
  printf '%-16s %-6s %s->%s %-10s %s->%s %-10s\n' "vec_expf" "$t" \
    "${PPV[allon,$t]:-?}" "${PPV[vec_expf,$t]:-?}" "($(g "${PPV[vec_expf,$t]}" "${PPV[allon,$t]}"))" \
    "${TGV[allon,$t]:-?}" "${TGV[vec_expf,$t]:-?}" "($(g "${TGV[vec_expf,$t]}" "${TGV[allon,$t]}"))"
done
