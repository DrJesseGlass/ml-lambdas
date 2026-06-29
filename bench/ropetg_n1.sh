#!/usr/bin/env bash
# N1 FUSED-ROPE SHORT-TG SWEEP (runs ON the Graviton2 box). Fused RoPE removes a
# FIXED per-token decode overhead (narrow/to_dtype/contiguous/apply_op3). Its % gain
# should be LARGEST at short generation: per-token time is smallest there (small KV),
# so the fixed saving is a bigger fraction. At tg=256 we measured it at the noise
# floor; this checks tg=8..256 to find where (if anywhere) it actually pays off.
#
# Isolates CANDLE_ROPE_FUSED on (default) vs off, everything else at default, in the
# full stack. Pinned to b7fc384b (still has the toggle). Small pp so context starts
# short. Extra reps/warmup because short-tg runs are noisy.
set -uo pipefail
CD=${CD:-$HOME/candle}
MD=${MD:-$HOME/ml-lambdas/qwen-lambda/models}
Q4KM=$MD/Qwen3-0.6B-Q4_K_M.gguf
Q6MC=$MD/Qwen3-0.6B-Q6packed-mc.gguf
PIN=${PIN:-b7fc384b}
PP=${PP:-32}; REPS=${REPS:-5}; WARMUP=${WARMUP:-2}
T_LIST=${T_LIST:-"1 6"}; TGS=${TGS:-"8 16 32 64 128 256"}
BIN=$CD/target/release/examples/quantized-qwen3-bench
PACKER=$CD/target/release/examples/gguf-requant
jget() { sed -nE "s/.*\"$2\":([0-9.]+).*/\1/p" <<<"$1"; }
declare -A ON OFF

echo "=== FUSED-ROPE SHORT-TG  host=$(hostname) $(nproc)vCPU  pp=$PP reps=$REPS warmup=$WARMUP  T=$T_LIST  tg=$TGS  pin=$PIN ==="
[ -f "$Q4KM" ] || { echo "NO Q4_K_M at $Q4KM - abort"; exit 1; }
echo "## building pinned model-core $PIN (bench + gguf-requant, native + f16-attn-dot) ..."
( cd "$CD" && git checkout -f -q "$PIN" && RUSTFLAGS="-C target-cpu=native" \
    cargo build --release --example quantized-qwen3-bench --example gguf-requant \
    --features candle-nn/f16-attn-dot ) || { echo "  BUILD FAILED"; exit 1; }
echo "  built $(cd "$CD" && git rev-parse --short HEAD)"
echo "## baking $Q6MC (q4 lm_head + Q6Kx8 residuals) ..."
rm -f "$Q6MC"; "$PACKER" --input "$Q4KM" --output "$Q6MC" --pack --tensors token_embd,output --dtype q4k | tail -1

dec() { # env t tg -> decode tok/s
  local env=$1 t=$2 tg=$3 out
  out=$(env $env RAYON_NUM_THREADS=$t CANDLE_NUM_THREADS=$t CANDLE_KV_PREALLOC=512 \
      taskset -c "0-$((t-1))" "$BIN" --model "$Q6MC" --pp "$PP" --tg "$tg" \
      --reps "$REPS" --warmup "$WARMUP" --json)
  jget "$out" tg_tok_s_median
}

for t in $T_LIST; do
  for tg in $TGS; do
    ON[$t,$tg]=$(dec "" "$t" "$tg")
    OFF[$t,$tg]=$(dec "CANDLE_ROPE_FUSED=0" "$t" "$tg")
    printf '   t=%s tg=%-4s  rope_on=%-9s rope_off=%-9s\n' "$t" "$tg" "${ON[$t,$tg]}" "${OFF[$t,$tg]}"
  done
done

g() { awk -v a="${1:-0}" -v b="${2:-0}" 'BEGIN{printf (b>0)?"%+.1f%%":"-",(a/b-1)*100}'; }
echo
echo "=== fused RoPE decode gain (rope_on vs rope_off) - expect larger % at short tg ==="
printf '%-6s %-6s %-11s %-11s %-8s\n' thr tg rope_on rope_off gain
printf '%s\n' "-------------------------------------------------"
for t in $T_LIST; do
  for tg in $TGS; do
    printf '%-6s %-6s %-11s %-11s %-8s\n' "$t" "$tg" \
      "${ON[$t,$tg]:-?}" "${OFF[$t,$tg]:-?}" "$(g "${ON[$t,$tg]}" "${OFF[$t,$tg]}")"
  done
  echo
done
