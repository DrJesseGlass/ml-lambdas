#!/usr/bin/env bash
# N1 TG-LENGTH SWEEP (runs ON the Graviton2 box). Question: does candle's
# decode/llama ratio RISE toward llama as generation length grows, or fall? This
# settles where the historical "~0.9x decode" came from - if the ratio climbs with
# tg, the 90% lived at LONG generation, not short.
#
# Fixed pp=512, threads=6 (the deploy tier). Sweep tg = 128 256 512 1024. explore
# (the historical artifact where 0.9x was seen) + model-core, both on a q4 lm_head,
# vs llama - ONE session / ONE llama build so the tg curve is apples-to-apples.
# Mean decode position = 512 + tg/2, so the sweep spans ~576..1024 in KV length.
set -uo pipefail
CD=${CD:-$HOME/candle}
MD=${MD:-$HOME/ml-lambdas/qwen-lambda/models}
Q4KM=$MD/Qwen3-0.6B-Q4_K_M.gguf
Q4OUT=$MD/Qwen3-0.6B-Q4out.gguf
Q6MC=$MD/Qwen3-0.6B-Q6packed-mc.gguf
Q6EXP=$MD/Qwen3-0.6B-Q6packed-explore.gguf
LB=${LB:-$HOME/llama.cpp/build/bin/llama-bench}
PP=${PP:-512}; REPS=${REPS:-3}; T=${T:-6}; TGS=${TGS:-"128 256 512 1024"}
BIN=$CD/target/release/examples/quantized-qwen3-bench
PACKER=$CD/target/release/examples/gguf-requant
ON="CANDLE_PREFILL_LANEROW=1 CANDLE_F16_KV=1 CANDLE_ROPE_FUSED=1 CANDLE_PAR_ELEMWISE=1"

jget() { sed -nE "s/.*\"$2\":([0-9.]+).*/\1/p" <<<"$1"; }
declare -A TGV   # TGV[label,tg] = decode tok/s

echo "=== TG SWEEP  host=$(hostname)  $(nproc) vCPU  pp=$PP threads=$T reps=$REPS  tg=$TGS ==="
[ -f "$Q4KM" ] || { echo "NO Q4_K_M at $Q4KM - abort"; exit 1; }

build_branch() { # branch
  local br=$1
  echo "## building $br (bench + gguf-requant, native + f16-attn-dot) ..."
  ( cd "$CD" && git checkout -f -q "$br" && RUSTFLAGS="-C target-cpu=native" \
      cargo build --release --example quantized-qwen3-bench --example gguf-requant \
      --features candle-nn/f16-attn-dot ) || { echo "   $br BUILD FAILED"; return 1; }
  echo "   branch: $(cd "$CD" && git rev-parse --short HEAD)"
}

sweep_cand() { # label model
  local label=$1 model=$2
  [ -f "$model" ] || { echo "   ($label: no model $model)"; return; }
  for tg in $TGS; do
    local out
    out=$(env $ON RAYON_NUM_THREADS=$T CANDLE_NUM_THREADS=$T CANDLE_KV_PREALLOC=2048 \
        taskset -c "0-$((T-1))" "$BIN" --model "$model" --pp "$PP" --tg "$tg" \
        --reps "$REPS" --warmup 1 --json)
    TGV[$label,$tg]=$(jget "$out" tg_tok_s_median)
    printf '   %-10s tg=%-5s  decode=%s tok/s\n' "$label" "$tg" "${TGV[$label,$tg]}"
  done
}

if build_branch explore/rayon-trim-q6k-packing; then
  echo "## baking $Q6EXP (explore packer) ..."; rm -f "$Q6EXP"
  "$PACKER" --input "$Q4KM" --output "$Q6EXP" --pack | tail -1
  echo "## explore decode vs tg ..."; sweep_cand explore "$Q6EXP"
fi

if build_branch lambda-optimized/model-core; then
  echo "## baking $Q6MC (model-core, q4 lm_head) ..."; rm -f "$Q6MC"
  "$PACKER" --input "$Q4KM" --output "$Q6MC" --pack --tensors token_embd,output --dtype q4k | tail -1
  echo "## model-core decode vs tg ..."; sweep_cand modelcore "$Q6MC"
fi

med='def med: sort | if length==0 then "nan" elif length%2==1 then .[length/2|floor] else (.[length/2-1]+.[length/2])/2 end;'
if [ -x "$LB" ] && [ -f "$Q4OUT" ]; then
  echo "## llama decode vs tg ..."
  for tg in $TGS; do
    out=$("$LB" -m "$Q4OUT" -t "$T" -p "$PP" -n "$tg" -r "$REPS" -o json 2>/dev/null)
    TGV[llama,$tg]=$(jq -r "$med"'[.[]|select(.n_prompt==0)][0].samples_ts | med' <<<"$out")
    printf '   %-10s tg=%-5s  decode=%s tok/s\n' llama "$tg" "${TGV[llama,$tg]}"
  done
fi

echo
printf '%-8s' tg; for c in explore modelcore llama; do printf ' %-12s' "$c"; done
printf ' | %-10s %-10s\n' exp/llama mc/llama
printf '%s\n' "----------------------------------------------------------------------"
r() { awk -v a="${1:-0}" -v b="${2:-0}" 'BEGIN{printf (b>0)?"%.3f":"-", a/b}'; }
for tg in $TGS; do
  printf '%-8s' "$tg"
  for c in explore modelcore llama; do printf ' %-12s' "${TGV[$c,$tg]:-?}"; done
  printf ' | %-10s %-10s\n' "$(r "${TGV[explore,$tg]}" "${TGV[llama,$tg]}")" "$(r "${TGV[modelcore,$tg]}" "${TGV[llama,$tg]}")"
done
echo
echo "=== if the ratio column RISES down the rows, candle approaches llama as tg grows (90% lives at long tg) ==="
