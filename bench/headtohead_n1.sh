#!/usr/bin/env bash
# N1 HEAD-TO-HEAD (runs ON the Graviton2 box). explore/rayon-trim-q6k-packing vs
# lambda-optimized/model-core vs llama.cpp - all weight-matched (q4 lm_head + Q6
# residuals from the SAME Q4_K_M), identical pp/tg/reps, ONE box session so the
# ratios vs llama are directly comparable.
#
# Goal: settle whether explore's flash-attn rework + rayon-trim actually reaches
# ~0.7x prefill / 0.9x decode vs llama, against model-core's measured 0.58x /
# 0.72x - i.e. is the gap real, portable code or just earlier measurement
# conditions (different llama build, tg128 vs tg256).
#
#   explore   = explore/rayon-trim-q6k-packing on its Q6packed (Q4Kx8 baked
#               prefill + Q6Kx8 residuals + q4 lm_head). Flash-attn rework +
#               rayon-trim are baked into the binary.
#   modelcore = lambda-optimized/model-core on Q6packed-mc (Q6Kx8 residuals +
#               runtime Q4 repack/lanerow prefill + f16-KV + fused rope +
#               par-elemwise).
#   llama     = reference on Q4out.
#
# Two candle builds. Each branch: checkout -f -> build bench+packer (f16-attn-dot)
# -> bake its own gguf -> bench. The binary path is shared, so each branch is
# benched FULLY before the next build overwrites it. All CANDLE_* opt toggles are
# forced ON for both (=1 is always the fast direction; a branch that predates a
# toggle just ignores it), so each runs at its best.
set -uo pipefail
CD=${CD:-$HOME/candle}
MD=${MD:-$HOME/ml-lambdas/qwen-lambda/models}
Q4KM=$MD/Qwen3-0.6B-Q4_K_M.gguf
Q4OUT=$MD/Qwen3-0.6B-Q4out.gguf
Q6MC=$MD/Qwen3-0.6B-Q6packed-mc.gguf
Q6EXP=$MD/Qwen3-0.6B-Q6packed-explore.gguf
LB=${LB:-$HOME/llama.cpp/build/bin/llama-bench}
PP=${PP:-512}; TG=${TG:-256}; REPS=${REPS:-3}; THREADS=${THREADS:-"1 2 4 6"}
BIN=$CD/target/release/examples/quantized-qwen3-bench
PACKER=$CD/target/release/examples/gguf-requant
ON="CANDLE_PREFILL_LANEROW=1 CANDLE_F16_KV=1 CANDLE_ROPE_FUSED=1 CANDLE_PAR_ELEMWISE=1"

jget() { sed -nE "s/.*\"$2\":([0-9.]+).*/\1/p" <<<"$1"; }
declare -A PPV TGV

echo "=== HEAD-TO-HEAD  host=$(hostname)  $(nproc) vCPU  pp=$PP tg=$TG reps=$REPS  threads=$THREADS ==="
[ -f "$Q4KM" ] || { echo "NO Q4_K_M at $Q4KM - abort"; exit 1; }
[ -f "$Q4OUT" ] || echo "WARN: no Q4out at $Q4OUT - llama row will be blank"

build_branch() { # branch
  local br=$1
  echo "## building $br (bench + gguf-requant, native + f16-attn-dot) ..."
  if ! ( cd "$CD" && git checkout -f -q "$br" && RUSTFLAGS="-C target-cpu=native" \
         cargo build --release --example quantized-qwen3-bench --example gguf-requant \
         --features candle-nn/f16-attn-dot ); then
    echo "   $br BUILD FAILED"; return 1
  fi
  echo "   branch: $(cd "$CD" && git rev-parse --abbrev-ref HEAD) @ $(cd "$CD" && git rev-parse --short HEAD)"
}

cand() { # label model
  local label=$1 model=$2
  [ -f "$model" ] || { echo "   ($label: no model $model)"; return; }
  for t in $THREADS; do
    local out
    out=$(env $ON RAYON_NUM_THREADS=$t CANDLE_NUM_THREADS=$t CANDLE_KV_PREALLOC=512 \
        taskset -c "0-$((t-1))" "$BIN" --model "$model" --pp "$PP" --tg "$TG" \
        --reps "$REPS" --warmup 1 --json)
    PPV[$label,$t]=$(jget "$out" pp_tok_s_median)
    TGV[$label,$t]=$(jget "$out" tg_tok_s_median)
    printf '   %-12s t=%s  pp=%-8s tg=%-8s\n' "$label" "$t" "${PPV[$label,$t]}" "${TGV[$label,$t]}"
  done
}

# 1. explore (bench it fully before the model-core build overwrites the binary)
if build_branch explore/rayon-trim-q6k-packing; then
  echo "## baking $Q6EXP (explore packer: Q4Kx8 prefill + Q6Kx8 residuals) ..."
  rm -f "$Q6EXP"; "$PACKER" --input "$Q4KM" --output "$Q6EXP" --pack | tail -1
  echo "## explore (flash-attn rework + rayon-trim) ..."
  cand explore "$Q6EXP"
fi

# 2. model-core
if build_branch lambda-optimized/model-core; then
  # Match explore + llama: q4 lm_head. --pack alone now leaves lm_head at Q6
  # (packer default flipped to opt-in), which is a heavier output projection and
  # unfairly slows decode - so requant token_embd,output -> q4k like the deploy artifact.
  echo "## baking $Q6MC (model-core packer: q4 lm_head + Q6Kx8 residuals) ..."
  rm -f "$Q6MC"; "$PACKER" --input "$Q4KM" --output "$Q6MC" --pack \
    --tensors token_embd,output --dtype q4k | tail -1
  echo "## model-core (lanerow + f16-KV + fused rope + par-elemwise) ..."
  cand modelcore "$Q6MC"
fi

# 3. llama
med='def med: sort | if length==0 then "nan" elif length%2==1 then .[length/2|floor] else (.[length/2-1]+.[length/2])/2 end;'
if [ -x "$LB" ] && [ -f "$Q4OUT" ]; then
  echo "## llama.cpp (Q4out) ..."
  for t in $THREADS; do
    out=$("$LB" -m "$Q4OUT" -t "$t" -p "$PP" -n "$TG" -r "$REPS" -o json 2>/dev/null)
    PPV[llama,$t]=$(jq -r "$med"'[.[]|select(.n_gen==0)][0].samples_ts | med' <<<"$out")
    TGV[llama,$t]=$(jq -r "$med"'[.[]|select(.n_prompt==0)][0].samples_ts | med' <<<"$out")
    printf '   %-12s t=%s  pp=%-8.2f tg=%-8.2f\n' llama "$t" "${PPV[llama,$t]}" "${TGV[llama,$t]}"
  done
else
  echo "## (skipping llama - no llama-bench or Q4out)"
fi

# report
echo
printf '%-12s %-26s %-26s\n' config "pp_t/s (per thr)" "tg_t/s (per thr)"
printf '%s\n' "--------------------------------------------------------------------"
for c in explore modelcore llama; do
  pp=""; tg=""
  for t in $THREADS; do pp+="${PPV[$c,$t]:-?}/"; tg+="${TGV[$c,$t]:-?}/"; done
  printf '%-12s %-26s %-26s\n' "$c" "${pp%/}" "${tg%/}"
done
echo "(thread order: $THREADS)"
echo
HT=$(echo "$THREADS" | awk '{print $NF}')
r() { awk -v a="${1:-0}" -v b="${2:-0}" 'BEGIN{printf (b>0)?"%.2fx":"-", a/b}'; }
echo "=== ratios @${HT}t (the deploy tier) ==="
echo "  explore   / llama : pp=$(r "${PPV[explore,$HT]}" "${PPV[llama,$HT]}")  decode=$(r "${TGV[explore,$HT]}" "${TGV[llama,$HT]}")"
echo "  modelcore / llama : pp=$(r "${PPV[modelcore,$HT]}" "${PPV[llama,$HT]}")  decode=$(r "${TGV[modelcore,$HT]}" "${TGV[llama,$HT]}")"
echo "  explore / modelcore: pp=$(r "${PPV[explore,$HT]}" "${PPV[modelcore,$HT]}")  decode=$(r "${TGV[explore,$HT]}" "${TGV[modelcore,$HT]}")  <- the delta to port"
