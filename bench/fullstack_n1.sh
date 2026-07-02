#!/usr/bin/env bash
# N1 FULL-STACK BENCH (runs ON the Graviton2 box). The complete model-core deploy
# stack vs a new-main-equivalent baseline vs llama, weight-matched (q4 lm_head +
# Q6 residuals).
#
#   model-core = lambda-optimized/model-core on Q6packed-mc.gguf - Q6Kx8 residuals
#                baked + Q4 runtime-repack/lanerow prefill + f16-KV decode + fused
#                rope + par-elemwise (every optimization ON, the deploy artifact).
#   newmain    = the SAME binary with every opt toggle OFF, on Q4out.gguf. The
#                toggles revert each path to the upstream behavior, so this is a
#                same-binary proxy for vanilla new-main (no second checkout/build,
#                and avoids the bench example not existing on origin/main):
#                  CANDLE_PREFILL_LANEROW=0 -> #3643 Q4 ;  CANDLE_F16_KV=0 -> f32 KV
#                  CANDLE_ROPE_FUSED=0 -> op rope ;  CANDLE_PAR_ELEMWISE=0 -> serial
#                  Q4out.gguf (no Q6Kx8) -> Q6_K residuals run scalar.
#   llama      = reference on Q4out.gguf.
#
# Q6packed-mc is baked here from Q4_K_M (the box's old Q6packed.gguf is the
# incompatible explore format). Sweeps THREADS for the multi-vCPU deploy tiers.
set -uo pipefail
CD=${CD:-$HOME/candle}
MD=${MD:-$HOME/ml-lambdas/qwen-lambda/models}
Q4KM=$MD/Qwen3-0.6B-Q4_K_M.gguf
Q4OUT=$MD/Qwen3-0.6B-Q4out.gguf
Q6MC=$MD/Qwen3-0.6B-Q6packed-mc.gguf
LB=${LB:-$HOME/llama.cpp/build/bin/llama-bench}
PP=${PP:-512}; TG=${TG:-256}; REPS=${REPS:-3}; THREADS=${THREADS:-"1 2 4 6"}
BIN=$CD/target/release/examples/quantized-qwen3-bench
OFF="CANDLE_PREFILL_LANEROW=0 CANDLE_F16_KV=0 CANDLE_ROPE_FUSED=0 CANDLE_PAR_ELEMWISE=0"

jget() { sed -nE "s/.*\"$2\":([0-9.]+).*/\1/p" <<<"$1"; }
declare -A PPV TGV

echo "=== FULL-STACK BENCH  host=$(hostname)  $(nproc) vCPU  pp=$PP tg=$TG reps=$REPS  threads=$THREADS ==="
echo "## building model-core (bench + gguf-requant, native + f16-attn-dot) ..."
if ! ( cd "$CD" && git checkout -q model-core-ext && RUSTFLAGS="-C target-cpu=native" \
       cargo build --release --example quantized-qwen3-bench --example gguf-requant \
       --features candle-nn/f16-attn-dot ); then
  echo "   model-core BUILD FAILED"; exit 1
fi
echo "   branch: $(cd "$CD" && git rev-parse --abbrev-ref HEAD) @ $(cd "$CD" && git rev-parse --short HEAD)"
[ -f "$Q4KM" ] || { echo "   NO Q4_K_M at $Q4KM - abort"; exit 1; }
[ -f "$Q4OUT" ] || echo "   WARN: no Q4out at $Q4OUT - newmain/llama rows will be blank"
if [ ! -f "$Q6MC" ]; then
  # q4 lm_head deploy artifact: --pack alone now leaves lm_head at Q6 (packer
  # default is opt-in), a heavier output projection that slows decode. Requant
  # token_embd,output -> q4k to match the Q4out baseline + llama.
  echo "## baking $Q6MC ..."
  "$CD/target/release/examples/gguf-requant" --input "$Q4KM" --output "$Q6MC" --pack \
    --tensors token_embd,output --dtype q4k | tail -1
fi

cand() { # label model env
  local label=$1 model=$2 env=$3
  [ -f "$model" ] || { echo "   ($label: no model $model)"; return; }
  for t in $THREADS; do
    local out
    out=$(env $env RAYON_NUM_THREADS=$t CANDLE_NUM_THREADS=$t CANDLE_KV_PREALLOC=512 \
        taskset -c "0-$((t-1))" "$BIN" --model "$model" --pp "$PP" --tg "$TG" \
        --reps "$REPS" --warmup 1 --json)
    PPV[$label,$t]=$(jget "$out" pp_tok_s_median)
    TGV[$label,$t]=$(jget "$out" tg_tok_s_median)
    local rd bd; rd=$(jget "$out" gguf_read_ms); bd=$(jget "$out" model_build_ms)
    printf '   %-14s t=%s  pp=%-8s tg=%-8s  load: read=%sms build=%sms\n' \
      "$label" "$t" "${PPV[$label,$t]}" "${TGV[$label,$t]}" "$rd" "$bd"
  done
}

# Boot A/B: sequential vs parallel weight load on the deploy artifact, COLD (drop
# page cache before each load so the read cost is real) and with peak RSS. Parallel
# load reads/constructs every tensor across the rayon pool; this is where the cold
# I/O latency (vs warm bandwidth) win shows up, at no RSS or quality cost.
dropcache() { command -v sudo >/dev/null 2>&1 && sudo sh -c 'sync; echo 3 >/proc/sys/vm/drop_caches' 2>/dev/null; }
bootab() { # model t
  local model=$1 t=${2:-6}
  [ -f "$model" ] || { echo "   (bootab: no model $model)"; return; }
  command -v /usr/bin/time >/dev/null 2>&1 || { echo "   (bootab: no /usr/bin/time)"; return; }
  for mode in seq par; do
    local flag=""; [ "$mode" = par ] && flag="--parallel"
    dropcache
    local out rss rd bd
    out=$(env RAYON_NUM_THREADS=$t CANDLE_NUM_THREADS=$t CANDLE_KV_PREALLOC=512 \
        taskset -c "0-$((t-1))" /usr/bin/time -v "$BIN" --model "$model" \
        --pp "$PP" --tg "$TG" --reps 1 --warmup 1 $flag --json 2>/tmp/bootab_$mode.err)
    rss=$(sed -nE 's/.*Maximum resident set size \(kbytes\): ([0-9]+).*/\1/p' /tmp/bootab_$mode.err)
    rd=$(jget "$out" gguf_read_ms); bd=$(jget "$out" model_build_ms)
    printf '   boot[%-3s] t=%s  read=%sms build=%sms  peakRSS=%sMB\n' \
      "$mode" "$t" "$rd" "$bd" "$(( ${rss:-0} / 1024 ))"
  done
}
echo "## BOOT A/B (Q6packed-mc, cold cache, sequential vs parallel load) ..."
bootab "$Q6MC" "$(echo "$THREADS" | awk '{print $NF}')"

echo "## model-core (Q6packed-mc, full deploy stack) ..."
cand model-core "$Q6MC" ""
echo "## newmain-equiv (Q4out, all opt toggles OFF) ..."
cand newmain "$Q4OUT" "$OFF"

med='def med: sort | if length==0 then "nan" elif length%2==1 then .[length/2|floor] else (.[length/2-1]+.[length/2])/2 end;'
if [ -x "$LB" ] && [ -f "$Q4OUT" ]; then
  echo "## llama.cpp (Q4out) ..."
  for t in $THREADS; do
    out=$("$LB" -m "$Q4OUT" -t "$t" -p "$PP" -n "$TG" -r "$REPS" -o json 2>/dev/null)
    PPV[llama,$t]=$(jq -r "$med"'[.[]|select(.n_gen==0)][0].samples_ts | med' <<<"$out")
    TGV[llama,$t]=$(jq -r "$med"'[.[]|select(.n_prompt==0)][0].samples_ts | med' <<<"$out")
    printf '   %-14s t=%s  pp=%-8.2f tg=%-8.2f\n' llama "$t" "${PPV[llama,$t]}" "${TGV[llama,$t]}"
  done
else
  echo "## (skipping llama - no llama-bench or Q4out)"
fi

echo
printf '%-14s %-24s %-24s\n' config "pp_t/s (per thr)" "tg_t/s (per thr)"
printf '%s\n' "----------------------------------------------------------------"
for c in model-core newmain llama; do
  pp=""; tg=""
  for t in $THREADS; do pp+="${PPV[$c,$t]:-?}/"; tg+="${TGV[$c,$t]:-?}/"; done
  printf '%-14s %-24s %-24s\n' "$c" "${pp%/}" "${tg%/}"
done
echo "(thread order: $THREADS)"
echo
HT=$(echo "$THREADS" | awk '{print $NF}')
r() { awk -v a="${1:-0}" -v b="${2:-0}" 'BEGIN{printf (b>0)?"%.2fx":"-", a/b}'; }
echo "=== ratios @${HT}t (the deploy tier) ==="
echo "  model-core / newmain : pp=$(r "${PPV[model-core,$HT]}" "${PPV[newmain,$HT]}")  DECODE=$(r "${TGV[model-core,$HT]}" "${TGV[newmain,$HT]}")  <- what the full stack adds"
echo "  model-core / llama   : pp=$(r "${PPV[model-core,$HT]}" "${PPV[llama,$HT]}")  decode=$(r "${TGV[model-core,$HT]}" "${TGV[llama,$HT]}")  <- gap to reference"
