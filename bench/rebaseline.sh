#!/usr/bin/env bash
# CORRECTED RE-BASELINE (runs ON the N1 box). The first version benched explore on
# Q4_K_M + runtime Q4-pack, which leaves the Q6_K residuals UNPACKED (Q6Kx8 is
# baked-only) -> crippled decode. This benches each engine at its REAL best, weight-
# matched (all q4 lm_head + 28 q6_K):
#   explore-Q6packed = explore/rayon-trim on Q6packed.gguf (Q4Kx8+Q6Kx8 baked,
#                      barrier-pool, f16-KV, par-elemwise) -> our true deploy decode.
#   newmain-Q4out    = vanilla 9bcfd982 on Q4out.gguf (same weights, unpacked;
#                      its #3643 kernel + #3634 attention, default).
#   llama-Q4out      = reference on Q4out.gguf.
# Settles: does our decode actually beat new main + how close to llama (the prior
# "0.9x llama, scales with cores"). THREADS shows the scaling.
set -uo pipefail
CD=${CD:-$HOME/candle}
MD=${MD:-$HOME/ml-lambdas/qwen-lambda/models}
Q6PACKED=$MD/Qwen3-0.6B-Q6packed.gguf
Q4OUT=$MD/Qwen3-0.6B-Q4out.gguf
LB=${LB:-$HOME/llama.cpp/build/bin/llama-bench}
PP=${PP:-512}; TG=${TG:-256}; REPS=${REPS:-3}; THREADS=${THREADS:-"2 4 6"}
BIN=$CD/target/release/examples/quantized-qwen3-bench

jget() { sed -nE "s/.*\"$2\":([0-9.]+).*/\1/p" <<<"$1"; }
declare -A PPV TGV

cand() {  # ref label model features "env"
  local ref=$1 label=$2 model=$3 feat=$4 env=$5
  echo "## building $label ($ref, $(basename "$model")) ..."
  if ! ( cd "$CD" && git checkout -q "$ref" && RUSTFLAGS="-C target-cpu=native" \
         cargo build --release --example quantized-qwen3-bench $feat >/dev/null 2>&1 ); then
    echo "   $label BUILD FAILED"; return
  fi
  for t in $THREADS; do
    local out
    out=$(env $env RAYON_NUM_THREADS=$t CANDLE_NUM_THREADS=$t \
        CANDLE_QMATMUL_PREFILL_THREADS=$t CANDLE_QMATMUL_DECODE_THREADS=$t \
        CANDLE_KV_PREALLOC=512 \
        taskset -c "0-$((t-1))" "$BIN" --model "$model" --pp "$PP" --tg "$TG" \
        --reps "$REPS" --warmup 1 --json)
    PPV[$label,$t]=$(jget "$out" pp_tok_s_median)
    TGV[$label,$t]=$(jget "$out" tg_tok_s_median)
    printf '   %-18s t=%s  pp=%-8s tg=%-8s\n' "$label" "$t" "${PPV[$label,$t]}" "${TGV[$label,$t]}"
  done
}

med='def med: sort | if length==0 then "nan" elif length%2==1 then .[length/2|floor] else (.[length/2-1]+.[length/2])/2 end;'
llama() {  # model
  echo "## llama.cpp ($(basename "$1")) ..."
  for t in $THREADS; do
    local out
    out=$("$LB" -m "$1" -t "$t" -p "$PP" -n "$TG" -r "$REPS" -o json 2>/dev/null)
    PPV[llama,$t]=$(jq -r "$med"'[.[]|select(.n_gen==0)][0].samples_ts | med' <<<"$out")
    TGV[llama,$t]=$(jq -r "$med"'[.[]|select(.n_prompt==0)][0].samples_ts | med' <<<"$out")
    printf '   %-18s t=%s  pp=%-8.2f tg=%-8.2f\n' llama "$t" "${PPV[llama,$t]}" "${TGV[llama,$t]}"
  done
}

echo "=== CORRECTED RE-BASELINE  host=$(hostname)  weight-matched (q4 lm_head)  pp=$PP tg=$TG reps=$REPS ==="
cand explore/rayon-trim-q6k-packing  explore-Q6packed  "$Q6PACKED" "--features f16-attn-dot" "CANDLE_PAR_ELEMWISE=1"
cand rebaseline/newmain              newmain-Q4out     "$Q4OUT"    ""                        ""
llama "$Q4OUT"

echo
printf '%-18s %-18s %-18s\n' config "pp_t/s (per thr)" "tg_t/s (per thr)"
printf '%s\n' "------------------------------------------------------------------"
for c in explore-Q6packed newmain-Q4out llama; do
  pp=""; tg=""
  for t in $THREADS; do pp+="${PPV[$c,$t]:-?}/"; tg+="${TGV[$c,$t]:-?}/"; done
  printf '%-18s %-18s %-18s\n' "$c" "${pp%/}" "${tg%/}"
done
echo "(thread order: $THREADS)"
echo
echo "=== verdict (highest thread count) ==="
HT=$(echo "$THREADS" | awk '{print $NF}')
r() { awk -v a="${1:-0}" -v b="${2:-0}" 'BEGIN{printf (b>0)?"%.2fx":"-", a/b}'; }
echo "  explore / newmain @${HT}t : pp=$(r "${PPV[explore-Q6packed,$HT]}" "${PPV[newmain-Q4out,$HT]}")  DECODE=$(r "${TGV[explore-Q6packed,$HT]}" "${TGV[newmain-Q4out,$HT]}")  <- does OUR decode beat new main?"
echo "  explore / llama   @${HT}t : pp=$(r "${PPV[explore-Q6packed,$HT]}" "${PPV[llama,$HT]}")  decode=$(r "${TGV[explore-Q6packed,$HT]}" "${TGV[llama,$HT]}")  <- reproduce the ~0.9x?"
echo "  newmain / llama   @${HT}t : pp=$(r "${PPV[newmain-Q4out,$HT]}" "${PPV[llama,$HT]}")  decode=$(r "${TGV[newmain-Q4out,$HT]}" "${TGV[llama,$HT]}")"
