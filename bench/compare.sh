#!/usr/bin/env bash
# Head-to-head candle vs llama.cpp on the same box, same model, matched cores.
#
# For each core count: pin both engines to the same physical cores with taskset
# (simulating a Lambda vCPU tier) and measure prefill (pp) + decode (tg) tok/s
# plus peak RSS. Reports a side-by-side table with the candle/llama ratio.
#
# Linux only (needs taskset). Run on an EC2 Graviton2 box for Lambda-faithful
# silicon. Requires: jq, taskset, a built llama.cpp (`llama-bench`), and the
# candle checkout.
#
# Usage:
#   MODEL=~/ml-lambdas/qwen-lambda/models/Qwen3-0.6B-Q4_K_M.gguf \
#   CANDLE_DIR=~/candle LLAMA_BENCH=~/llama.cpp/build/bin/llama-bench \
#   CORES="2 4" ./bench/compare.sh
set -euo pipefail

MODEL="${MODEL:?set MODEL to the GGUF path}"
# candle and llama can run DIFFERENT GGUFs (same logical weights): candle loads our
# offline pre-packed Q4Kx8/Q6Kx8 artifact (llama can't read it), llama loads the
# standard GGUF and repacks to 8x4 at load. Both default to MODEL for back-compat.
CANDLE_MODEL="${CANDLE_MODEL:-$MODEL}"
LLAMA_MODEL="${LLAMA_MODEL:-$MODEL}"
CANDLE_DIR="${CANDLE_DIR:?set CANDLE_DIR to the candle checkout}"
LLAMA_BENCH="${LLAMA_BENCH:-llama-bench}"
CORES="${CORES:-2 4}"
PP="${PP:-512}"
TG="${TG:-128}"
REPS="${REPS:-5}"
# Build flags for the candle bench. Graviton2 == native here; override if needed.
RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
# Cargo features for the "best candle" deploy config. f16-attn-dot matches the
# deploy (native fmla.8h attention dot, ~4.5% N1 decode) - the decomposed candle-nn
# defaults it OFF, so we must opt in to measure best candle. Set CANDLE_FEATURES=""
# to build the bit-exact (feature-off) variant.
CANDLE_FEATURES="${CANDLE_FEATURES:-f16-attn-dot}"
# Extra env passed to the candle bench (the deploy sets CANDLE_KV_PREALLOC=512).
CANDLE_KV_PREALLOC="${CANDLE_KV_PREALLOC:-512}"
# Provenance: where to append a fully-labelled CSV of every row (cpu/threads/pp/tg/
# models/build all recorded so a ratio is never ambiguous later). Empty = no CSV.
CSV="${CSV:-bench/compare_results.csv}"

CANDLE_BIN="$CANDLE_DIR/target/release/examples/quantized-qwen3-bench"
FEAT_ARGS=""
[ -n "$CANDLE_FEATURES" ] && FEAT_ARGS="--features $CANDLE_FEATURES"

echo "Building candle bench (RUSTFLAGS=$RUSTFLAGS, features='$CANDLE_FEATURES')..."
( cd "$CANDLE_DIR" && RUSTFLAGS="$RUSTFLAGS" \
    cargo build --release --example quantized-qwen3-bench $FEAT_ARGS >/dev/null )

# ---- provenance: capture everything needed to interpret a ratio later ----
CPU_TYPE="$(lscpu 2>/dev/null | awk -F: '/Model name/{gsub(/^ +/,"",$2);print $2;exit}')"
[ -z "${CPU_TYPE:-}" ] && CPU_TYPE="$(uname -m)"
HOST="$(hostname)"
CANDLE_GIT="$(cd "$CANDLE_DIR" && git rev-parse --short HEAD 2>/dev/null || echo '?')"
CANDLE_BRANCH="$(cd "$CANDLE_DIR" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
LLAMA_GIT="$(cd "$(dirname "$LLAMA_BENCH")/../.." 2>/dev/null && git rev-parse --short HEAD 2>/dev/null || echo '?')"
CANDLE_MODEL_ID="$(basename "$CANDLE_MODEL")"
LLAMA_MODEL_ID="$(basename "$LLAMA_MODEL")"
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

printf '\n==== provenance ====\n'
printf '  host          : %s\n' "$HOST"
printf '  cpu           : %s\n' "$CPU_TYPE"
printf '  pp / tg / reps: %s / %s / %s\n' "$PP" "$TG" "$REPS"
printf '  candle        : %s @ %s (%s)  features=[%s]  kv_prealloc=%s\n' \
  "$CANDLE_MODEL_ID" "$CANDLE_BRANCH" "$CANDLE_GIT" "$CANDLE_FEATURES" "$CANDLE_KV_PREALLOC"
printf '  llama.cpp     : %s @ (%s)\n' "$LLAMA_MODEL_ID" "$LLAMA_GIT"
printf '  rustflags     : %s\n' "$RUSTFLAGS"

if [ -n "$CSV" ]; then
  mkdir -p "$(dirname "$CSV")"
  if [ ! -f "$CSV" ]; then
    echo "stamp,host,cpu,threads,pp,tg,reps,engine,candle_model,llama_model,candle_branch,candle_git,candle_features,llama_git,pp_tok_s,tg_tok_s,peak_mb,pp_ratio_c_over_l,tg_ratio_c_over_l" >"$CSV"
  fi
fi

# Run "$@" with stdout to $1, polling peak RSS (KB) of the process, echoed last.
run_with_rss() {
  local outfile="$1"; shift
  "$@" >"$outfile" 2>/dev/null &
  local pid=$! peak=0 rss
  while kill -0 "$pid" 2>/dev/null; do
    rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    [ -n "${rss:-}" ] && [ "$rss" -gt "$peak" ] && peak=$rss
    sleep 0.2
  done
  wait "$pid"
  echo "$peak"
}

tc=$(mktemp); tl=$(mktemp)
trap 'rm -f "$tc" "$tl"' EXIT

printf '\n%-6s %-10s %-12s %-12s %-10s\n' cores engine pp_t/s tg_t/s peak_MB
printf '%s\n' "--------------------------------------------------------"

for c in $CORES; do
  cpus="0-$((c - 1))"

  c_rss=$(run_with_rss "$tc" \
    env RAYON_NUM_THREADS="$c" CANDLE_NUM_THREADS="$c" \
    CANDLE_QMATMUL_DECODE_THREADS="$c" CANDLE_QMATMUL_PREFILL_THREADS="$c" \
    CANDLE_KV_PREALLOC="$CANDLE_KV_PREALLOC" ${CANDLE_EXTRA:-} \
    taskset -c "$cpus" "$CANDLE_BIN" \
    --model "$CANDLE_MODEL" --pp "$PP" --tg "$TG" --reps "$REPS" --json)
  c_pp=$(jq -r '.pp_tok_s_median' "$tc")
  c_tg=$(jq -r '.tg_tok_s_median' "$tc")

  l_rss=$(run_with_rss "$tl" \
    taskset -c "$cpus" "$LLAMA_BENCH" \
    -m "$LLAMA_MODEL" -t "$c" -p "$PP" -n "$TG" -r "$REPS" -o json)
  # llama-bench's avg_ts is the *mean* over reps; the candle side reports the
  # median (.*_median) and the README calls this a median run. Take the median of
  # llama-bench's per-rep samples_ts so both sides use the same statistic and one
  # noisy rep can't skew only the denominator of the ratio.
  l_med='def med: sort | if length==0 then "nan" elif length%2==1 then .[length/2|floor] else (.[length/2-1]+.[length/2])/2 end;'
  l_pp=$(jq -r "$l_med"'[.[]|select(.n_gen==0)][0].samples_ts | med' "$tl")
  l_tg=$(jq -r "$l_med"'[.[]|select(.n_prompt==0)][0].samples_ts | med' "$tl")

  c_mb=$(echo "$c_rss/1024" | bc -l); l_mb=$(echo "$l_rss/1024" | bc -l)
  pp_ratio=$(awk -v a="$c_pp" -v b="$l_pp" 'BEGIN{printf "%.4f", a/b}')
  tg_ratio=$(awk -v a="$c_tg" -v b="$l_tg" 'BEGIN{printf "%.4f", a/b}')

  printf '%-6s %-10s %-12.2f %-12.2f %-10.1f\n' "$c" candle    "$c_pp" "$c_tg" "$c_mb"
  printf '%-6s %-10s %-12.2f %-12.2f %-10.1f\n' "$c" llama.cpp "$l_pp" "$l_tg" "$l_mb"
  printf '%-6s %-10s %-12s %-12s\n' "$c" "ratio c/l" "$(printf '%.2fx' "$pp_ratio")" "$(printf '%.2fx' "$tg_ratio")"
  echo

  if [ -n "$CSV" ]; then
    base="$STAMP,$HOST,\"$CPU_TYPE\",$c,$PP,$TG,$REPS"
    meta="$CANDLE_MODEL_ID,$LLAMA_MODEL_ID,$CANDLE_BRANCH,$CANDLE_GIT,$CANDLE_FEATURES,$LLAMA_GIT"
    printf '%s,candle,%s,%.2f,%.2f,%.1f,%.4f,%.4f\n' "$base" "$meta" "$c_pp" "$c_tg" "$c_mb" "$pp_ratio" "$tg_ratio" >>"$CSV"
    printf '%s,llama.cpp,%s,%.2f,%.2f,%.1f,,\n' "$base" "$meta" "$l_pp" "$l_tg" "$l_mb" >>"$CSV"
  fi
done

[ -n "$CSV" ] && echo "appended rows to $CSV"
