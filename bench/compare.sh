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
CANDLE_DIR="${CANDLE_DIR:?set CANDLE_DIR to the candle checkout}"
LLAMA_BENCH="${LLAMA_BENCH:-llama-bench}"
CORES="${CORES:-2 4}"
PP="${PP:-512}"
TG="${TG:-128}"
REPS="${REPS:-5}"
# Build flags for the candle bench. Graviton2 == native here; override if needed.
RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"

CANDLE_BIN="$CANDLE_DIR/target/release/examples/quantized-qwen3-bench"

echo "Building candle bench (RUSTFLAGS=$RUSTFLAGS)..."
( cd "$CANDLE_DIR" && RUSTFLAGS="$RUSTFLAGS" \
    cargo build --release --example quantized-qwen3-bench >/dev/null )

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
    env CANDLE_QMATMUL_DECODE_THREADS="$c" CANDLE_QMATMUL_PREFILL_THREADS="$c" \
    taskset -c "$cpus" "$CANDLE_BIN" \
    --model "$MODEL" --pp "$PP" --tg "$TG" --reps "$REPS" --json)
  c_pp=$(jq -r '.pp_tok_s_median' "$tc")
  c_tg=$(jq -r '.tg_tok_s_median' "$tc")

  l_rss=$(run_with_rss "$tl" \
    taskset -c "$cpus" "$LLAMA_BENCH" \
    -m "$MODEL" -t "$c" -p "$PP" -n "$TG" -r "$REPS" -o json)
  l_pp=$(jq -r '[.[]|select(.n_gen==0)][0].avg_ts' "$tl")
  l_tg=$(jq -r '[.[]|select(.n_prompt==0)][0].avg_ts' "$tl")

  printf '%-6s %-10s %-12.2f %-12.2f %-10.1f\n' "$c" candle    "$c_pp" "$c_tg" "$(echo "$c_rss/1024" | bc -l)"
  printf '%-6s %-10s %-12.2f %-12.2f %-10.1f\n' "$c" llama.cpp "$l_pp" "$l_tg" "$(echo "$l_rss/1024" | bc -l)"
  awk -v cpp="$c_pp" -v ctg="$c_tg" -v lpp="$l_pp" -v ltg="$l_tg" -v c="$c" \
    'BEGIN{printf "%-6s %-10s %-12s %-12s\n", c, "ratio c/l", \
      sprintf("%.2fx", cpp/lpp), sprintf("%.2fx", ctg/ltg)}'
  echo
done
