#!/usr/bin/env bash
# i8mm (SMMLA) Q4_K prefill A/B on Graviton3/4 (Neoverse-V1/V2).
#
# SMMLA does a 2x2 int8 outer-product-accumulate per instruction (~4 SDOTs of
# work) - the lever for large-M prefill (vision models: a page of vision tokens
# is a huge prompt). Graviton2/N1 lacks i8mm; Graviton3 (c7g) and Graviton4
# (c8g) have it. This measures the prefill win by toggling the SAME binary:
#   CANDLE_PREFILL_I8MM=1  -> SMMLA prefill GEMM (gemm_q4kx8_q8k_i8mm)
#   CANDLE_PREFILL_I8MM=0  -> the SDOT prefill GEMM (apples-to-apples)
# across prefill lengths PP and core counts CORES, and (optionally) vs llama.cpp.
#
# The i8mm path lives in candle's RUNTIME-repack route (matmul_q4k_packed), so the
# bench loads a PLAIN Q4_K gguf and sets CANDLE_MATMUL_PACKED_Q4K=1. (The baked
# PackedQ4Kx8 artifact is NOT wired to i8mm yet - it stays SDOT.) Decode (m<4) is
# unaffected by the toggle, so tg t/s is a built-in sanity check (should not move).
#
# Linux + i8mm hardware only. The build MUST enable i8mm (target-cpu=native on a
# c7g/c8g implies neoverse-v1/v2 -> i8mm); the script verifies smmla is actually
# in the binary and that the on-box i8mm unit tests pass before benching.
#
# Usage:
#   MODEL=~/ml-lambdas/qwen-lambda/models/Qwen3-0.6B-Q4_K_M.gguf \
#   CANDLE_DIR=~/candle PP="512 2048 4096" CORES="1 4" ./bench/i8mm_prefill.sh
#
# REMEMBER to STOP the c7g/c8g box when done (it is billable).
set -euo pipefail

# A plain Q4_K gguf (NOT the pre-packed Q4Kx8 artifact - the i8mm path repacks at
# runtime from Q4_K). Same default as compare.sh's MODEL.
MODEL="${MODEL:?set MODEL to the plain Q4_K GGUF path}"
CANDLE_DIR="${CANDLE_DIR:?set CANDLE_DIR to the candle checkout}"
PP="${PP:-512 2048 4096}"          # prefill lengths (mimic vision token counts)
TG="${TG:-32}"                     # short decode; only a sanity check here
CORES="${CORES:-1 4}"
REPS="${REPS:-5}"
CANDLE_FEATURES="${CANDLE_FEATURES:-f16-attn-dot}"  # match the deploy
CANDLE_KV_PREALLOC="${CANDLE_KV_PREALLOC:-512}"
# target-cpu=native on Graviton3/4 enables i8mm (and bf16/sve) + the right
# scheduler model. Override to pin, e.g. RUSTFLAGS="-C target-feature=+i8mm".
RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
# Optional head-to-head vs a llama.cpp i8mm build (its prefill also uses SMMLA).
LLAMA_BENCH="${LLAMA_BENCH:-}"
LLAMA_MODEL="${LLAMA_MODEL:-$MODEL}"
CSV="${CSV:-bench/i8mm_prefill_results.csv}"

CANDLE_BIN="$CANDLE_DIR/target/release/examples/quantized-qwen3-bench"
FEAT_ARGS=""
[ -n "$CANDLE_FEATURES" ] && FEAT_ARGS="--features $CANDLE_FEATURES"

# ---- hardware guard: refuse to run (and bill) on a box without i8mm ----
if ! grep -qw i8mm /proc/cpuinfo 2>/dev/null; then
  echo "ERROR: no 'i8mm' in /proc/cpuinfo - this is not Graviton3/4 (Neoverse-V1/V2)." >&2
  echo "       The SMMLA path needs i8mm; run this on a c7g/c8g box. Aborting." >&2
  exit 1
fi

echo "Building candle bench (+i8mm via RUSTFLAGS=$RUSTFLAGS, features='$CANDLE_FEATURES')..."
( cd "$CANDLE_DIR" && RUSTFLAGS="$RUSTFLAGS" \
    cargo build --release --example quantized-qwen3-bench $FEAT_ARGS >/dev/null )

# ---- build guard: confirm smmla actually made it into the binary ----
if command -v objdump >/dev/null 2>&1; then
  if objdump -d "$CANDLE_BIN" 2>/dev/null | grep -qiw smmla; then
    echo "  ok: 'smmla' present in the binary (i8mm compiled in)."
  else
    echo "ERROR: no 'smmla' in the built binary - i8mm was NOT compiled, so" >&2
    echo "       CANDLE_PREFILL_I8MM=1 would silently run the slow scalar twin." >&2
    echo "       Set RUSTFLAGS to enable i8mm (e.g. -C target-feature=+i8mm)." >&2
    exit 1
  fi
else
  echo "  warn: objdump not found - skipping the smmla-in-binary check."
fi

# ---- correctness on real hardware: run the i8mm unit tests under +i8mm ----
echo "Validating the real SMMLA instruction on this box (i8mm unit tests)..."
( cd "$CANDLE_DIR" && RUSTFLAGS="$RUSTFLAGS" \
    cargo test --release -p candle-core i8mm >/dev/null ) \
  && echo "  ok: i8mm kernels match reference on real hardware." \
  || { echo "ERROR: i8mm unit tests FAILED on hardware - do not trust the bench." >&2; exit 1; }

# ---- provenance ----
CPU_TYPE="$(lscpu 2>/dev/null | awk -F: '/Model name/{gsub(/^ +/,"",$2);print $2;exit}')"
[ -z "${CPU_TYPE:-}" ] && CPU_TYPE="$(uname -m)"
HOST="$(hostname)"
CANDLE_GIT="$(cd "$CANDLE_DIR" && git rev-parse --short HEAD 2>/dev/null || echo '?')"
CANDLE_BRANCH="$(cd "$CANDLE_DIR" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
MODEL_ID="$(basename "$MODEL")"
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

printf '\n==== provenance ====\n'
printf '  host          : %s\n' "$HOST"
printf '  cpu           : %s\n' "$CPU_TYPE"
printf '  pp / tg / reps: [%s] / %s / %s\n' "$PP" "$TG" "$REPS"
printf '  candle        : %s @ %s (%s)  features=[%s]\n' \
  "$MODEL_ID" "$CANDLE_BRANCH" "$CANDLE_GIT" "$CANDLE_FEATURES"
printf '  rustflags     : %s\n' "$RUSTFLAGS"
[ -n "$LLAMA_BENCH" ] && printf '  llama.cpp     : %s\n' "$LLAMA_BENCH"

if [ -n "$CSV" ]; then
  mkdir -p "$(dirname "$CSV")"
  [ ! -f "$CSV" ] && echo "stamp,host,cpu,threads,pp,tg,reps,engine,model,candle_branch,candle_git,candle_features,pp_tok_s,tg_tok_s,peak_mb,pp_speedup_i8mm_over_sdot" >"$CSV"
fi

# Run "$@" to $1, polling peak RSS (KB), echoed last. (Same helper as compare.sh.)
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

# Run the candle bench once; args: outfile cores pp i8mm(0|1). Echoes peak RSS KB.
run_candle() {
  local out="$1" c="$2" pp="$3" i8="$4"
  run_with_rss "$out" \
    env RAYON_NUM_THREADS="$c" CANDLE_NUM_THREADS="$c" \
    CANDLE_QMATMUL_DECODE_THREADS="$c" CANDLE_QMATMUL_PREFILL_THREADS="$c" \
    CANDLE_KV_PREALLOC="$CANDLE_KV_PREALLOC" \
    CANDLE_MATMUL_PACKED_Q4K=1 CANDLE_PREFILL_I8MM="$i8" \
    taskset -c "0-$((c - 1))" "$CANDLE_BIN" \
    --model "$MODEL" --pp "$pp" --tg "$TG" --reps "$REPS" --json
}

ts=$(mktemp); ti=$(mktemp); tl=$(mktemp)
trap 'rm -f "$ts" "$ti" "$tl"' EXIT

l_med='def med: sort | if length==0 then "nan" elif length%2==1 then .[length/2|floor] else (.[length/2-1]+.[length/2])/2 end;'

printf '\n%-6s %-6s %-14s %-12s %-12s %-10s\n' cores pp engine pp_t/s tg_t/s peak_MB
printf '%s\n' "----------------------------------------------------------------------"

for c in $CORES; do
  for pp in $PP; do
    s_rss=$(run_candle "$ts" "$c" "$pp" 0)   # SDOT
    s_pp=$(jq -r '.pp_tok_s_median' "$ts"); s_tg=$(jq -r '.tg_tok_s_median' "$ts")
    i_rss=$(run_candle "$ti" "$c" "$pp" 1)   # i8mm/SMMLA
    i_pp=$(jq -r '.pp_tok_s_median' "$ti"); i_tg=$(jq -r '.tg_tok_s_median' "$ti")

    s_mb=$(echo "$s_rss/1024" | bc -l); i_mb=$(echo "$i_rss/1024" | bc -l)
    spd=$(awk -v a="$i_pp" -v b="$s_pp" 'BEGIN{printf "%.4f", a/b}')

    printf '%-6s %-6s %-14s %-12.2f %-12.2f %-10.1f\n' "$c" "$pp" candle-sdot  "$s_pp" "$s_tg" "$s_mb"
    printf '%-6s %-6s %-14s %-12.2f %-12.2f %-10.1f\n' "$c" "$pp" candle-i8mm  "$i_pp" "$i_tg" "$i_mb"
    printf '%-6s %-6s %-14s %-12s\n' "$c" "$pp" "i8mm/sdot pp" "$(printf '%.2fx' "$spd")"

    if [ -n "$LLAMA_BENCH" ]; then
      run_with_rss "$tl" taskset -c "0-$((c - 1))" "$LLAMA_BENCH" \
        -m "$LLAMA_MODEL" -t "$c" -p "$pp" -n "$TG" -r "$REPS" -o json >/dev/null
      l_pp=$(jq -r "$l_med"'[.[]|select(.n_gen==0)][0].samples_ts | med' "$tl")
      l_tg=$(jq -r "$l_med"'[.[]|select(.n_prompt==0)][0].samples_ts | med' "$tl")
      ivl=$(awk -v a="$i_pp" -v b="$l_pp" 'BEGIN{printf "%.2f", a/b}')
      printf '%-6s %-6s %-14s %-12.2f %-12.2f\n' "$c" "$pp" llama.cpp "$l_pp" "$l_tg"
      printf '%-6s %-6s %-14s %-12s\n' "$c" "$pp" "i8mm/llama pp" "${ivl}x"
    fi
    echo

    if [ -n "$CSV" ]; then
      base="$STAMP,$HOST,\"$CPU_TYPE\",$c,$pp,$TG,$REPS"
      meta="$MODEL_ID,$CANDLE_BRANCH,$CANDLE_GIT,$CANDLE_FEATURES"
      printf '%s,candle-sdot,%s,%.2f,%.2f,%.1f,\n' "$base" "$meta" "$s_pp" "$s_tg" "$s_mb" >>"$CSV"
      printf '%s,candle-i8mm,%s,%.2f,%.2f,%.1f,%.4f\n' "$base" "$meta" "$i_pp" "$i_tg" "$i_mb" "$spd" >>"$CSV"
    fi
  done
done

[ -n "$CSV" ] && echo "appended rows to $CSV"
echo
echo "NOTE: i8mm peak_MB > sdot peak_MB is expected - the i8mm host caches BOTH"
echo "      weight layouts (SDOT BlockQ4Kx8 for decode + laneq for prefill)."
echo "REMEMBER: stop the c7g/c8g box now that the bench is done (it is billable)."
