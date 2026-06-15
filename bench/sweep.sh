#!/usr/bin/env bash
# Tier-sweep cost/token harness.
#
# For each Lambda memory tier: pin the qmatmul thread pools to that tier's vCPU
# count (Lambda grants ~1 vCPU per 1769 MB; `available_parallelism()` reports the
# HOST cores, not your quota, so we MUST set these explicitly), invoke the
# function, and read back per-phase tok/s plus billed duration.
#
# Reports cost/token so the winner is the tier at the knee of the scaling curve,
# not the fastest raw tok/s.
#
# Usage: FN=qwen-lambda ./bench/sweep.sh
set -euo pipefail

FN="${FN:?set FN to the Lambda function name}"
PROMPT="${PROMPT:-Write a Rust function to calculate the factorial of a given number.}"
MAX_TOKENS="${MAX_TOKENS:-256}"
# arm64 Lambda price per GB-second (us-east-1, on-demand). Adjust per region/arch.
PRICE_PER_GB_S="${PRICE_PER_GB_S:-0.0000133334}"

# "memory_mb:vcpus" — vCPUs is round(memory/1769). Edit to taste.
TIERS=("1769:1" "3540:2" "5308:3" "7076:4" "8848:5" "10240:6")

printf '%-8s %-6s %-12s %-12s %-12s %-14s\n' \
  MB vCPU prefill_t/s decode_t/s billed_ms usd_per_1k_tok

for tier in "${TIERS[@]}"; do
  mb="${tier%%:*}"; vcpu="${tier##*:}"

  aws lambda update-function-configuration \
    --function-name "$FN" \
    --memory-size "$mb" \
    --environment "Variables={MODEL_PATH=/opt/model/Qwen3-0.6B-Q4_K_M.gguf,TOKENIZER_PATH=/opt/model/tokenizer.json,CANDLE_QMATMUL_DECODE_THREADS=$vcpu,CANDLE_QMATMUL_PREFILL_THREADS=$vcpu}" \
    >/dev/null
  aws lambda wait function-updated --function-name "$FN"

  # Warm the container first so we measure steady-state, not cold start.
  payload=$(printf '{"prompt":%s,"max_tokens":%s,"temperature":0}' \
    "$(jq -Rn --arg p "$PROMPT" '$p')" "$MAX_TOKENS")
  aws lambda invoke --function-name "$FN" --payload "$payload" \
    --cli-binary-format raw-in-base64-out /tmp/warm.json >/dev/null

  out=$(aws lambda invoke --function-name "$FN" --payload "$payload" \
    --cli-binary-format raw-in-base64-out --log-type Tail \
    --query 'LogResult' --output text /tmp/out.json | base64 -d)

  pf=$(jq -r '.prefill_tok_s' /tmp/out.json)
  dc=$(jq -r '.decode_tok_s' /tmp/out.json)
  gen=$(jq -r '.generated_tokens' /tmp/out.json)
  billed=$(echo "$out" | sed -n 's/.*Billed Duration: \([0-9]*\) ms.*/\1/p' | head -1)

  # cost/token over the billed wall time of this invocation.
  usd_per_1k=$(awk -v mb="$mb" -v ms="$billed" -v g="$gen" -v p="$PRICE_PER_GB_S" \
    'BEGIN{ if(g>0){ printf "%.6f", (mb/1024)*(ms/1000)*p/g*1000 } else { print "n/a" } }')

  printf '%-8s %-6s %-12.2f %-12.2f %-12s %-14s\n' \
    "$mb" "$vcpu" "$pf" "$dc" "$billed" "$usd_per_1k"
done
