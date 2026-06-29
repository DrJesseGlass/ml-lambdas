#!/usr/bin/env bash
# Cold-start measurement for the deployed qwen-lambda. Forces a cold container,
# then invokes once cold + once warm and prints the breakdown from:
#   (a) the AWS REPORT line  - Init Duration, Duration, Billed, Max Memory Used
#   (b) the response JSON     - cold_start, load_ms, warmup_ms, prefill_ms, decode_ms
#   (c) our stderr log        - COLDSTART gguf_read_ms / model_build_ms (load split)
#
# Cold start is forced by bumping a nonce env var, which recycles all warm
# containers (the env merge preserves existing vars). Requires aws cli + jq and a
# deployed container Lambda.
#
# Usage:  FN=<function-name> [REGION=us-east-1] [PROMPT="..."] ./coldstart.sh
set -euo pipefail
FN=${FN:?set FN=<lambda function name>}
REGION=${REGION:-us-east-1}
PROMPT=${PROMPT:-"What is the capital of France?"}
MAXTOK=${MAXTOK:-16}
PAYLOAD=$(jq -nc --arg p "$PROMPT" --argjson n "$MAXTOK" '{prompt:$p,max_tokens:$n,temperature:0}')
RESP=/tmp/coldstart_resp.json

echo "=== forcing cold start: bump COLDSTART_NONCE (merged into existing env) ==="
CUR=$(aws lambda get-function-configuration --function-name "$FN" --region "$REGION" \
  --query 'Environment.Variables' --output json)
[ "$CUR" = "null" ] && CUR='{}'
NEWENV=$(jq -nc --argjson v "$CUR" --arg n "$(date +%s)" '{Variables: ($v + {COLDSTART_NONCE:$n})}')
aws lambda update-function-configuration --function-name "$FN" --region "$REGION" \
  --environment "$NEWENV" >/dev/null
aws lambda wait function-updated --function-name "$FN" --region "$REGION"

invoke() { # label
  local label=$1 logb64
  logb64=$(aws lambda invoke --function-name "$FN" --region "$REGION" \
    --cli-binary-format raw-in-base64-out --payload "$PAYLOAD" \
    --log-type Tail --query 'LogResult' --output text "$RESP")
  echo "--- $label: response ---"
  jq -c '{cold_start,load_ms,warmup_ms,prompt_tokens,generated_tokens,prefill_ms,decode_ms}' "$RESP" \
    2>/dev/null || cat "$RESP"
  echo "--- $label: logs (load split + AWS REPORT) ---"
  echo "$logb64" | base64 --decode | grep -E "COLDSTART|cold-start init|^REPORT" || echo "  (no matching log lines)"
  echo
}

echo "=== COLD invoke (paid the load + warmup) ==="
invoke cold
echo "=== WARM invoke (reused container) ==="
invoke warm

echo "=== read: Init Duration = container+load+warmup; Duration(cold) ~ first request;"
echo "    load_ms = gguf_read + model_build; warmup_ms = KV alloc + repack + 1 fwd ==="
