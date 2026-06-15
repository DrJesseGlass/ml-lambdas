#!/usr/bin/env bash
# Download the Qwen3-0.6B Q4_K_M GGUF + tokenizer into ./models so the Docker
# build can bake them in. Uses curl (hf-hub downloads have stalled on this repo).
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p models

GGUF_URL="https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf"
TOK_URL="https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json"

echo "Fetching GGUF weights..."
curl -fL --retry 3 -o models/Qwen3-0.6B-Q4_K_M.gguf "$GGUF_URL"
echo "Fetching tokenizer..."
curl -fL --retry 3 -o models/tokenizer.json "$TOK_URL"

ls -lh models/
