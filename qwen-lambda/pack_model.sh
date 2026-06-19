#!/usr/bin/env bash
# Produce the deploy artifact models/Qwen3-0.6B-Q4packed.gguf.
#
# Pipeline (the deploy recipe):
#   Qwen3-0.6B-Q4_K_M.gguf                      (upstream, unsloth)
#     --requant tied embedding/lm_head Q6_K->Q4_K-->  Qwen3-0.6B-Q4out.gguf
#     --pack (offline BlockQ4Kx8 interleave)    -->   Qwen3-0.6B-Q4packed.gguf
#
# Q4out.gguf is a STANDARD, llama.cpp-loadable GGUF (the one to host on HuggingFace).
# Q4packed.gguf is CANDLE-FORK-ONLY (GgmlDType::Q4Kx8): a single in-RAM copy with no
# runtime repack, ~16% faster 1-vCPU Graviton2 decode than llama.cpp. Run once at
# deploy/CI (it is deterministic).
#
# Quality: requant moves the lm_head Q6_K->Q4_K only (+2.2% wikitext-style PPL);
# attn_v/ffn_down stay Q6_K (Q4_K_M's protected tensors). NOT standard Q4_K_M.
set -euo pipefail
cd "$(dirname "$0")"

# Path to the DrJesseGlass/candle fork (must be the branch with Q4Kx8 support).
CANDLE_DIR="${CANDLE_DIR:-../../candle}"
MODELS=models

# 1. Ensure the upstream Q4_K_M GGUF + tokenizer are present (both, for the image).
if [ ! -f "$MODELS/Qwen3-0.6B-Q4_K_M.gguf" ] || [ ! -f "$MODELS/tokenizer.json" ]; then
  ./fetch_model.sh
fi

# 2. Build the offline requantizer/packer from the candle fork.
GR="$CANDLE_DIR/target/release/examples/gguf-requant"
if [ ! -x "$GR" ]; then
  echo "Building gguf-requant in $CANDLE_DIR ..."
  ( cd "$CANDLE_DIR" && RUSTFLAGS="-C target-cpu=native" \
      cargo build --release --example gguf-requant )
fi

# 3. Requant tied embedding/lm_head Q6_K -> Q4_K  (Q4out: standard, HF-hostable).
#    If you host Q4out.gguf on HF, replace this with a curl download instead.
if [ ! -f "$MODELS/Qwen3-0.6B-Q4out.gguf" ]; then
  "$GR" --input  "$MODELS/Qwen3-0.6B-Q4_K_M.gguf" \
        --output "$MODELS/Qwen3-0.6B-Q4out.gguf" \
        --tensors token_embd,output.weight --dtype q4k
fi

# 4. Bake the pre-packed Q4Kx8 layout (Q4packed: the candle-fork deploy artifact).
"$GR" --input  "$MODELS/Qwen3-0.6B-Q4out.gguf" \
      --output "$MODELS/Qwen3-0.6B-Q4packed.gguf" --pack

ls -lh "$MODELS"/*.gguf
