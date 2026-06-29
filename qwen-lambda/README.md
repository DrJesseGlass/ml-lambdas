# qwen-lambda

Qwen3-0.6B served on AWS Lambda (Graviton2 / Neoverse-N1) via a candle fork tuned
for the most efficient possible serverless inference. On a 1-vCPU Lambda it
**beats llama.cpp decode by ~16%** at a single in-RAM weight copy.

## Deploy recipe (the 4 levers)

1. **Model = pre-packed `Q4packed.gguf`.** Built offline from the upstream
   `Q4_K_M.gguf` by `./pack_model.sh`:
   - requant the tied embedding/lm_head **Q6_K -> Q4_K** (`Q4out.gguf`, +2.2% PPL;
     `attn_v`/`ffn_down` stay Q6_K) - a *standard, llama.cpp-loadable* GGUF;
   - `--pack` it into the candle-only **`Q4Kx8`** interleaved layout (`Q4packed.gguf`)
     so the model loads as a **single copy with no runtime repack** and runs the fast
     packed SDOT kernel directly.
2. **`CANDLE_KV_PREALLOC` = the task's max context.** The KV cache grows on demand,
   so this is just the initial per-layer reservation. ~0.11 MB/position, so
   512 ~ 58 MB, 256 ~ 29 MB, 128 ~ 15 MB vs the 1024 default's 117 MB. Set per
   function. (Default baked in the image: 512.)
3. **Plain `read` load** (`engine.rs` -> `from_gguf`). mmap exists in the fork but only
   helps *load* time, not RAM, and inflates RSS via readahead - `read` is the clean
   path.
4. **Pools auto-size to the Lambda's vCPUs** (`std::thread::available_parallelism`),
   so no thread env is needed.

## Build & deploy

```bash
# 0. (one-time) PUSH the candle fork branch the Cargo.toml pins:
#    DrJesseGlass/candle  branch  fused-int8-gemm   (carries Q4Kx8 + gguf-requant --pack)

# 1. Fetch the upstream Q4_K_M GGUF + tokenizer, then bake the packed artifact.
#    CANDLE_DIR points at the candle fork checkout (default ../../candle).
./fetch_model.sh
CANDLE_DIR=../../candle ./pack_model.sh      # -> models/Qwen3-0.6B-Q4packed.gguf

# 2. Build the arm64 image (build context is the workspace root). The packed GGUF
#    is baked in; CANDLE_KV_PREALLOC defaults to 512.
docker build --platform linux/arm64 -f qwen-lambda/Dockerfile -t qwen-lambda:arm64 .

# 3. Push to ECR and deploy as a container Lambda (arm64). Tune the function's
#    CANDLE_KV_PREALLOC env to the workload's max context.
```

To deploy the portable/standard model instead (no candle-fork dependency for the
weights), build with `--build-arg MODEL_FILE=Qwen3-0.6B-Q4out.gguf` and set
`CANDLE_MATMUL_PACKED_Q4K=1` to runtime-pack (costs ~+250 MB RAM).

## Measured (Graviton2 c6g, 1 vCPU, decode)

| config | peak RSS | decode t/s | vs llama.cpp |
|---|---|---|---|
| upstream Q4_K_M, runtime-packed, KV 1024 | ~877 MB | 21.2 | +17% |
| **Q4packed + `CANDLE_KV_PREALLOC=128`** | **~601 MB** | **21.1** | **+16%** |
| (Q4_K, packed off - smallest, slowest) | ~514 MB* | 18.4 | ~tie |

\*KV=128. The packed file is larger on disk (452 vs 357 MB) but a single RAM copy.

## HuggingFace hosting

- **Host `Q4out.gguf`** - it's a standard GGUF (llama.cpp-compatible). Honest name:
  it is *not* `Q4_K_M` (lm_head is Q4_K, not Q6_K); call it e.g.
  `Qwen3-0.6B-q4k-lmhead.gguf`.
- **Do not** publish `Q4packed.gguf` as a general model - it uses `GgmlDType::Q4Kx8`
  (value 1000), readable only by this candle fork and tied to its `BlockQ4Kx8`
  layout version. Produce it at deploy with `pack_model.sh`.
