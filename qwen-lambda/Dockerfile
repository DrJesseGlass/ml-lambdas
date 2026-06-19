# Multi-stage build for the qwen-lambda container image (arm64 / Graviton2 default).
#
# Build context is the WORKSPACE ROOT (ml-lambdas/) so the shared lambda-core
# crate and the lockfile are visible. Build NATIVELY for the target arch to avoid
# cross-compiling the C deps (onig). On an Apple-Silicon host `--platform
# linux/arm64` runs without emulation:
#
#   docker build --platform linux/arm64 -f qwen-lambda/Dockerfile -t qwen-lambda:arm64 .
#
# x86 baseline (the one apples-to-apples comparison run):
#
#   docker build --platform linux/amd64 -f qwen-lambda/Dockerfile \
#     --build-arg TARGET_CPU=x86-64-v3 \
#     --build-arg RUNTIME_BASE=public.ecr.aws/lambda/provided:al2023-x86_64 \
#     -t qwen-lambda:amd64 .

ARG RUNTIME_BASE=public.ecr.aws/lambda/provided:al2023-arm64

# ---- Builder ----------------------------------------------------------------
FROM rust:1-bookworm AS builder
# Graviton2 == Neoverse N1 (NEON + dotprod), the path the kernels were tuned on.
ARG TARGET_CPU=neoverse-n1
ENV RUSTFLAGS="-C target-cpu=${TARGET_CPU}"
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake clang pkg-config && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY lambda-core ./lambda-core
COPY qwen-lambda/Cargo.toml ./qwen-lambda/Cargo.toml
COPY qwen-lambda/src ./qwen-lambda/src
RUN cargo build --release -p qwen-lambda && cp target/release/qwen-lambda /bootstrap

# ---- Runtime ----------------------------------------------------------------
FROM ${RUNTIME_BASE}
# Bake the model + tokenizer into the image (no cold-start download).
#
# MODEL_FILE selects the baked GGUF (default = the candle-packed deploy artifact,
# produced by `./pack_model.sh` BEFORE this build):
#   Qwen3-0.6B-Q4packed.gguf  -> pre-packed Q4Kx8, single RAM copy, no runtime
#                                repack, ~16% faster 1-vCPU decode than llama.cpp.
#   Qwen3-0.6B-Q4out.gguf     -> standard/portable Q4_K (set CANDLE_MATMUL_PACKED_Q4K=1
#                                to runtime-pack at the +~250MB memory cost).
ARG MODEL_FILE=Qwen3-0.6B-Q4packed.gguf
COPY qwen-lambda/models/${MODEL_FILE} /opt/model/model.gguf
COPY qwen-lambda/models/tokenizer.json /opt/model/tokenizer.json
ENV MODEL_PATH=/opt/model/model.gguf \
    TOKENIZER_PATH=/opt/model/tokenizer.json \
    # KV cache initial prealloc per layer (the cache grows on demand). Size to the
    # task's max context to cut idle RAM (~0.11 MB/position): 512=58MB, 256=29MB,
    # 128=15MB vs the 1024 default's 117MB. Override per Lambda function as needed.
    CANDLE_KV_PREALLOC=512
# candle's quantized matmul pools auto-size to the Lambda's allocated vCPUs
# (std::thread::available_parallelism), so no thread env is required.
# The custom-runtime base executes /var/runtime/bootstrap.
COPY --from=builder /bootstrap /var/runtime/bootstrap
