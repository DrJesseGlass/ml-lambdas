# ml-lambdas

A Cargo workspace of AWS Lambda services that run quantized LLMs on CPU via the
optimized [candle fork](https://github.com/DrJesseGlass/candle) (`QK_4_GEMV`
branch). One crate per model; a shared `lambda-core` crate owns everything that
isn't model-specific.

| Crate | What |
|---|---|
| `lambda-core` | Model-agnostic serving harness: request/response types, warm-once lifecycle, request mutex, `spawn_blocking`, JSON IO, per-phase timing. No ML deps. Exposes the `TextModel` trait + `serve()`. |
| `qwen-lambda` | Qwen3-0.6B Q4_K_M. Implements `TextModel`; `main` is ~10 lines. |
| `bench/` | Shared `sweep.sh` cost/token harness (function-name driven). |

Adding a model = a new crate that implements `TextModel` and calls
`lambda_core::serve(loader)`. The public/private split is clean: `lambda-core` is
public; a private model can be another crate here, or a separate **private** repo
that depends on `lambda-core` by git — nothing else changes.

## Defaults

- **Arch:** arm64 / Graviton2 (Neoverse N1 — NEON + dotprod, the tuned path; ~20%
  cheaper per GB-s than x86). x86 is a one-flag baseline only — Lambda won't
  reliably schedule AVX-512 hosts, and the `_xr`/SDOT kernels are NEON-only.
- **Packaging:** container image, model baked in (no cold-start download).
- **Interface:** JSON request/response (built for the cost/token benchmark).

## Build & deploy (qwen-lambda)

Run from the workspace root; the Docker build context is the root so the shared
crate is visible.

```bash
# 1. Fetch weights (~390 MB GGUF + tokenizer) into qwen-lambda/models
./qwen-lambda/fetch_model.sh

# 2. Build the arm64 image (native on Apple Silicon — no emulation)
docker build --platform linux/arm64 -f qwen-lambda/Dockerfile -t qwen-lambda:arm64 .

# 3. Push to ECR
aws ecr create-repository --repository-name qwen-lambda || true
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)
REPO="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/qwen-lambda"
aws ecr get-login-password | docker login --username AWS --password-stdin "$REPO"
docker tag qwen-lambda:arm64 "$REPO:arm64"
docker push "$REPO:arm64"

# 4. Create the function (arm64; generous tier to start — tune via the sweep)
aws lambda create-function \
  --function-name qwen-lambda \
  --package-type Image --code ImageUri="$REPO:arm64" \
  --architectures arm64 --memory-size 3540 --timeout 120 \
  --role <execution-role-arn> \
  --environment 'Variables={CANDLE_QMATMUL_DECODE_THREADS=2,CANDLE_QMATMUL_PREFILL_THREADS=2}'
```

## Invoke

```bash
aws lambda invoke --function-name qwen-lambda \
  --cli-binary-format raw-in-base64-out \
  --payload '{"prompt":"Explain quicksort.","max_tokens":128,"temperature":0}' \
  /dev/stdout
```

Response: `prefill_tok_s`, `decode_tok_s`, `prompt_tokens`, `generated_tokens`,
`cold_start`, `load_ms`.

## The thread-pinning gotcha (read this)

The candle qmatmul pools size themselves from `available_parallelism()`, which on
Lambda returns the **host's** core count, not the vCPU fraction your memory tier
grants (Lambda throttles via CFS quota, not core masking). Unset, the pools
oversubscribe and thrash. **Always set `CANDLE_QMATMUL_DECODE_THREADS` and
`CANDLE_QMATMUL_PREFILL_THREADS`** to the tier's vCPU count (≈ `round(memory_mb /
1769)`). `bench/sweep.sh` does this per tier.

## Benchmarking

Two layers, because they answer different questions.

### 1. Parity vs llama.cpp — `bench/compare.sh` (EC2 Graviton2)

The head-to-head. Runs both engines on the **same box, same GGUF, pinned to the
same physical cores** with `taskset`, and measures prefill (pp) + decode (tg)
tok/s and peak RSS. This is the fast iteration loop — re-run after each candle
change without a deploy cycle. Matched to `llama-bench` methodology via the
`quantized-qwen3-bench` candle example (dummy-token prefill, greedy decode, N
reps, median) so the numbers are comparable.

```bash
MODEL=~/ml-lambdas/qwen-lambda/models/Qwen3-0.6B-Q4_K_M.gguf \
CANDLE_DIR=~/candle LLAMA_BENCH=~/llama.cpp/build/bin/llama-bench \
CORES="2 4" ./bench/compare.sh
```

Prints, per core count, a candle / llama.cpp / `ratio c/l` row. `taskset -c
0-1` + `*_THREADS=2` simulates a 2-vCPU tier; sweep `CORES` to cover the tiers.
Caveat: this is a faithful *relative* result and approximate absolute — EC2
dedicated cores ≠ Lambda's CFS-quota vCPU, so cost/token still comes from layer 2.

### 2. Cost/token — `bench/sweep.sh` (real Lambda)

The production truth. Target is **minimum cost/token**, not max tok/s — a bigger
tier isn't wasteful if the extra vCPUs buy proportional speedup. Prefill scales
with vCPUs; decode (bandwidth-bound) plateaus early, so the sweet-spot tier
differs by workload. Only Lambda gives real cost/token and cold-start (microVM,
CFS throttling, tiered vCPU).

```bash
FN=qwen-lambda ./bench/sweep.sh
```

### Workflow

Iterate candle + run `compare.sh` on EC2 until the parity ratio is where you want
it, then deploy and run `sweep.sh` for the cost/token verdict at the chosen tier.

## Notes

- The candle dep is pinned to `branch = "QK_4_GEMV"` (public). If a future fork
  branch is private, the in-Docker `cargo build` needs creds (BuildKit `--ssh
  default` + an `ssh://` git URL).
