# Canonical benchmark pipeline: candle vs llama.cpp on Graviton/N1

The single source of truth for how we run candle-vs-llama.cpp speed experiments for
the Qwen3-0.6B Lambda deploy. Read this before running anything - it exists so we stop
re-making the same experimental mistakes (wrong model, stale build, mislabeled run).

Deploy target is AWS Lambda on Graviton2 / Neoverse-N1 (no vendor BLAS, homogeneous
cores). M1 is NOT representative (Accelerate AMX prefill + E-core decode penalty) - use
it only as a smoke test, never for a ratio we report.

---

## 1. The models (know exactly what each one is)

All derived from the upstream unsloth `Qwen3-0.6B-Q4_K_M.gguf`. Dtypes below are from
llama's own reader (`llama-gguf <model> r`, second pass prints `type = ...`).

| model | token_embd / lm_head | attn_v, ffn_down (residual) | other matmul | who reads it |
|---|---|---|---|---|
| `Qwen3-0.6B-Q4_K_M`   | **q6_K** | **q6_K** (28 tensors) | q4_K | llama + candle |
| `Qwen3-0.6B-Q4out`    | **q4_K** (requanted) | **q6_K** (kept) | q4_K | llama + candle |
| `Qwen3-0.6B-Q4packed` | q4_K (tied->Q4Kx8 output) | **q6_K** (unpacked) | **Q4Kx8** | candle ONLY |
| `Qwen3-0.6B-Q6packed` | q4_K (tied->Q4Kx8 output) | **Q6Kx8** (packed) | **Q4Kx8** | candle ONLY |

Key facts that have bitten us:

- **Q4out does NOT drop the Q6_K residual.** Only the tied embedding/lm_head is
  requanted q6_K -> q4_K. Every `ffn_down`/`attn_v` stays q6_K. Proven:
  `llama-gguf Q4out r` shows `blk.5.ffn_down.weight ... type = q6_K`,
  `token_embd.weight ... type = q4_K`.
- That lighter q4_K lm_head is why **llama decodes faster on Q4out than on Q4_K_M**
  (~21.8 vs ~19.5 t/s @ 1 vCPU, 128/256) - the output projection is half the bytes.
  This is expected, not a bug.
- `Q4packed`/`Q6packed` are **candle-fork-only** (GgmlDType::Q4Kx8=1000, Q6Kx8=1001).
  llama.cpp cannot read them. They are `Q4out`'s identical weights, pre-interleaved.

Build the packed artifacts with `qwen-lambda/pack_model.sh` (Q4_K_M -> Q4out -> pack).
`gguf-requant --pack` now packs BOTH q4_K (->Q4Kx8) and q6_K (->Q6Kx8) matmul weights.

---

## 2. The comparison is weight-matched, not file-matched

candle and llama run DIFFERENT files that hold the SAME numeric weights, each in that
engine's fast layout:

- **candle** loads `Qwen3-0.6B-Q6packed.gguf` (our offline Q4Kx8 + Q6Kx8 interleave).
- **llama** loads `Qwen3-0.6B-Q4out.gguf` and repacks q4_K/q6_K -> its `8x4` layout at
  load (`ggml_gemv_q4_K_8x4_q8_K`).

Both carry the same 169 q4_K + 28 q6_K matmul weights and the same q4_K lm_head. Neither
side has a dtype or weight advantage; we are measuring the two repacking IMPLEMENTATIONS.
Do NOT run llama on Q4_K_M while running candle on Q6packed - that compares different
lm_heads and is the #1 way to get a misleading ratio.

`compare.sh` supports this via `CANDLE_MODEL` and `LLAMA_MODEL` (both default to `MODEL`).

---

## 3. "Best candle" build config (must match the deploy)

The decomposed candle branch defaults some deploy wins OFF. To measure best candle:

- **`--features f16-attn-dot`** (cargo, build-time). candle-nn defaults it OFF for
  upstream bit-exactness; the deploy (branch `fused-int8-gemm` candle-nn) has it ON.
  ~+4.5% N1 decode. `compare.sh` builds with `CANDLE_FEATURES=f16-attn-dot` by default.
  (The candle-examples passthrough feature was added 2026-06-21; before that the bench
  silently built WITHOUT it - a measurement error.)
- **`CANDLE_KV_PREALLOC=512`** (env) - matches the deploy.
- **`RUSTFLAGS=-C target-cpu=native`** - Graviton2 == native on the box.
- Pre-packed model (Q6packed) so no runtime repack; packed kernels dispatch
  automatically from the Q4Kx8/Q6Kx8 dtype.
- sdot-chain, R-tune, fused-rope, flash kernels, rayon-trim are baked into the branch.

Sanity check that packing is engaged: the candle packing ladder must be monotonic in
prefill (Q4_K_M < Q4packed < Q6packed). If Q6packed isn't fastest in prefill, the
packed path is not active - stop and debug before trusting any ratio.

---

## 4. Provenance: every number must be self-describing

A ratio is meaningless without its operating point. **The c/l ratio moves along the
pp/tg curve AND the thread count** - 0.85x decode at tg=256 becomes 0.92x at tg=512 on
the same box. So we record, per row (`bench/compare_results.csv`, written by compare.sh):

`stamp, host, cpu, threads, pp, tg, reps, engine, candle_model, llama_model,
candle_branch, candle_git, candle_features, llama_git, pp_tok_s, tg_tok_s, peak_mb,
pp_ratio, tg_ratio`

Mistakes this guards against (all real, all hit us):

- **Stale `.git`**: we rsync candle SOURCE to the box but excluded `.git`, so
  `git rev-parse` reported the box's old `QK_4_GEMV` branch while the binary was actually
  `explore/rayon-trim-q6k-packing@2dacebc2`. Fix: sync `.git` too (it's only ~27 MB) so
  provenance is truthful. ALWAYS verify the printed branch/commit matches what you built.
- **Wrong feature set** (f16-attn-dot off) - now labeled in `candle_features`.
- **Wrong model** (Q4_K_M vs Q4out) - now labeled in `candle_model`/`llama_model`.

---

## 5. How to run (the recipe)

On the EC2 N1 box (`i-05bea320332bed88b`, c6g.2xlarge, us-east-1; resume per
`ec2-parity-bench-box` memory; STOP it when done):

```bash
cd ~/ml-lambdas
M=~/ml-lambdas/qwen-lambda/models
MODEL=$M/Qwen3-0.6B-Q4out.gguf \
CANDLE_MODEL=$M/Qwen3-0.6B-Q6packed.gguf \
LLAMA_MODEL=$M/Qwen3-0.6B-Q4out.gguf \
CANDLE_DIR=~/candle \
LLAMA_BENCH=~/llama.cpp/build/bin/llama-bench \
CANDLE_FEATURES=f16-attn-dot \
CORES="1" PP=128 TG=256 REPS=3 \
bash bench/compare.sh
```

Rules:
- **Validate 1 vCPU FIRST**, confirm setup, THEN scale threads (1 2 3 4 6). Don't burn a
  full matrix on an unvalidated config.
- **Pin cores** with taskset (compare.sh does `taskset -c 0-(c-1)`); set candle thread
  pools explicitly (`CANDLE_QMATMUL_{DECODE,PREFILL}_THREADS`) - `available_parallelism`
  sees host cores, not the simulated tier.
- **One run at a time.** Concurrent benches share the box CPU and contaminate each
  other's timings (and lose stdout). Serialize.
- **reps=2-3 is plenty** - the candle bench is extremely stable (pp spread ~0.03% over
  10 reps). No need for 5+.
- Both sides use the MEDIAN statistic (compare.sh medians llama-bench's per-rep
  `samples_ts`, candle reports `*_median`).

---

## 6. Validated results (N1, Neoverse-N1 c6g.2xlarge, 1 vCPU)

candle `explore/rayon-trim-q6k-packing@2dacebc2` features=[f16-attn-dot] kv_prealloc=512;
llama-bench `c1304d7b2`. candle=Q6packed, llama=Q4out (weight-matched).

| pp/tg | candle pp | llama pp | pp c/l | candle tg | llama tg | tg c/l | candle MB | llama MB |
|---|---|---|---|---|---|---|---|---|
| 128/256 | 36.42 | 50.84 | 0.72x | 18.66 | 21.85 | 0.85x | 559 | 755 |
| 128/512 | 36.61 | 51.13 | 0.72x | 17.68 | 19.18 | 0.92x | 664 | 778 |

candle packing ladder (1 vCPU, 128/256, proves Q6Kx8 win + packing engaged):

| candle model | prefill | decode |
|---|---|---|
| Q4_K_M (stock) | 28.29 | 14.21 |
| Q4out | 28.33 | 16.28 |
| Q4packed (Q4Kx8) | 33.73 | 18.99 |
| **Q6packed (+Q6Kx8)** | **36.66** | 18.89 |

Reading:
- **Q6Kx8 packing buys +8.7% prefill** (33.73 -> 36.66), **decode neutral** (memory-BW
  bound at m=1, packing changes no bytes moved). Matches the "same speed, keeps Q6
  quality" goal with a prefill bonus.
- **candle decode is near parity** (0.85-0.92x, improves with longer gen) and uses ~26%
  less RAM (good for Lambda memory tiers).
- **candle prefill is the gap** (steady 0.72x). The packed-SDOT prefill GEMM loses to
  llama's i8mm `8x4`. This is the improvement target, not the weights.
  - **In progress (i8mm/SMMLA prefill kernel):** a faithful port of llama's
    `ggml_gemm_q4_K_8x8_q8_K` i8mm branch now exists in candle
    (`gemm_q4kx8_q8k_i8mm` + `matmul_q4kx8l_prepacked_i8mm`), wired into the
    prefill dispatch (m>=4, `+i8mm` builds). Validated locally via a scalar SMMLA
    twin (bit-identical to hardware). NEEDS a Graviton3 (c7g) / Graviton4 (c8g) box
    to run - N1/Graviton2 has no i8mm. Bench it with `bench/i8mm_prefill.sh` (same
    binary, `CANDLE_PREFILL_I8MM=1` vs `=0`, at pp512/2048/4096). The existing
    parity box is Graviton2 - a NEW c7g/c8g instance is required.

---

## 7. Open work (in priority order)

1. **Instruction-parity breakdown** (the per-component analysis): perf-stat
   instructions / IPC / L1-miss for each phase (matmul prefill, matmul decode, flash,
   rope, norm, sampling, activation-quant) to localize the 0.72x prefill gap. Kernel
   microbench already shows our Q4K kernel is leaner per-instruction; the gap is i8mm
   (llama prefill) + glue, so attribute it precisely before optimizing.
2. **Multithread** (1 2 3 4 6 vCPU): does candle close or widen the gap as cores rise?
   rayon vs llama threading; likely interacts with packing.
3. **KV cache** optimization (in relation to packing), then larger prefill/gen.
4. **Scale-up matrix**: threads x pp/tg, full CSV, only after the above are understood.
