# Instruction-parity breakdown: candle vs llama.cpp (Qwen3-0.6B, N1, 1 vCPU)

Per-component RAW instruction counts so we know, in absolute numbers, what to shrink in
candle. **llama is the FROZEN reference** (measured once below; its kernels don't change -
never re-run it). Re-run only candle after each tweak and watch the target component fall.

Tooling: `bench/instr_breakdown.sh <engine> <pp> <tg>` - perf-record on `instructions:u`,
maps symbols -> components, multiplies self-share by the perf-stat total. 1 thread, pinned,
single pass (candle `--warmup 0 --reps 1`; llama `-r 1 --no-warmup`). Phases isolated:
prefill = `pp=512 tg=1`, decode = `pp=8 tg=256`.

CRITICAL METHODOLOGY NOTE: llama-bench runs an internal WARMUP by default - it doubles the
prefill instruction count (149.6B -> 78.2B with `--no-warmup`). ALWAYS pass `--no-warmup`
or the llama numbers are ~2x inflated and the comparison is garbage. (This bit us once.)

Measured 2026-06-21: candle `explore/rayon-trim-q6k-packing@2dacebc2` features=[f16-attn-dot]
on Q6packed; llama `c1304d7b2` on Q4out (weight-matched). Neoverse-N1, c6g.2xlarge.

## Prefill (512 tokens, one pass)

| component        | candle (instr) | llama FROZEN (instr) | candle - llama |
|------------------|---------------:|---------------------:|---------------:|
| **total**        | **94.45 B**    | **78.24 B**          | +16.2 B (1.21x)|
| total cycles     | 42.0 B         | 32.08 B              | 1.31x          |
| IPC              | 2.25           | 2.44                 |                |
| **matmul**       | **69.93 B**    | **46.82 B**          | **+23.1 B (1.49x)** |
| other*           | 19.07 B        | 1.35 B               | +17.7 B        |
| attention(flash) | 0.06 B*        | 15.02 B              | *see note      |
| transcendental   | 3.49 B         | 0.04 B               | +3.45 B        |
| act_quant        | 0.37 B         | 4.77 B               | -4.40 B        |
| rope/norm/elem/mem | ~1.6 B       | ~6.5 B               |                |

*Classifier note: candle's prefill flash-attention currently falls into "other"/"transcendental"
(its kernel is an inlined closure that perf labels `&F>::call`, not matched by the attention
regex). llama's flash is cleanly `ggml_compute_forward_flash_attn_ext_tiled` (15.0 B). So
candle "other" 19.07 B + transcendental ~3.5 B is largely candle's prefill attention+softmax.
Fix the classifier / add a `#[inline(never)]` marker before optimizing attention.

## Decode (256 tokens)

| component        | candle (instr) | llama FROZEN (instr) | candle - llama |
|------------------|---------------:|---------------------:|---------------:|
| **total**        | **70.32 B**    | **77.24 B**          | -6.9 B (0.91x) |
| total cycles     | 31.57 B        | 34.50 B              | 0.92x          |
| IPC              | 2.23           | 2.24                 |                |
| **matmul**       | **61.58 B**    | **63.21 B**          | **0.97x (parity)** |
| attention        | 4.68 B         | 2.13 B               | +2.55 B        |
| transcendental   | 1.72 B         | 0.60 B               | +1.12 B        |
| rope             | 0.10 B         | 0.66 B               | -0.56 B        |
| memory           | 0.56 B         | 4.23 B               | -3.67 B        |

## What this says (targets, in priority order)

HARDWARE FACT (verified on the box): Graviton2 / Neoverse-N1 is ARMv8.2-A. It HAS
`asimddp` (SDOT/UDOT dotprod) but NOT `i8mm` (FEAT_I8MM / SMMLA, which is ARMv8.6 - only
Graviton3/Neoverse-V1+). So **SMMLA is NOT an option on the deploy target.** Both engines
use SDOT. (i8mm would only matter if we ever deploy on Graviton3/4.)

1. **PREFILL parity = a WIDER SDOT GEMM micro-tile (not i8mm).** The whole prefill gap is
   matmul: candle 69.93 B vs llama 46.82 B (1.49x), SAME SDOT instruction family. The
   difference is tile width / accumulator count per weight load:
   - llama `ggml_gemm_q4_K_8x4_q8_K` = 8 channels x 4 rows = **32 acc** (also has an 8x8 = 64).
   - candle prefill = `nc4mr4` (4x4 = 16 acc) or `nc8mr2` (8x2 = 16 acc) - HALF the reuse.
   Wider tiles reuse each loaded weight across more activation columns, cutting instructions
   per MAC. Lever: add an 8x4 (or 8x8) SDOT micro-kernel to `matmul_q4kx8_prepacked` /
   `matmul_q6kx8_prepacked` for the m>=4 prefill path. N1 has 32 NEON regs, so 32 acc fits;
   the old "MR=4 spills on X925" note was a different core - re-tune on N1. R-tune already
   raised the `_xr` cap to R<=8, so the scaffolding exists.
   - pp-specific: decode (m=1 GEMV) keeps its current kernel - it's already at parity and a
     wide tile doesn't apply, so this will NOT regress decode.

2. **DECODE is already at parity** - candle is even slightly leaner (0.91x total, 0.97x
   matmul). Do not spend effort on decode matmul. Remaining decode excesses:
   - transcendental: candle 1.72 B vs llama 0.60 B (2.9x). This is `expf` (flash-attention
     softmax) + `sinf/cosf` (RoPE). candle calls scalar libm; llama uses a vectorized
     polynomial `exp`/`soft_max`. Easy-ish win: vectorized expf approximation in the softmax.
   - attention: candle 4.68 B vs llama 2.13 B (2.2x) - the flash decode kernel.

3. **Fix the attention classifier** (candle flash -> "other"/"transcendental") before any
   attention work, so the breakdown attributes it. Candidate: `#[inline(never)]` on the flash
   kernels. Candle's prefill attention is hiding in prefill "other" (19 B).

## CHANGELOG: kernel optimization session (2026-06-22, commit 92e24966)

What shipped, what it gained on N1 (1 vCPU, pp128/tg256), and how it was tested - so we
don't re-derive or forget the feature flags.

### Changes shipped (all bit-exact / coherent)
- **gemm_q4kx_q8k (neon.rs)** - low-register-pressure rewrite of the packed Q4_K GEMM:
  hoisted all base pointers per block; `core::array::from_fn` (no dead `movi` zero-init);
  two-pass lo/hi with only NC*2 live weight regs/pass (NOT a held `q4` array - that spilled
  on N1); **MR==1 (decode) specialized** to stream per-channel (no NC-wide arrays).
- **gemm_q6kx8_q8k (neon.rs)** - pointer-hoisting (the Q6 twin; no held-array/dup-load issue).
- **Vectorized softmax exp (causal.rs)** - GATED `CANDLE_VEC_SOFTMAX_EXP=1`, NEON poly exp
  (~1e-6, byte-identical greedy output). Default OFF.

### Gains on N1 (the real, shippable result)
- **Decode speed +5.1%**: 18.66 -> 19.62 t/s. Ratio vs llama **0.85x -> 0.90x**.
- **Decode instructions -4.8%**: 70.32 B -> 66.92 B, ALL from matmul 61.58 -> 58.12 B (-5.6%,
  the MR=1 streaming). candle decode is now LEANER than llama (66.9 vs 77.2 B); the residual
  0.90x speed gap is IPC (2.17 vs 2.24) / cache, NOT instruction count.
- **Prefill: UNCHANGED** (0.72x; matmul instr 69.5 B invariant across all source variants -
  N1's LLVM already optimizes the original; the M1 +28% never transferred). See deep-dive above.

### Caveats / open items (don't forget)
- **vec-softmax targets the WRONG function for decode.** It edits `run_causal_attn_cpu`
  (prefill attention). Decode runs `causal_decode_f16kv_interleaved` - untouched. So vec-softmax
  helps PREFILL attention only; decode transcendental stayed flat (1.72->1.81 B). TODO: port the
  poly-exp to `causal_decode_f16kv_interleaved` for the decode transcendental win (~1.2 B vs llama).
- **Remaining decode gaps vs llama** (candle heavier): attention +2.5 B (4.64 vs 2.13 B - candle's
  decode flash kernel is heavier than ggml's tiled one) + transcendental +1.2 B. ~3.7 B / 5.5%,
  partly IPC-bound.
- **Prefill 1.5x gap (candle 69.5 B vs llama 47 B)**: NOT closable by source-level Rust on N1
  (compiler floor, proven by 3 invariant variants). Needs hand-written aarch64 `asm!` matching
  ggml's 8x4 (deliberate register allocation + instruction scheduling to delete the ~22 B of
  address/movi/spill overhead LLVM leaves), OR accept the gap.

### Feature flags / env knobs
- `CANDLE_VEC_SOFTMAX_EXP=1` - NEON poly softmax exp (prefill attn only, for now).
- `CANDLE_PACKED_PREFILL={nc4mr4(default),nc8mr2,nc8mr4}` - prefill tile; nc4mr4 best on N1
  (nc8mr4 = 32 acc spills, -20%).
- `CANDLE_FEATURES=f16-attn-dot` (build) - deploy "best candle"; `CANDLE_KV_PREALLOC=512`.

### Experimental design (so the comparison stays honest)
- Speed: `bench/compare.sh` (candle Q6packed vs llama Q4out, weight-matched, WITH warmup =
  steady-state, median statistic, full CSV provenance). Instr/cycles: `bench/instr_breakdown.sh`
  (perf, llama needs `--no-warmup` or 2x inflated; candle `--warmup 0 --reps 1`). Phase isolation:
  prefill=`pp512/tg1`, decode=`pp8/tg256`. ALWAYS confirm kernel opts on N1 (M1 != N1 - register-
  pressure-trading wins do NOT transfer; proven 3x this session).

## MULTI-THREAD SCALING - candle scales WORSE than llama (N1, 2026-06-23) - the deploy lever

The 1-vCPU work above is the wrong place to keep digging: Lambda grants 2-6 vCPUs at higher
memory tiers, and candle's RATIO vs llama DEGRADES with threads (bench/compare.sh CORES="1 2 4 6",
pp128/tg256):

| cores | prefill c/l | decode c/l | candle decode t/s | llama decode t/s |
|-------|-------------|------------|-------------------|------------------|
| 1     | 0.72x       | 0.90x      | 19.84             | 22.10            |
| 2     | 0.66x       | 0.71x      | 28.35             | 39.91            |
| 4     | 0.61x       | 0.62x      | 42.56             | 68.90            |
| 6     | 0.60x       | **0.51x**  | 47.18             | 91.84            |

- candle decode per-doubling: 1.43x (1->2), 1.50x (2->4), **1.11x (4->6)** - basically STOPS scaling
  past 4 cores. llama decode keeps scaling ~1.8x/doubling. Prefill scales OK (candle ~1.85x vs llama
  ~1.99x), so the ratio only drifts 0.72->0.60; DECODE is the collapse (0.90->0.51).

### Diagnosis: memory CONTENTION, not rayon overhead (candle decode, 1 vs 4 threads, clean)
Same workload (pp8/tg256), CANDLE_QMATMUL_*_THREADS=T + taskset -c 0..T-1:

| candle decode      | 1 thread | 4 threads | delta |
|--------------------|----------|-----------|-------|
| total instructions | 3.28e11  | 3.40e11   | +3.7% (rayon/coord overhead is SMALL) |
| total cycles       | 1.48e11  | 1.67e11   | +12.8% |
| IPC                | 2.21     | 2.04      | -8%   |
| backend-stall      | 37.8%    | 42.1%     | worse |
| L2d refills (far-mem) | 2.09e8 | **5.36e8** | **2.56x** |

The instruction count BARELY rises (+3.7%) -> the poor scaling is NOT coordination/rayon overhead.
It's MEMORY: far-memory traffic 2.56x and stalls rise -> the 4 cores thrash the shared system-level
cache / DRAM. Multi-threading pushes the bottleneck deeper into shared memory, and candle is hit
harder than llama (which scales ~1.8x). (Caveat: the llama 1-vs-4 perf comparison was contaminated -
the 1-thread llama run omitted `-t` so it used default threads - so the clean evidence is the scaling
CURVE, not the llama stall deltas.)

### Why candle likely contends more (hypothesis -> next investigation)
candle parallelizes each SMALL decode matmul (a 1-token GEMV) over cores via rayon (split the n/8
channel groups). That is FINE granularity: cores sync per-matmul (many per token x 256 tokens) and
their working sets interleave in the shared cache -> thrash. llama likely divides work more coarsely
(each thread owns a contiguous output range across the whole op), keeping per-core working sets
separate and reducing shared-cache pressure. CANDIDATE LEVER: coarsen candle's decode parallel grain
and/or make the channel-group split cache-aware (contiguous per core), to cut the 2.56x refill blowup.
This is likely TRACTABLE (a scheduling/partitioning change), unlike the 1-thread kernel floor.

## STALL ANALYSIS - the IPC gap localized (N1, 2026-06-22, bench/perf_stall.sh)

Measured backend/frontend stall % + L1/L2 refills to settle memory-bound vs dependency-bound:

| phase   | IPC  | backend-stall | frontend-stall | L2/L1 refill | verdict |
|---------|------|---------------|----------------|--------------|---------|
| decode  | 2.22 | **38.7%**     | 1.3%           | 23.0%        | **memory/bandwidth-bound** |
| prefill | 2.25 | 32.1%         | 2.0%           | 12.4%        | **compute/dependency-bound** |

- **Decode is memory-bound**: ~39% of cycles stalled in the backend with real L2 traffic (23% of
  L1 misses go to L2+). The CPU waits on streamed weights, not compute. => CUTTING INSTRUCTIONS DOES
  NOT SPEED UP DECODE. Confirmed by experiment (below). candle decode is already instr-leaner than
  llama; the 0.90x gap is the memory wall + llama's better prefetch/scheduling. Near the floor.
- **Prefill is compute/dependency-bound**: 32% backend stall but LOW L2/L1 (12.4%) - weights are
  amortized over 512 rows, so the stalls are SDOT/vmlaq/vaddvq dependency latency + the address/spill
  overhead, NOT memory. => the 22B instruction overhead IS the prefill lever, and hand-asm (delete
  overhead + schedule loads ahead of SDOTs to fill the stall slots) is the evidence-based fix.

### Decode opt experiments that BACKFIRED (gated OFF; kept for other HW / record)
Both confirm "decode is memory-bound, instr cuts don't help":
- **vec-softmax in decode** (CANDLE_VEC_SOFTMAX_EXP, online_softmax.rs): transcendental 1.81 -> 0.87 B
  (-0.94B, worked!) but decode SLOWER (20.04 -> slower) - the poly's 6-deep Horner FMA DEPENDENCY
  CHAIN has worse latency than libm expf despite fewer instructions. (Still helps PREFILL attn, where
  it's vectorized across the score vector - no serial chain.)
- **tiled decode attention** (CANDLE_TILED_DECODE_ATTN, causal.rs): attention 4.64 -> 5.70 B (+1.06B,
  BACKFIRED) + slower. The online-softmax acc-rescales were NOT the bottleneck (max stabilizes fast);
  the tile's score-buffer store/load + separate max-pass cost more than the rescale it saved.
- Both greedy-identical to default (temp 0); default (online + libm) is FASTEST on N1. Decode is at
  its practical floor. Net decode win this whole effort = the MR=1 matmul streaming (-3.46B, +5%),
  which helped because it cut LOADS (memory traffic), not just instructions.

### Order of attack (user, 2026-06-22)
1. Wider SDOT GEMM tile (8x4/8x8) for prefill - RULED OUT (see below; regresses 20% on N1).
2. Prefill matmul kernel micro-opt (address arithmetic + constant hoisting) - the REAL pp lever.
3. transcendental (expf/sincos) - knock out if easy (vectorized exp), helps BOTH phases.
4. attention (flash kernel) - after the above; fix classifier first.

## Prefill matmul deep-dive: WHERE the 1.5x cycles go (N1, 2026-06-22)

Levers RULED OUT on hardware: wider tile `nc8mr4` REGRESSES prefill 20% on N1 (32 acc spills
the 32 NEON regs; reverted default to `nc4mr4`). lane=channel layout = 0.57x on M1 (4x activation
bandwidth). So the gap is INTRINSIC to the kernel - localized below by `perf annotate` (cycles:u).

Per prefill pass (512 tokens, 1 vCPU N1), ABSOLUTE CYCLES:
- candle total prefill = 44.13 B cyc;  matmul (gemm_q4kx 56.4% + gemm_q6kx8 17.3%) = 32.5 B cyc.
- llama  total prefill = 29.55 B cyc;  matmul (gemm_q4_K 57.5% + gemm_q6_K 11.9%) = 20.5 B cyc.
- candle matmul / llama matmul = **1.59x cycles** (the gap; decode matmul is at parity).

Inside the Q4 kernel (candle `gemm_q4kx_q8k` = 24.88 B cyc/pass vs llama `ggml_gemm_q4_K_8x4_q8_K`
= 17.00 B cyc/pass), cycles by instruction (absolute B cyc, NOT percent):

| instruction        | candle B cyc | llama B cyc | candle EXCESS | what it is / fix |
|--------------------|-------------:|------------:|--------------:|------------------|
| `add`              | **4.07**     | 0.61        | **+3.46**     | address arithmetic - recomputes `base+(c*32)` each iter; FIX: walk pointers |
| `movi`             | **2.35**     | ~0.0        | **+2.35**     | re-materializes masks (0xF)/zero in-loop; FIX: hoist constants to regs |
| loads `ldp/ldr/ldur` | 4.47       | 1.82        | +2.65         | partly a symptom of the address recompute |
| `cmp`/`b.*` loop   | ~2.2         | low         | +~1.5         | nested NC/MR/chunk loops; FIX: restructure/unroll |
| `mla`/`smlal` scale| 2.72         | 2.21        | +0.51         | per-sub-block Q4_K scale - mostly inherent |
| `sdot` (real MACs) | 3.30         | 3.72        | -0.42         | candle already lean here |
| `ushr` nibble unpack| 0.83        | 1.88        | -1.05         | candle cheaper here |

candle Q4 kernel EXCESS = 24.88 - 17.00 = **7.88 B cyc**. **74% of it is just two fixable things:**
**address arithmetic (`add`, +3.46 B) + constant materialization (`movi`, +2.35 B) = +5.81 B.**
Neither needs an algorithm/layout change - both are codegen/loop-structure fixes in
`gemm_q4kx_q8k` (and the Q6 twin `gemm_q6kx8_q8k`, same profile). Closing them should bring the
Q4 kernel from 24.9 B -> ~17.5 B cyc, near llama's 17.0 B = prefill near-parity.

llama is the FLOOR: even its kernel is only 21.9% `sdot` - the rest is the irreducible Q4_K
unpack/scale. We are NOT going below ~17 B; the target is to delete candle's ~5.8 B of avoidable
address/constant overhead.

## How to track progress

After each candle tweak, re-run ONLY candle and diff the component column:
```
bash bench/instr_breakdown.sh candle 512 1   # prefill target = matmul 69.93B -> ?
bash bench/instr_breakdown.sh candle 8 256    # decode regression guard = total 70.32B
```
llama columns above are frozen - do not re-measure.
