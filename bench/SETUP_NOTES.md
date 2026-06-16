# EC2 parity-bench setup — choices & issues

Notes from standing up the EC2 Graviton2 box that runs `bench/compare.sh`
(candle vs llama.cpp parity). Captures the decisions made and the problems hit,
so the next run (and the next person) doesn't re-learn them.

## What we were doing

Running the head-to-head parity benchmark, which needs Linux `taskset` + a built
`llama.cpp` on real Graviton silicon — i.e. an EC2 box, not a Mac. `ec2_setup.sh`
provisions that box in one command; `compare.sh` runs the comparison.

## Choices

| Decision | Choice | Why |
|---|---|---|
| Instance | **c6g.2xlarge** (8 vCPU / 16 GB) | Graviton2 = dedicated Neoverse N1 cores, no SMT. 8 cores lets us sweep `CORES="2 4 8"`; headroom for builds. |
| AMI | **Amazon Linux 2023 arm64** | First-party arm64; `ec2_setup.sh` handles both `dnf` (AL2023) and `apt` (Ubuntu). |
| SSH key | **Imported `~/.ssh/id_rsa_jeshli.pub`** as EC2 key pair `ml-lambdas-bench` | The pre-existing `ml-experiments-key` had no private key on this Mac. Importing our own pubkey means we already hold the private key. |
| Security group | `sg-...` SSH (22) from the user's `/32` only | No reason to expose 22 to the world. |
| Code transfer | **`rsync` the working tree** (excl. `.git`, `target/`, model files) | Carries *uncommitted* bench-script edits a GitHub clone would miss; GGUF is fetched on the box, not uploaded. |
| Model | **Fetched on the box** via `fetch_model.sh` | EC2 download (136 MB/s) beats uploading 390 MB over home internet. |
| Docker locally | **Not needed** | Parity run is pure SSH + AWS CLI. The Lambda image (for the later cost/token sweep) can be built on the EC2 box itself — native arm64 — so local Colima/Docker is optional. |

## Issues we hit (and fixes — all folded into `ec2_setup.sh`)

1. **`dnf` curl conflict.** AL2023 ships `curl-minimal`, which *conflicts* with
   the full `curl` package — requesting `curl` aborts the whole install.
   → Dropped `curl` from the `dnf` list (the command is already present); kept it
   for `apt`, where it may be absent.

2. **`openssl-sys` build failure.** The candle example pulls in `openssl-sys`,
   which needs OpenSSL dev headers + `pkg-config`.
   → Added `openssl-devel` + `pkgconf-pkg-config` (dnf) / `libssl-dev` +
   `pkg-config` (apt).

3. **OOM during the llama.cpp build — the big one.** AL2023 has **no swap**, and
   `cmake --build -j8` spawns 8 `cc1plus` processes that together blow past 16 GB.
   The kernel OOM-killer fired, the build thrashed in a respawn loop, and **sshd
   became unreachable** (banner-exchange timeouts) — while AWS status checks
   stayed green and CPU sat at ~47% (memory-bound, not compute-bound). Diagnosed
   from the serial console (`cc1plus invoked oom-killer`). Required a forced
   stop/start to recover.
   → Two-part fix in `ec2_setup.sh`: (a) add an 8 G swapfile if none exists;
   (b) cap llama.cpp at `-j4` and build only the `--target llama-bench` we use.
   After the fix the build completed with **0 B swap used** — the `-j` cap alone
   kept peak memory sane; swap is the belt-and-suspenders.

4. **Statistic mismatch in `compare.sh`** (caught in PR review). The candle side
   reported the median (`*_median`) but the llama.cpp side read llama-bench's
   *mean* `avg_ts`, so one noisy rep skewed only the ratio's denominator.
   → Now takes the **median of llama-bench's `samples_ts`** so both sides use the
   same statistic.

## Results (c6g.2xlarge, Qwen3-0.6B Q4_K_M, pp=512 tg=128, 5 reps, median)

```
cores  engine     pp_t/s    tg_t/s    peak_MB
2      candle      40.93     16.34     605
2      llama.cpp   87.45     35.09     885
2      ratio c/l   0.47x     0.47x
4      candle      74.48     28.16     607
4      llama.cpp  173.96     62.76     885
4      ratio c/l   0.43x     0.45x
8      candle     120.83     38.63     606
8      llama.cpp  327.01     72.72     885
8      ratio c/l   0.37x     0.53x
```

Read: candle is currently **~0.37–0.47× of llama.cpp on prefill** and
**~0.45–0.53× on decode** — i.e. llama.cpp is roughly 2× faster on this box, and
candle's prefill gap *widens* as cores scale (0.47×→0.37×), suggesting candle's
prefill parallelism scales worse than llama.cpp's here. candle does use notably
less peak RSS (~605 MB vs ~885 MB). This is the relative iteration signal; the
cost/token verdict still comes from the Lambda sweep (layer 2).

## Cost handling

- The box is **stopped** (not terminated) between sessions — keeps the ~15-min
  provisioning for only ~$2.40/mo of EBS; compute billing only during actual runs.
- The build swapfile is **removed** after runs (`swapoff` + `rm`); `ec2_setup.sh`
  recreates it idempotently next time.

## Open items

- Commit the `compare.sh` median fix + `ec2_setup.sh` fixes to the `compare`
  branch / PR so they're durable (currently rsync'd to the box + in the working
  tree).
- For the Lambda cost/token sweep: the box needs Docker installed and an IAM
  instance profile with ECR push perms (it has no role today).
