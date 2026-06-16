#!/usr/bin/env bash
# One-shot provisioning for the EC2 Graviton2 parity box.
#
# Takes a fresh arm64 EC2 instance (Amazon Linux 2023 or Ubuntu) to "one command
# from a compare.sh result": installs build deps + Rust, clones the candle fork
# at the tuned branch, clones and builds llama.cpp, and verifies the GGUF is in
# place. Idempotent — re-run it to pull/rebuild after a candle change.
#
# It does NOT run the benchmark; it prints the exact compare.sh invocation to run
# next (with the dirs it just created already filled in).
#
# Usage (from the repo root on the EC2 box):
#   ./bench/ec2_setup.sh
#
# Override any of these via env:
#   CANDLE_DIR     where to clone candle           (default ~/candle)
#   CANDLE_REPO    candle fork URL                 (default the QK_4_GEMV fork)
#   CANDLE_BRANCH  candle branch                   (default QK_4_GEMV)
#   LLAMA_DIR      where to clone llama.cpp        (default ~/llama.cpp)
#   LLAMA_REPO     llama.cpp URL                   (default upstream)
#   SKIP_DEPS=1    skip the package-manager step (already provisioned)
set -euo pipefail

CANDLE_DIR="${CANDLE_DIR:-$HOME/candle}"
CANDLE_REPO="${CANDLE_REPO:-https://github.com/DrJesseGlass/candle}"
CANDLE_BRANCH="${CANDLE_BRANCH:-QK_4_GEMV}"
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp}"

# Repo root = parent of this script's dir, so the model path resolves no matter
# where the checkout lives.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="${MODEL:-$REPO_ROOT/qwen-lambda/models/Qwen3-0.6B-Q4_K_M.gguf}"

say() { printf '\n=== %s ===\n' "$*"; }

# --- 1. system deps -------------------------------------------------------
if [ "${SKIP_DEPS:-0}" != "1" ]; then
  say "Installing build deps"
  if command -v dnf >/dev/null 2>&1; then
    # NB: don't request `curl` here — AL2023 ships `curl-minimal`, which provides
    # the curl command and *conflicts* with the full `curl` package (dnf aborts).
    # openssl-devel + pkgconf: the candle example's openssl-sys dep needs them.
    sudo dnf install -y git cmake gcc gcc-c++ make jq bc util-linux \
                        openssl-devel pkgconf-pkg-config
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y git cmake build-essential jq bc curl util-linux \
                            libssl-dev pkg-config
  else
    echo "No supported package manager (dnf/apt-get) found; install git cmake" \
         "a C/C++ toolchain, jq, bc, curl, and taskset manually." >&2
    exit 1
  fi
else
  say "SKIP_DEPS=1 — skipping package install"
fi

# taskset (util-linux) is what makes the core-pinning in compare.sh work.
command -v taskset >/dev/null 2>&1 || {
  echo "taskset not found after install — compare.sh needs it." >&2; exit 1; }
# curl is needed for rustup + fetch_model.sh (preinstalled on AL2023, may not be
# on a minimal Ubuntu — apt installs it above).
command -v curl >/dev/null 2>&1 || {
  echo "curl not found — needed for rustup and model fetch." >&2; exit 1; }

# --- 1b. swap safety net --------------------------------------------------
# AL2023 ships with NO swap. The parallel C++ build below peaks hard (each
# cc1plus can want ~2 GB); on a 15 GB box -j8 OOM-kills the compilers and
# thrashes sshd to death. A swapfile turns a hard OOM into (worst case) slow
# progress. Idempotent — skip if any swap is already on.
if [ "$(swapon --show=NAME --noheadings | wc -l)" -eq 0 ]; then
  say "No swap present — adding an 8G swapfile"
  sudo dd if=/dev/zero of=/swapfile bs=1M count=8192 status=none
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile
fi

# --- 2. Rust toolchain ----------------------------------------------------
if ! command -v cargo >/dev/null 2>&1; then
  say "Installing Rust (rustup)"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
# shellcheck disable=SC1091
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
say "Rust: $(cargo --version)"

# --- 3. candle fork -------------------------------------------------------
if [ -d "$CANDLE_DIR/.git" ]; then
  say "Updating candle ($CANDLE_DIR @ $CANDLE_BRANCH)"
  git -C "$CANDLE_DIR" fetch origin "$CANDLE_BRANCH"
  git -C "$CANDLE_DIR" checkout "$CANDLE_BRANCH"
  git -C "$CANDLE_DIR" pull --ff-only origin "$CANDLE_BRANCH"
else
  say "Cloning candle ($CANDLE_BRANCH) into $CANDLE_DIR"
  git clone --branch "$CANDLE_BRANCH" "$CANDLE_REPO" "$CANDLE_DIR"
fi

# Pre-build the bench so the first compare.sh run is warm (compare.sh rebuilds
# anyway; this just front-loads the cost and surfaces build errors here).
say "Building candle quantized-qwen3-bench (release)"
( cd "$CANDLE_DIR" && RUSTFLAGS="-C target-cpu=native" \
    cargo build --release --example quantized-qwen3-bench )

# --- 4. llama.cpp ---------------------------------------------------------
if [ -d "$LLAMA_DIR/.git" ]; then
  say "Updating llama.cpp ($LLAMA_DIR)"
  git -C "$LLAMA_DIR" pull --ff-only
else
  say "Cloning llama.cpp into $LLAMA_DIR"
  git clone "$LLAMA_REPO" "$LLAMA_DIR"
fi

say "Building llama.cpp (Release, llama-bench target)"
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release
# Cap parallelism (each cc1plus can need ~2 GB) and build ONLY the target we use.
# Default `-j` == all cores, which is what OOM-bombed the 15 GB box; 4 keeps peak
# memory sane while staying fast. Override with LLAMA_JOBS on a bigger box.
LLAMA_JOBS="${LLAMA_JOBS:-$(nproc)}"
[ "$LLAMA_JOBS" -gt 4 ] && LLAMA_JOBS=4
cmake --build "$LLAMA_DIR/build" -j "$LLAMA_JOBS" --target llama-bench

LLAMA_BENCH="$LLAMA_DIR/build/bin/llama-bench"
[ -x "$LLAMA_BENCH" ] || {
  echo "llama-bench not found at $LLAMA_BENCH after build." >&2; exit 1; }

# --- 5. model -------------------------------------------------------------
if [ -f "$MODEL" ]; then
  say "Model present: $MODEL"
else
  say "Model missing — fetching"
  "$REPO_ROOT/qwen-lambda/fetch_model.sh"
fi

# --- done -----------------------------------------------------------------
say "Setup complete — run the parity comparison with:"
cat <<EOF

  MODEL=$MODEL \\
  CANDLE_DIR=$CANDLE_DIR \\
  LLAMA_BENCH=$LLAMA_BENCH \\
  CORES="2 4" $SCRIPT_DIR/compare.sh

EOF
