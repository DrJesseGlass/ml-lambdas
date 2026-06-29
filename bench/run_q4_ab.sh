#!/usr/bin/env bash
# ONE-SHOT N1 runner for the Q4_K lane=row A/B (ours vs upstream #3643), from the Mac.
#
# Resumes the stopped Graviton2 box, syncs the local candle checkout (current branch
# = cpu-optimized/q6k-packed, INCLUDING the uncommitted quantized-qwen3-bench example)
# + bench scripts, builds the bench + generation examples natively (so dotprod/lane=row
# is compiled in), runs:
#   1. CORRECTNESS - greedy generation under CANDLE_PREFILL_LANEROW=0 (#3643) vs =1
#      (ours); the text must match (lane=row prefill agrees with #3643 on real weights).
#   2. SPEED - lanerow_ab.sh: prefill t/s, ours vs #3643, swept over threads x pp.
# Fetches results and ALWAYS stops the box on exit (EXIT trap - survives Ctrl-C/failure).
#
# Robust to SSH drops: the box runs build+benches under nohup and touches a DONE
# sentinel; this script polls for it rather than holding a live SSH.
#
# Prereqs: aws cli configured, ~/.ssh/id_rsa_jeshli present. Box already has ~/candle,
# ~/ml-lambdas, the Q4_K_M GGUF (else fetched via qwen-lambda/fetch_model.sh).
#
# Usage:
#   ./bench/run_q4_ab.sh                                  # defaults
#   PP="512 2048" THREADS="1 2 4 6" REPS=3 ./bench/run_q4_ab.sh
#   KEEP_UP=1 ./bench/run_q4_ab.sh                        # leave box up afterwards
set -euo pipefail

INSTANCE=${INSTANCE:-i-05bea320332bed88b}
REGION=${REGION:-us-east-1}
SG=${SG:-sg-00589e993f0f63172}
KEY=${KEY:-$HOME/.ssh/id_rsa_jeshli}
SSH_USER=${SSH_USER:-ec2-user}
CANDLE_DIR=${CANDLE_DIR:-/Users/jesseglass/Documents/jeshli/solo/candle}
BENCH_DIR=${BENCH_DIR:-/Users/jesseglass/Documents/jeshli/solo/ml-lambdas/bench}
PP=${PP:-"512 2048"}
THREADS=${THREADS:-"1 2 4 6"}
REPS=${REPS:-3}
KEEP_UP=${KEEP_UP:-0}
MAXWAIT=${MAXWAIT:-3600}

say() { printf '\n=== %s ===\n' "$*"; }
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=30)

command -v aws >/dev/null || { echo "aws cli not found"; exit 1; }
[ -f "$KEY" ] || { echo "ssh key $KEY not found"; exit 1; }

# ---- always stop the box on exit (unless KEEP_UP=1) -------------------------
stop_box() {
  if [ "$KEEP_UP" = 1 ]; then
    echo "KEEP_UP=1 - leaving box running (stop it: aws ec2 stop-instances --instance-ids $INSTANCE --region $REGION)"
    return
  fi
  say "stopping box (cost control)"
  aws ec2 stop-instances --instance-ids "$INSTANCE" --region "$REGION" \
    --query 'StoppingInstances[0].CurrentState.Name' --output text 2>&1 || true
}
trap stop_box EXIT

# ---- start + wait running ---------------------------------------------------
say "starting $INSTANCE"
aws ec2 start-instances --instance-ids "$INSTANCE" --region "$REGION" \
  --query 'StartingInstances[0].CurrentState.Name' --output text
aws ec2 wait instance-running --instance-ids "$INSTANCE" --region "$REGION"
DNS=$(aws ec2 describe-instances --instance-ids "$INSTANCE" --region "$REGION" \
  --query 'Reservations[0].Instances[0].PublicDnsName' --output text)
echo "public DNS: $DNS"

# ---- ensure SG ingress for our current IP -----------------------------------
MYIP=$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')
if ! aws ec2 describe-security-groups --group-ids "$SG" --region "$REGION" \
      --query 'SecurityGroups[0].IpPermissions[?FromPort==`22`].IpRanges[].CidrIp' \
      --output text | tr '\t' '\n' | grep -qx "$MYIP/32"; then
  echo "adding SSH ingress for $MYIP/32"
  aws ec2 authorize-security-group-ingress --group-id "$SG" --region "$REGION" \
    --protocol tcp --port 22 --cidr "$MYIP/32" >/dev/null 2>&1 || true
fi

# ---- wait for sshd ----------------------------------------------------------
say "waiting for sshd"
for i in $(seq 1 30); do
  if "${SSH[@]}" "$SSH_USER@$DNS" "echo ok" 2>/dev/null | grep -q ok; then
    echo "ssh up"; break
  fi
  sleep 6
  [ "$i" = 30 ] && { echo "sshd never came up"; exit 1; }
done

# ---- sync candle source (incl uncommitted bench example) + bench scripts -----
say "syncing candle (excl target) + bench scripts"
rsync -az --delete --exclude '/target/' --exclude '*.gguf' \
  -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$CANDLE_DIR/" "$SSH_USER@$DNS:~/candle/"
rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$BENCH_DIR/" "$SSH_USER@$DNS:~/ml-lambdas/bench/"

# ---- box-side runner: build + correctness + speed + DONE sentinel -----------
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RES="q4ab_results_$STAMP.txt"
DONE="q4ab_done_$STAMP"
say "launching box-side build + Q4 A/B (nohup) -> ~/$RES"
"${SSH[@]}" "$SSH_USER@$DNS" "cat > ~/q4ab_box.sh" <<BOX
#!/usr/bin/env bash
set -uo pipefail
RES=~/$RES; DONE=~/$DONE
: > "\$RES"
log(){ echo "\$@" | tee -a "\$RES"; }
CD=~/candle
G=~/ml-lambdas/qwen-lambda/models/Qwen3-0.6B-Q4_K_M.gguf
THREADS="$THREADS"; PP="$PP"; REPS="$REPS"

log "## Q4_K lane=row A/B (ours vs upstream #3643)  \$(date -u)  host=\$(hostname)  \$(nproc) vCPU"
log "   branch: \$(cd \$CD && git rev-parse --abbrev-ref HEAD 2>/dev/null) @ \$(cd \$CD && git rev-parse --short HEAD 2>/dev/null)"

# model present? else fetch
if [ ! -f "\$G" ]; then
  log "## model missing, fetching ..."
  ( bash ~/ml-lambdas/qwen-lambda/fetch_model.sh ) >>"\$RES" 2>&1 || log "(fetch_model.sh errored)"
fi
[ -f "\$G" ] || { log "NO MODEL at \$G - abort"; touch "\$DONE"; exit 1; }

# build native (dotprod -> lane=row kernel compiled in)
log "## building quantized-qwen3-bench + quantized-qwen3 (native) ..."
if ! ( cd "\$CD" && RUSTFLAGS="-C target-cpu=native" cargo build --release \
       --example quantized-qwen3-bench --example quantized-qwen3 >/tmp/build.log 2>&1 ); then
  log "BUILD FAILED:"; tail -40 /tmp/build.log | tee -a "\$RES"; touch "\$DONE"; exit 1
fi
B=\$CD/target/release/examples/quantized-qwen3-bench
QQ=\$CD/target/release/examples/quantized-qwen3

# 1. CORRECTNESS: greedy text must match between #3643(=0) and ours(=1)
log ""
log "## CORRECTNESS: greedy gen (temp 0), #3643(=0) vs ours(=1), same prompt"
PROMPT="The history of computing began with"
for lr in 0 1; do
  env CANDLE_PREFILL_LANEROW=\$lr CANDLE_NUM_THREADS=4 RAYON_NUM_THREADS=4 \
    "\$QQ" --model "\$G" --prompt "\$PROMPT" --sample-len 48 --temperature 0 \
    > /tmp/gen_\$lr.txt 2>/dev/null || log "(gen lr=\$lr errored)"
done
if diff -q /tmp/gen_0.txt /tmp/gen_1.txt >/dev/null 2>&1; then
  log "  IDENTICAL greedy output -> lane=row prefill matches #3643 on real weights. OK"
else
  log "  DIFFERS (f32 reassoc can flip late tokens; inspect):"
  diff /tmp/gen_0.txt /tmp/gen_1.txt | head -20 | tee -a "\$RES"
fi

# 2. SPEED: prefill t/s, ours vs #3643, swept over threads x pp
log ""
log "## SPEED A/B (lanerow_ab.sh)"
B="\$B" G="\$G" THREADS="\$THREADS" PP="\$PP" REPS="\$REPS" \
  bash ~/ml-lambdas/bench/lanerow_ab.sh 2>&1 | tee -a "\$RES" || log "(lanerow_ab.sh errored)"

log ""
log "## DONE  \$(date -u)"
touch "\$DONE"
BOX
"${SSH[@]}" "$SSH_USER@$DNS" "chmod +x ~/q4ab_box.sh && nohup bash ~/q4ab_box.sh >/dev/null 2>&1 & echo launched PID \$!"

# ---- poll for the DONE sentinel (survives SSH blips) ------------------------
say "waiting for box run to finish (polling ~/$DONE)"
waited=0; step=20
while ! "${SSH[@]}" "$SSH_USER@$DNS" "test -f ~/$DONE" 2>/dev/null; do
  sleep "$step"; waited=$((waited+step))
  printf '  ... %ds (tail: %s)\n' "$waited" \
    "$("${SSH[@]}" "$SSH_USER@$DNS" "tail -1 ~/$RES 2>/dev/null" 2>/dev/null || echo '?')"
  [ "$waited" -ge "$MAXWAIT" ] && { echo "timed out after ${MAXWAIT}s"; break; }
done

# ---- fetch + print results --------------------------------------------------
say "RESULTS"
"${SSH[@]}" "$SSH_USER@$DNS" "cat ~/$RES" 2>/dev/null || echo "(could not fetch results)"
mkdir -p "$BENCH_DIR/n1_runs"
scp -i "$KEY" -o StrictHostKeyChecking=no "$SSH_USER@$DNS:~/$RES" \
  "$BENCH_DIR/n1_runs/$RES" 2>/dev/null && echo "saved -> bench/n1_runs/$RES" || true

# stop_box runs via the EXIT trap
