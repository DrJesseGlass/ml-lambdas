#!/usr/bin/env bash
# ONE-SHOT N1 benchmark runner (runs from the Mac).
#
# Resumes the stopped Graviton2 box, syncs the local candle checkout + bench
# scripts, builds the deploy bench binary, runs the lane=row A/B + the multi-thread
# prefill stall profile, fetches results, and ALWAYS stops the box on exit (cost
# control - an EXIT trap, so even Ctrl-C / a failure stops it).
#
# Robust to SSH drops: the box does build+benches under nohup and touches a DONE
# sentinel; this script polls for the sentinel rather than holding a live SSH.
#
# Prereqs: aws cli configured, ~/.ssh/id_rsa_jeshli present. The box already has
# ~/candle, ~/ml-lambdas, the GGUFs, and a built llama.cpp (provisioned earlier).
#
# Usage:
#   ./bench/run_n1.sh                       # defaults
#   PP="512 2048" THREADS="1 2 4" ./bench/run_n1.sh
#   KEEP_UP=1 ./bench/run_n1.sh             # leave the box running afterwards
set -euo pipefail

INSTANCE=${INSTANCE:-i-05bea320332bed88b}
REGION=${REGION:-us-east-1}
SG=${SG:-sg-00589e993f0f63172}
KEY=${KEY:-$HOME/.ssh/id_rsa_jeshli}
SSH_USER=${SSH_USER:-ec2-user}
CANDLE_DIR=${CANDLE_DIR:-/Users/jesseglass/Documents/jeshli/solo/candle}
BENCH_DIR=${BENCH_DIR:-/Users/jesseglass/Documents/jeshli/solo/ml-lambdas/bench}
# Bench knobs (passed through to the box-side scripts).
PP=${PP:-"512 2048"}
THREADS=${THREADS:-"1 2 4"}
REPS=${REPS:-3}
KEEP_UP=${KEEP_UP:-0}

say() { printf '\n=== %s ===\n' "$*"; }
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=30)

command -v aws >/dev/null || { echo "aws cli not found"; exit 1; }
[ -f "$KEY" ] || { echo "ssh key $KEY not found"; exit 1; }

# ---- always stop the box on exit (unless KEEP_UP=1) -------------------------
stop_box() {
  if [ "$KEEP_UP" = 1 ]; then
    echo "KEEP_UP=1 - leaving box running (remember to stop it: aws ec2 stop-instances --instance-ids $INSTANCE --region $REGION)"
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

# ---- sync candle source + bench scripts -------------------------------------
say "syncing candle (excl target) + bench scripts"
rsync -az --delete --exclude '/target/' --exclude '*.gguf' \
  -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$CANDLE_DIR/" "$SSH_USER@$DNS:~/candle/"
rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$BENCH_DIR/" "$SSH_USER@$DNS:~/ml-lambdas/bench/"

# ---- box-side runner: build + both benches + DONE sentinel ------------------
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RES="n1_results_$STAMP.txt"
DONE="n1_done_$STAMP"
say "launching box-side build + benches (nohup) -> ~/$RES"
"${SSH[@]}" "$SSH_USER@$DNS" "cat > ~/n1_box_run.sh" <<BOX
#!/usr/bin/env bash
set -uo pipefail
RES=~/$RES; DONE=~/$DONE
: > "\$RES"
log() { echo "\$@" | tee -a "\$RES"; }
log "## RE-BASELINE: new main vs our explore vs llama (6-vCPU)  \$(date -u +%H:%M:%SZ)"
log "   (rebaseline.sh does its own multi-config builds: newmain-default, explore-fullstack, llama)"
THREADS="4 6" PP=512 TG=128 REPS=3 \
  bash ~/ml-lambdas/bench/rebaseline.sh >>"\$RES" 2>&1 || log "(rebaseline.sh errored)"
log ""
log "## DONE  \$(date -u +%H:%M:%SZ)"
touch "\$DONE"
BOX
"${SSH[@]}" "$SSH_USER@$DNS" "chmod +x ~/n1_box_run.sh && nohup bash ~/n1_box_run.sh >/dev/null 2>&1 & echo launched PID \$!"

# ---- poll for the DONE sentinel (survives SSH blips) ------------------------
say "waiting for box run to finish (polling ~/$DONE)"
MAXWAIT=${MAXWAIT:-3600}; waited=0; step=20
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
