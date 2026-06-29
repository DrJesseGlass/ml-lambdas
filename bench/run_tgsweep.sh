#!/usr/bin/env bash
# ONE-SHOT N1 HEAD-TO-HEAD runner (from the Mac). Resumes the Graviton2 box, syncs
# the local candle checkout INCLUDING .git (so the box can checkout BOTH
# explore/rayon-trim-q6k-packing and lambda-optimized/model-core), runs
# bench/tgsweep_n1.sh box-side (two builds, two bakes, benches explore vs
# model-core vs llama over the thread tiers), fetches results, and ALWAYS stops
# the box on exit (cost control - EXIT trap).
#
# Box-side does TWO cold candle builds + two bakes + two full bench sweeps, so
# MAXWAIT is generous. Robust to SSH drops: box runs under nohup + a DONE sentinel.
#
# Usage:
#   ./bench/run_tgsweep.sh
#   THREADS="1 2 4 6" PP=512 TG=256 REPS=3 ./bench/run_tgsweep.sh
#   KEEP_UP=1 ./bench/run_tgsweep.sh
set -euo pipefail

INSTANCE=${INSTANCE:-i-05bea320332bed88b}
REGION=${REGION:-us-east-1}
SG=${SG:-sg-00589e993f0f63172}
KEY=${KEY:-$HOME/.ssh/id_rsa_jeshli}
SSH_USER=${SSH_USER:-ec2-user}
CANDLE_DIR=${CANDLE_DIR:-/Users/jesseglass/Documents/jeshli/solo/candle}
BENCH_DIR=${BENCH_DIR:-/Users/jesseglass/Documents/jeshli/solo/ml-lambdas/bench}
TGS=${TGS:-"128 256 512 1024"}
T=${T:-6}
PP=${PP:-512}
REPS=${REPS:-3}
KEEP_UP=${KEEP_UP:-0}
MAXWAIT=${MAXWAIT:-7200}

say() { printf '\n=== %s ===\n' "$*"; }
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=30)

command -v aws >/dev/null || { echo "aws cli not found"; exit 1; }
[ -f "$KEY" ] || { echo "ssh key $KEY not found"; exit 1; }

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

say "starting $INSTANCE"
aws ec2 start-instances --instance-ids "$INSTANCE" --region "$REGION" \
  --query 'StartingInstances[0].CurrentState.Name' --output text
aws ec2 wait instance-running --instance-ids "$INSTANCE" --region "$REGION"
DNS=$(aws ec2 describe-instances --instance-ids "$INSTANCE" --region "$REGION" \
  --query 'Reservations[0].Instances[0].PublicDnsName' --output text)
echo "public DNS: $DNS"

MYIP=$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')
if ! aws ec2 describe-security-groups --group-ids "$SG" --region "$REGION" \
      --query 'SecurityGroups[0].IpPermissions[?FromPort==`22`].IpRanges[].CidrIp' \
      --output text | tr '\t' '\n' | grep -qx "$MYIP/32"; then
  echo "adding SSH ingress for $MYIP/32"
  aws ec2 authorize-security-group-ingress --group-id "$SG" --region "$REGION" \
    --protocol tcp --port 22 --cidr "$MYIP/32" >/dev/null 2>&1 || true
fi

say "waiting for sshd"
for i in $(seq 1 30); do
  if "${SSH[@]}" "$SSH_USER@$DNS" "echo ok" 2>/dev/null | grep -q ok; then echo "ssh up"; break; fi
  sleep 6
  [ "$i" = 30 ] && { echo "sshd never came up"; exit 1; }
done

# sync candle source (incl .git so the box can checkout BOTH branches; excl target
# + *.gguf - the box bakes both Q6packed ggufs itself) + bench scripts
say "syncing candle (excl target/gguf, incl .git) + bench scripts"
rsync -az --delete --exclude '/target/' --exclude '*.gguf' \
  -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$CANDLE_DIR/" "$SSH_USER@$DNS:~/candle/"
rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$BENCH_DIR/" "$SSH_USER@$DNS:~/ml-lambdas/bench/"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RES="tgsweep_results_$STAMP.txt"
DONE="tgsweep_done_$STAMP"
say "launching box-side tg-sweep bench (nohup) -> ~/$RES"
"${SSH[@]}" "$SSH_USER@$DNS" "cat > ~/tgsweep_box.sh" <<BOX
#!/usr/bin/env bash
set -uo pipefail
RES=~/$RES; DONE=~/$DONE
: > "\$RES"
TGS="$TGS" T="$T" PP="$PP" REPS="$REPS" \
  bash ~/ml-lambdas/bench/tgsweep_n1.sh > "\$RES" 2>&1 || echo "(tgsweep_n1.sh errored)" >> "\$RES"
touch "\$DONE"
BOX
"${SSH[@]}" "$SSH_USER@$DNS" "chmod +x ~/tgsweep_box.sh && nohup bash ~/tgsweep_box.sh >/dev/null 2>&1 & echo launched PID \$!"

say "waiting for box run to finish (polling ~/$DONE, up to ${MAXWAIT}s)"
waited=0; step=30
while ! "${SSH[@]}" "$SSH_USER@$DNS" "test -f ~/$DONE" 2>/dev/null; do
  sleep "$step"; waited=$((waited+step))
  printf '  ... %ds (tail: %s)\n' "$waited" \
    "$("${SSH[@]}" "$SSH_USER@$DNS" "tail -1 ~/$RES 2>/dev/null" 2>/dev/null || echo '?')"
  [ "$waited" -ge "$MAXWAIT" ] && { echo "timed out after ${MAXWAIT}s"; break; }
done

say "RESULTS"
"${SSH[@]}" "$SSH_USER@$DNS" "cat ~/$RES" 2>/dev/null || echo "(could not fetch results)"
mkdir -p "$BENCH_DIR/n1_runs"
scp -i "$KEY" -o StrictHostKeyChecking=no "$SSH_USER@$DNS:~/$RES" \
  "$BENCH_DIR/n1_runs/$RES" 2>/dev/null && echo "saved -> bench/n1_runs/$RES" || true
# stop_box runs via the EXIT trap
