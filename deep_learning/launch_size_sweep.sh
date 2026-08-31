#!/bin/bash
# Runs LOCALLY. Launches the capacity size-down sweep: arm A on the existing GPU box, arms
# B/C/D on three fresh g5.2xlarge that bootstrap from the S3 env bundle and self-terminate.
#
# Prerequisite: make_sweep_bundle.sh has published miniconda3.tar + mlb.tar + s5cmd to
# s3://$BUCKET/deep_learning/sweep_bootstrap/. Without it the new boxes have no torch.
#
# THE LADDER (see ec2_size_sweep_arm.sh for what is held constant and why):
#   A  d_model 384  n_layers 6   <- identical to the A/B control architecture
#   B  d_model 256  n_layers 4
#   C  d_model 192  n_layers 3
#   D  d_model 128  n_layers 2
#
# WHY ARM A REUSES THE EXISTING BOX: it already holds the verified tensors on its instance
# store, so it needs no 27 GB pull, and reusing it keeps the sweep at three NEW instances
# (~$18 for ~5h) instead of four. It runs with SHUTDOWN=0 because that box is the project's
# working GPU box and a STOP would wipe /mnt/fast -- see [[reference-ec2]].
#
# WHY B/C/D SELF-TERMINATE: they are disposable. Every artifact is shipped to S3 by the arm
# script's EXIT trap before shutdown, so nothing on the box needs to outlive it, and a
# forgotten g5.2xlarge costs $1.21/h indefinitely.
#
# Usage:  bash deep_learning/launch_size_sweep.sh [sweep_id]
#         ARMS=C bash deep_learning/launch_size_sweep.sh 20260831   # relaunch one arm
#
# ARMS exists because arms fail independently. On 2026-08-31 arm C aborted at epoch-1
# validation with `LLVM ERROR: pthread_join failed` in all four DataLoader workers (SIGABRT,
# not SIGKILL -- so not an OOM) while A, B and D passed the same point cleanly and went on to
# epoch 2. Re-running the whole sweep to refill one rung wastes ~13 GPU-hours, and re-running
# arm A in particular would abort anyway on the trainer-already-running guard below.
set -uo pipefail
ARMS=${ARMS:-A B C D}
want() { case " $ARMS " in *" $1 "*) return 0;; *) return 1;; esac; }

BUCKET=${BUCKET:-mlb-265753586044-us-east-1-an}
SWEEP_ID=${1:-$(date -u +%Y%m%d)}
KEY=${KEY:-awstest}
PEM=${PEM:-/Users/michaelharoon/Documents/SENSITIVE/awstest.pem}
EXISTING_IP=${EXISTING_IP:-32.197.253.24}

# Same base AMI the existing GPU box runs, so the bundled env relocates onto an identical OS
# and driver stack: Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2023) 20260825.
AMI=${AMI:-ami-0d3378afe7683c867}
ITYPE=${ITYPE:-g5.2xlarge}
SG=${SG:-sg-0583a4de608a95a41}
SUBNET=${SUBNET:-subnet-013362590293de96a}
IAM=${IAM:-read-write-mlb-s3}

BOOT="s3://$BUCKET/deep_learning/sweep_bootstrap"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -i $PEM"

fail() { echo "ABORT: $*" >&2; exit 1; }

# --- preflight: the bundle must exist, or three boxes boot into nothing -----
for f in miniconda3.tar mlb.tar; do
  aws s3api head-object --bucket "$BUCKET" --key "deep_learning/sweep_bootstrap/$f" \
    >/dev/null 2>&1 || fail "missing $BOOT/$f — run make_sweep_bundle.sh on the GPU box first"
done
echo "bundle present; sweep_id=$SWEEP_ID"

# Publish the arm script the launcher is ACTUALLY running with, and have the boxes fetch it
# from here rather than out of mlb.tar.
#
# WHY: mlb.tar is a snapshot of the GPU box's rsync'd repo copy, which lags whatever is in the
# local working tree. On the first launch (2026-08-31) it predated ec2_size_sweep_arm.sh
# entirely, so user-data's `cp` from the extracted repo silently found nothing, sudo ran a
# non-existent path, and all three boxes sat idle at $1.21/h with a mounted instance store and
# no log to explain it. Shipping the script separately makes the launcher and the arms
# version-locked to each other by construction.
aws s3 cp "$(dirname "$0")/ec2_size_sweep_arm.sh" "$BOOT/ec2_size_sweep_arm.sh" >/dev/null \
  || fail "could not publish arm script to $BOOT"
echo "published arm script to $BOOT/ec2_size_sweep_arm.sh"

# --- arm A on the existing box ---------------------------------------------
if want A; then
echo "=== arm A -> existing box $EXISTING_IP ==="
$SSH "ec2-user@$EXISTING_IP" "pgrep -f 'train_unifie[d]' >/dev/null && echo BUSY" \
  | grep -q BUSY && fail "existing box already has a trainer running"
scp -o StrictHostKeyChecking=no -i "$PEM" \
  "$(dirname "$0")/ec2_size_sweep_arm.sh" "ec2-user@$EXISTING_IP:/home/ec2-user/" \
  || fail "could not ship arm script to $EXISTING_IP"
$SSH "ec2-user@$EXISTING_IP" \
  "ARM=A D_MODEL=384 N_LAYERS=6 SWEEP_ID=$SWEEP_ID SHUTDOWN=0 \
   nohup bash /home/ec2-user/ec2_size_sweep_arm.sh >/dev/null 2>&1 & sleep 3; echo launched" \
  || fail "arm A launch failed"
fi

# --- arms B/C/D on fresh boxes --------------------------------------------
launch_arm() {
  local arm=$1 dmodel=$2 nlayers=$3
  local ud
  ud=$(cat <<USERDATA
#!/bin/bash
exec > >(tee -a /var/log/sweep_bootstrap.log) 2>&1
set -x
echo "=== bootstrap arm $arm \$(date -u) ==="

# The instance store is REQUIRED, not an optimisation. Reading prepared tensors off EBS left
# the GPU idle at ~24,000 s/epoch on this exact model; on NVMe it is ~24 min. Identify the
# device by MODEL rather than by name -- nvme numbering is not stable across instance types,
# and mkfs on the wrong node would destroy the root volume.
DEV=\$(lsblk -dno NAME,MODEL | awk '/Instance Storage/ {print "/dev/"\$1; exit}')
if [ -z "\$DEV" ]; then echo "FATAL: no instance store found"; exit 1; fi
mkfs -t ext4 -F "\$DEV"
mkdir -p /mnt/fast
mount "\$DEV" /mnt/fast
chown ec2-user:ec2-user /mnt/fast

aws s3 cp $BOOT/s5cmd /usr/local/bin/s5cmd && chmod +x /usr/local/bin/s5cmd

# Env and repo relocate to the same absolute paths they were tarred from, which is what makes
# a plain tar sufficient in place of conda-pack.
cd /home/ec2-user
aws s3 cp $BOOT/miniconda3.tar /mnt/fast/miniconda3.tar
aws s3 cp $BOOT/mlb.tar        /mnt/fast/mlb.tar
tar -xf /mnt/fast/miniconda3.tar -C /home/ec2-user
tar -xf /mnt/fast/mlb.tar        -C /home/ec2-user
chown -R ec2-user:ec2-user /home/ec2-user/miniconda3 /home/ec2-user/mlb
rm -f /mnt/fast/miniconda3.tar /mnt/fast/mlb.tar

# From S3, not from the extracted repo: mlb.tar lags the working tree and once did not contain
# this script at all. A missing arm script here is fatal and must say so in the log.
aws s3 cp $BOOT/ec2_size_sweep_arm.sh /home/ec2-user/ec2_size_sweep_arm.sh \
  || { echo "FATAL: could not fetch arm script"; exit 1; }
chown ec2-user:ec2-user /home/ec2-user/ec2_size_sweep_arm.sh

# SHUTDOWN=1 with instance-initiated-shutdown-behavior=terminate: the arm ships its log and
# artifacts to S3 in its EXIT trap first, so a failed arm is still diagnosable after the box
# is gone.
#
# setsid + background so cloud-init FINISHES instead of blocking for five hours inside
# cloud-init-final.service. A long-running foreground child there is at the mercy of that
# unit's timeout, and a training run killed by systemd at some arbitrary epoch would look
# like a training failure.
setsid sudo -u ec2-user -i env ARM=$arm D_MODEL=$dmodel N_LAYERS=$nlayers \
  SWEEP_ID=$SWEEP_ID SHUTDOWN=1 \
  bash /home/ec2-user/ec2_size_sweep_arm.sh </dev/null >/dev/null 2>&1 &
echo "=== arm $arm detached, cloud-init returning \$(date -u) ==="
USERDATA
)
  local id
  id=$(aws ec2 run-instances \
    --image-id "$AMI" --instance-type "$ITYPE" --key-name "$KEY" \
    --security-group-ids "$SG" --subnet-id "$SUBNET" \
    --iam-instance-profile "Name=$IAM" \
    --instance-initiated-shutdown-behavior terminate \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":150,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --user-data "$ud" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=mlb-sweep-$arm},{Key=SweepId,Value=$SWEEP_ID}]" \
    --query 'Instances[0].InstanceId' --output text) || { echo "arm $arm launch FAILED"; return 1; }
  echo "arm $arm  d_model=$dmodel n_layers=$nlayers  instance=$id"
  echo "$arm $id $dmodel $nlayers" >> "/tmp/sweep_${SWEEP_ID}_instances.txt"
}

# Append, don't truncate: a single-arm relaunch must not erase the record of the arms already
# running, which is the only local map from arm letter to instance id.
touch "/tmp/sweep_${SWEEP_ID}_instances.txt"
want B && launch_arm B 256 4
want C && launch_arm C 192 3
want D && launch_arm D 128 2

echo
echo "=== launched. sweep_id=$SWEEP_ID ==="
cat "/tmp/sweep_${SWEEP_ID}_instances.txt"
echo "results land in s3://$BUCKET/deep_learning/size_sweep_$SWEEP_ID/<arm>/"
echo "arm A log:  ssh -i $PEM ec2-user@$EXISTING_IP 'tail -f ~/sweep_A.log'"
