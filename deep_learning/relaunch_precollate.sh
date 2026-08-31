#!/bin/bash
# Runs ON the GPU box. Stage 3 of the rebake only — `dataset_cache_new` already completed
# (8.8 GB, manifest valid, 2026-08-31T01:30) and precollate wrote nothing before the box
# wedged, so there is nothing to redo upstream.
#
# WHY THIS EXISTS SEPARATELY FROM rebake_prepared_tensors.sh:
# the first attempt livelocked the whole instance. `[MEM after build] RSS=30.3GB,
# System=32.4/33.3GB (97.4%)` is the measured cause: dataset_cache's build_and_save holds
# frames + train_ds + val_ds + test_ds live at once (it only `del frames` after all three
# splits exist), so the box entered the build phase with ~1GB of headroom. A 32GB swapfile
# was already active and it STILL livelocked — the kernel span in reclaim with no
# reclaimable pages, sshd could not fork, and the SSM agent died too. So swap is not the
# fix, and re-running the whole chain would just walk back into it.
#
# Three guards, in order of how much they actually buy:
#   1. MemoryMax via a systemd scope. A cgroup limit turns an unkillable box into a killed
#      JOB: the OOM killer targets this scope instead of the kernel spinning in global
#      reclaim. This is what converts "40 minutes of blind waiting then a reboot" into one
#      log line. It is the whole point of the wrapper.
#   2. --num-workers 4 instead of 8. Each precollate worker holds its own slice, so worker
#      count multiplies peak RSS. 8 was never validated as safe; it was a default.
#   3. swapon + fstab persistence. /mnt/fast was never in fstab, which is why the reboot
#      returned an empty mount even though the XFS filesystem on the instance store was
#      perfectly intact. nofail matters: a STOP (not reboot) wipes the instance store, and
#      without nofail the next boot would block on a device that has no filesystem.
#
# Usage:  nohup bash relaunch_precollate.sh >/dev/null 2>&1 &
# Log:    ~/precollate.log
set -u

LOG=/home/ec2-user/precollate.log
REPO=/home/ec2-user/mlb
PY=/home/ec2-user/miniconda3/envs/pred/bin/python
CACHE=/mnt/fast/dataset_cache_new
PREP=/mnt/fast/prepared_tensors_new
MEM_MAX=26G

exec >>"$LOG" 2>&1
echo "=== precollate start $(date -u +%FT%TZ) ==="

# --- mount + swap persistence ---------------------------------------------
# Idempotent: re-running must not append duplicate fstab lines or double-add swap.
if ! mountpoint -q /mnt/fast; then
  sudo mount /dev/nvme1n1 /mnt/fast || { echo "MOUNT FAILED"; exit 3; }
fi
grep -q 'LABEL=fastscratch' /etc/fstab || \
  echo 'LABEL=fastscratch /mnt/fast xfs defaults,nofail 0 2' | sudo tee -a /etc/fstab >/dev/null
grep -q '/mnt/fast/swapfile' /etc/fstab || \
  echo '/mnt/fast/swapfile none swap sw,nofail 0 0' | sudo tee -a /etc/fstab >/dev/null
swapon --show=NAME --noheadings | grep -q '/mnt/fast/swapfile' || \
  sudo swapon /mnt/fast/swapfile 2>/dev/null || echo "WARN: swapon failed, continuing"
free -g | head -3

# --- preflight ------------------------------------------------------------
[ -f "$CACHE/manifest.json" ] || { echo "ABORT: $CACHE/manifest.json missing"; exit 4; }
# The output dir must be EMPTY, not merely absent: a half-written memmap set from the
# wedged run would be silently reused as if complete.
if [ -n "$(ls -A "$PREP" 2>/dev/null)" ]; then
  echo "ABORT: $PREP is non-empty — inspect it before reusing"; exit 4
fi
echo "preflight ok: cache manifest present, $PREP empty"

# --- precollate under a cgroup memory cap ---------------------------------
echo "--- precollate (MemoryMax=$MEM_MAX, workers=4) $(date -u +%FT%TZ) ---"
cd "$REPO/deep_learning" || exit 5
# --uid/--gid: without them systemd-run executes as root and every .npy lands root-owned,
# which breaks the training user on the next read.
if ! sudo systemd-run --uid=ec2-user --gid=ec2-user --scope -q \
      -p MemoryMax=$MEM_MAX -p MemorySwapMax=32G \
      "$PY" -m mlb_dl.train_unified precollate \
        --dataset-cache "$CACHE" --output "$PREP" --num-workers 4; then
  echo "PRECOLLATE FAILED (check for a cgroup OOM kill: journalctl -k | grep -i oom)"
  exit 6
fi

du -sh "$PREP"
echo "=== PRECOLLATE COMPLETE $(date -u +%FT%TZ) ==="
# NO weather append step here, unlike the previous rebake. The source dataset_cache now
# carries weather_asof.npz, so precollate writes weather_asof.npy + wx_decision_hour.npy and
# sets has_weather_asof itself (precollate.py:272,316,346,464). Running
# append_weather_asof_to_prepared on this output would be redundant. It is still the right
# tool for retrofitting a prepared dir built from a cache that predates the artifact.
echo "NEXT: verify, then promote."
echo "  ${PY} verify_prepared_tensors.py --cache ${CACHE} --prepared ${PREP}"
