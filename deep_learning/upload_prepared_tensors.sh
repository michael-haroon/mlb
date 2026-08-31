#!/bin/bash
# Runs ON the GPU box. Uploads a verified prepared_tensors dir to a NEW dated S3 prefix.
#
# Run verify_prepared_tensors.py FIRST. This script only checks that the transfer was
# faithful; it cannot tell a correct artifact from a wrong one, and a 158 GiB void set was
# already uploaded once looking perfectly healthy.
#
# WHY A DATED PREFIX AND NEVER `deep_learning/prepared_tensors/`:
# that bare path still holds the VOID 1950-train set (157,150 train games, 158 GiB, uploaded
# 2026-08-27). Overwriting in place is exactly what made a stale artifact indistinguishable
# from a fresh one, and four sweep boxes were one `s5cmd sync` away from training on it.
# Deleting the void set is a destructive call that belongs to the user, not to this script.
#
# Usage:  bash upload_prepared_tensors.sh [prefix_date]     # default: today, UTC
# Log:    ~/upload_prepared.log
set -uo pipefail

SRC=${SRC:-/mnt/fast/prepared_tensors}
BUCKET=${BUCKET:-mlb-265753586044-us-east-1-an}
STAMP=${1:-$(date -u +%Y%m%d)}
DEST="s3://$BUCKET/deep_learning/prepared_tensors_$STAMP"
LOG=/home/ec2-user/upload_prepared.log

exec >>"$LOG" 2>&1
echo "=== upload start $(date -u +%FT%TZ) -> $DEST ==="

[ -f "$SRC/manifest.json" ] || { echo "ABORT: no manifest at $SRC"; exit 2; }

# Refuse to write into a non-empty prefix. Appending into one silently mixes two builds, and
# the mix is undetectable afterwards because every file name is identical between builds.
n=$(aws s3 ls "$DEST/" --recursive 2>/dev/null | wc -l)
if [ "$n" -gt 0 ]; then
  echo "ABORT: $DEST already holds $n objects — bump the date or delete deliberately"; exit 2
fi

if command -v s5cmd >/dev/null; then
  s5cmd cp "$SRC/*" "$DEST/" || { echo "UPLOAD FAILED: s5cmd nonzero"; exit 1; }
else
  aws s3 sync "$SRC" "$DEST/" || { echo "UPLOAD FAILED: aws sync nonzero"; exit 1; }
fi

# --- verify per file, not by total ----------------------------------------
# Two deliberate choices here, both learned the hard way on 2026-08-31:
#   1. Compare FILE sizes only. `du -sb` counts directory inodes, which S3 has no equivalent
#      of, so a perfect upload reported a 12,351-byte shortfall (3 x 4096 for train/val/test
#      plus 63 for the top dir) and failed a healthy transfer.
#   2. Compare name-by-name rather than summing. Equal totals can hide two offsetting errors,
#      and a short .npy is the dangerous case: np.load(mmap_mode='r') serves misaligned rows
#      instead of raising.
find "$SRC" -type f -printf "%P %s\n" | sort > /tmp/prep_local.txt
aws s3 ls "$DEST/" --recursive \
  | awk -v p="deep_learning/prepared_tensors_$STAMP/" '{n=$4; sub("^"p,"",n); print n, $3}' \
  | sort > /tmp/prep_remote.txt

echo "local files=$(wc -l < /tmp/prep_local.txt) remote objects=$(wc -l < /tmp/prep_remote.txt)"
if ! diff /tmp/prep_local.txt /tmp/prep_remote.txt; then
  echo "UPLOAD FAILED: per-file name/size mismatch above"; exit 1
fi
awk '{s+=$2} END {printf "verified %d objects, %d bytes\n", NR, s}' /tmp/prep_local.txt
echo "=== UPLOAD COMPLETE $(date -u +%FT%TZ) ==="
echo "Pull with: s5cmd cp '$DEST/*' /mnt/fast/prepared_tensors/"
