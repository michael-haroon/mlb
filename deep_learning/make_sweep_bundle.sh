#!/bin/bash
# Runs ON the existing GPU box. Publishes its conda env + repo to S3 so the sweep boxes get a
# BIT-IDENTICAL environment instead of resolving their own.
#
# WHY A BUNDLE AND NOT `pip install`:
# arm A runs on this box (torch 2.5.1+cu121). If B/C/D pip-installed their own torch they could
# land a different build, and numpy/torch minor versions change reduction order and cudnn
# kernel selection. In a sweep whose whole output is a val-loss ranking across architectures,
# a per-box library difference is indistinguishable from an architecture effect. The bundle
# makes the environment a constant by construction.
#
# The env is relocated to the SAME absolute path (/home/ec2-user/miniconda3) on the same base
# AMI, which is why a plain tar works and conda-pack is unnecessary.
#
# Usage:  bash make_sweep_bundle.sh
# Log:    ~/make_bundle.log
set -uo pipefail

BUCKET=${BUCKET:-mlb-265753586044-us-east-1-an}
DEST="s3://$BUCKET/deep_learning/sweep_bootstrap"
STAGE=${STAGE:-/mnt/fast/bundle}
LOG=/home/ec2-user/make_bundle.log

exec >>"$LOG" 2>&1
echo "=== bundle start $(date -u +%FT%TZ) ==="
fail() { echo "ABORT: $*"; exit 1; }

[ -x /home/ec2-user/miniconda3/envs/pred/bin/python ] || fail "no pred env to bundle"
mkdir -p "$STAGE"

# Record the exact versions the sweep will run on. This file is the audit trail for "were all
# four arms the same environment" and is cheap to keep next to the results.
/home/ec2-user/miniconda3/envs/pred/bin/python - <<'PYEOF' > "$STAGE/env_fingerprint.txt"
import numpy, torch, platform
print("python  ", platform.python_version())
print("torch   ", torch.__version__)
print("cuda    ", torch.version.cuda)
print("cudnn   ", torch.backends.cudnn.version())
print("numpy   ", numpy.__version__)
PYEOF
cat "$STAGE/env_fingerprint.txt"

# Uncompressed tar: in-region S3 throughput beats gzip's CPU cost on 6.8 GB, and the bundle is
# read exactly three times.
echo "--- tar env $(date -u +%FT%TZ) ---"
tar -cf "$STAGE/miniconda3.tar" -C /home/ec2-user miniconda3 || fail "env tar failed"
echo "--- tar repo $(date -u +%FT%TZ) ---"
tar -cf "$STAGE/mlb.tar" -C /home/ec2-user mlb || fail "repo tar failed"
ls -l "$STAGE"

# s5cmd rides along: the base DLAMI has no s5cmd, and a 27 GB tensor pull over `aws s3 sync`
# is several minutes slower per box. Shipping the exact binary this box uses also keeps the
# transfer tool a constant across arms.
cp /usr/local/bin/s5cmd "$STAGE/s5cmd" 2>/dev/null || echo "WARN: no s5cmd to bundle"

echo "--- upload $(date -u +%FT%TZ) ---"
if command -v s5cmd >/dev/null; then
  s5cmd cp "$STAGE/*" "$DEST/" || fail "upload failed"
else
  aws s3 cp "$STAGE/" "$DEST/" --recursive || fail "upload failed"
fi

# Verify per file, not by total -- a truncated tar extracts partially and the missing piece
# would only surface as an ImportError three minutes into a box's bootstrap.
for f in miniconda3.tar mlb.tar env_fingerprint.txt s5cmd; do
  loc=$(stat -c%s "$STAGE/$f")
  rem=$(aws s3api head-object --bucket "$BUCKET" \
        --key "deep_learning/sweep_bootstrap/$f" --query ContentLength --output text)
  [ "$loc" = "$rem" ] || fail "$f size mismatch local=$loc remote=$rem"
  echo "verified $f $loc bytes"
done
rm -f "$STAGE/miniconda3.tar" "$STAGE/mlb.tar"
echo "=== BUNDLE COMPLETE $(date -u +%FT%TZ) -> $DEST ==="
