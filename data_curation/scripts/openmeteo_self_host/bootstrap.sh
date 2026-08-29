#!/bin/bash
# Bootstrap Open-Meteo self-hosted API server on Is4gen.xlarge (Amazon Linux 2023, ARM64).
# Uses local NVMe instance store (7.5 TB) — no EBS needed for data.
# NVMe is ephemeral: data is lost on terminate. That's fine — fetch_weather.py
# writes all output to S3. The Open-Meteo data dir is scratch.
#
# Usage:
#   ssh ec2-user@<ip> 'bash -s' < bootstrap.sh

set -euo pipefail

DATADIR="/data/openmeteo"

# ── Find and format NVMe instance store ──────────────────────────────────────
# Is4gen.xlarge: root is nvme0n1, instance store is nvme1n1
NVME_DEV=$(lsblk -d -o NAME,MODEL --json | python3 -c "
import json, sys
d = json.load(sys.stdin)
for dev in d['blockdevices']:
    if 'nvme' in dev['name'] and dev['name'] != 'nvme0n1':
        print('/dev/' + dev['name'])
        break
" 2>/dev/null || echo "")

if [ -z "$NVME_DEV" ]; then
    # Fallback: first non-root nvme device
    NVME_DEV=$(lsblk -d -o NAME | grep nvme | grep -v nvme0 | head -1)
    NVME_DEV="/dev/${NVME_DEV}"
fi

echo "NVMe device: ${NVME_DEV}"
sudo mkfs.xfs -f "$NVME_DEV"
sudo mkdir -p "$DATADIR"
sudo mount "$NVME_DEV" "$DATADIR"
sudo chmod 777 "$DATADIR"  # Docker container user ≠ ec2-user; needs world-writable
echo "Mounted ${NVME_DEV} → ${DATADIR} ($(df -h "$DATADIR" | tail -1 | awk '{print $2}') available)"

# ── Python + AWS deps (for fetch_weather.py) ─────────────────────────────────
sudo dnf install -y docker python3.11 python3.11-pip
python3.11 -m pip install -q requests boto3 pandas numpy tqdm pyarrow

# ── Docker ────────────────────────────────────────────────────────────────────
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user

# ── Pull Open-Meteo image ─────────────────────────────────────────────────────
sudo docker pull ghcr.io/open-meteo/open-meteo:latest

# ── API server (no --restart: one-time backfill, terminate when done) ─────────
sudo docker run -d \
  --name openmeteo-api \
  -v "${DATADIR}:/app/data" \
  -p 8080:8080 \
  ghcr.io/open-meteo/open-meteo:latest

LOCAL_IP=$(curl -s http://169.254.169.254/latest/meta-data/local-ipv4 2>/dev/null || echo "localhost")
echo ""
echo "API server running at http://${LOCAL_IP}:8080"
echo "Data dir: ${DATADIR} (NVMe — ephemeral)"
echo ""
echo "Next: rsync scripts, then run sync_all.sh"
echo "  export OPENMETEO_DATADIR=${DATADIR}"
