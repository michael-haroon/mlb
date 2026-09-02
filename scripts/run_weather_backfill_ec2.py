#!/usr/bin/env python3
"""
scripts/run_weather_backfill_ec2.py
-------------------------------------
Launch an EC2 t3.medium to run the one-time weather backfill.

Usage:
  python3 scripts/run_weather_backfill_ec2.py
  python3 scripts/run_weather_backfill_ec2.py --start-year 2017 --skip-upload

The instance self-terminates when the backfill completes.
Logs are uploaded to s3://mlb-265753586044-us-east-1-an/artifacts/logs/ before shutdown.

Follows the same pattern as scripts/launch_pca_crosscheck_ec2.sh.
"""

import argparse
import os
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

import boto3

# ── Config ────────────────────────────────────────────────────────────────────
S3_BUCKET    = "mlb-265753586044-us-east-1-an"
S3_REGION    = "us-east-1"
CODE_S3_KEY  = "artifacts/code/mlb_weather.tar.gz"

AMI_ID         = "ami-0f47531f8c49bd1c6"  # Amazon Linux 2023, same as PCA crosscheck
INSTANCE_TYPE  = "t4g.medium"  # ARM64 to match ami-0f47531f8c49bd1c6 (Graviton)
SECURITY_GROUP = "sg-0583a4de608a95a41"
SUBNET_ID      = "subnet-013362590293de96a"
IAM_PROFILE    = "read-write-mlb-s3"
KEY_PAIR       = "awstest"

PIP_PACKAGES = "boto3 pyarrow pandas numpy requests tqdm"


def _upload_code() -> None:
    """Package fetch_weather.py into a tarball and upload to S3."""
    script = Path("data_curation/scripts/fetch_weather.py")
    if not script.exists():
        sys.exit(f"ERROR: {script} not found — run from the mlb/ project root")

    s3 = boto3.client("s3", region_name=S3_REGION)

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name

    with tarfile.open(tmp_path, "w:gz") as tar:
        tar.add(str(script), arcname="data_curation/scripts/fetch_weather.py")

    print(f"Uploading code → s3://{S3_BUCKET}/{CODE_S3_KEY} ...")
    s3.upload_file(tmp_path, S3_BUCKET, CODE_S3_KEY)
    os.unlink(tmp_path)
    print("Upload complete.")


def _user_data(start_year: int, end_year: int, partition: str) -> str:
    partition_flag = f"    --partition {partition} \\\n" if partition else ""
    return f"""#!/bin/bash
set -e
LOG_FILE="/var/log/weather_backfill.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Weather backfill starting $(date) partition={partition} ==="

dnf install -y python3.12 python3.12-pip
python3.12 -m pip install --quiet {PIP_PACKAGES}

aws s3 cp s3://{S3_BUCKET}/{CODE_S3_KEY} /tmp/mlb_weather.tar.gz
mkdir -p /home/ec2-user/mlb
tar -xzf /tmp/mlb_weather.tar.gz -C /home/ec2-user/mlb
mkdir -p /home/ec2-user/mlb/data/logs
cd /home/ec2-user/mlb

python3.12 data_curation/scripts/fetch_weather.py \\
    --mode backfill \\
    --start-year {start_year} \\
    --end-year {end_year} \\
{partition_flag}
EXIT_CODE=$?
echo "=== Script exited with code $EXIT_CODE at $(date) ==="

STAMP=$(date +%Y%m%d_%H%M%S)
PART=$(echo "{partition}" | tr '/' '-')
aws s3 cp "$LOG_FILE" s3://{S3_BUCKET}/artifacts/logs/weather_backfill_${{PART}}_${{STAMP}}.log || true

shutdown -h now
"""


def _launch_instance(ec2, start_year: int, end_year: int, partition: str,
                     total: int, idx: int) -> str:
    resp = ec2.run_instances(
        ImageId=AMI_ID,
        InstanceType=INSTANCE_TYPE,
        MinCount=1, MaxCount=1,
        KeyName=KEY_PAIR,
        SecurityGroupIds=[SECURITY_GROUP],
        SubnetId=SUBNET_ID,
        IamInstanceProfile={"Name": IAM_PROFILE},
        UserData=_user_data(start_year, end_year, partition),
        InstanceInitiatedShutdownBehavior="terminate",
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name",    "Value": f"weather-backfill-{idx}of{total}"},
                {"Key": "Purpose", "Value": "weather_ingest"},
                {"Key": "Partition", "Value": partition},
            ],
        }],
    )
    return resp["Instances"][0]["InstanceId"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch EC2 for weather backfill")
    parser.add_argument("--start-year",  type=int, default=2015)
    parser.add_argument("--end-year",    type=int, default=None)
    parser.add_argument("--partitions",  type=int, default=1,
                        help="Number of parallel instances to launch (each gets its "
                             "own IP and processes a non-overlapping venue subset).")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip code upload (reuse existing S3 tarball)")
    args = parser.parse_args()

    end_year = args.end_year or datetime.now().year

    if not args.skip_upload:
        _upload_code()

    ec2 = boto3.client("ec2", region_name=S3_REGION)
    instance_ids = []

    for idx in range(args.partitions):
        partition = f"{idx}/{args.partitions}" if args.partitions > 1 else ""
        iid = _launch_instance(ec2, args.start_year, end_year, partition,
                               args.partitions, idx)
        instance_ids.append(iid)
        print(f"  Launched partition {idx}/{args.partitions}: {iid}")

    print(f"\n{args.partitions} instance(s) running — each uses a separate IP/rate-limit quota.")
    print(f"All self-terminate on completion.")
    print(f"Monitor: aws ec2 describe-instances --instance-ids {' '.join(instance_ids)}"
          f" --query 'Reservations[].Instances[].[InstanceId,State.Name]' --output table")
    print(f"Logs:    s3://{S3_BUCKET}/artifacts/logs/weather_backfill_*.log")


if __name__ == "__main__":
    main()
