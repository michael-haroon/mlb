#!/usr/bin/env python3
"""
scripts/run_supplemental_backfill_ec2.py
-----------------------------------------
Launch a t4g.medium to run the supplemental Gumbo historical backfill:
standings by date, weekly rosters, season stats + platoon splits, venue info.

Usage:
  python3 scripts/run_supplemental_backfill_ec2.py
  python3 scripts/run_supplemental_backfill_ec2.py --table standings
  python3 scripts/run_supplemental_backfill_ec2.py --start-year 2020 --skip-upload

The instance self-terminates when the backfill completes.
Logs are uploaded to s3://mlb-265753586044-us-east-1-an/artifacts/logs/ before shutdown.

Pattern mirrors run_weather_backfill_ec2.py.
"""

import argparse
import os
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

import boto3

# ── Config ─────────────────────────────────────────────────────────────────────
S3_BUCKET    = "mlb-265753586044-us-east-1-an"
S3_REGION    = "us-east-1"
CODE_S3_KEY  = "artifacts/code/mlb_supplemental.tar.gz"

AMI_ID         = "ami-0f47531f8c49bd1c6"  # Amazon Linux 2023, ARM64 (Graviton)
INSTANCE_TYPE  = "t4g.medium"             # 4 GB RAM — safe for MAX_WORKERS=20
SECURITY_GROUP = "sg-0583a4de608a95a41"
SUBNET_ID      = "subnet-013362590293de96a"
IAM_PROFILE    = "read-write-mlb-s3"
KEY_PAIR       = "awstest"

PIP_PACKAGES = "boto3 pyarrow pandas requests"


def _upload_code() -> None:
    """Package both scripts into a tarball and upload to S3."""
    scripts = [
        Path("data_curation/scripts/daily_enrichment.py"),
        Path("data_curation/scripts/download_supplemental_history.py"),
    ]
    for s in scripts:
        if not s.exists():
            sys.exit(f"ERROR: {s} not found — run from the mlb/ project root")

    s3 = boto3.client("s3", region_name=S3_REGION)

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name

    with tarfile.open(tmp_path, "w:gz") as tar:
        for s in scripts:
            tar.add(str(s), arcname=str(s))

    print(f"Uploading code → s3://{S3_BUCKET}/{CODE_S3_KEY} ...")
    s3.upload_file(tmp_path, S3_BUCKET, CODE_S3_KEY)
    os.unlink(tmp_path)
    print("Upload complete.")


def _user_data(start_year: int, end_year: int, table: str,
               standings_workers: int, rosters_workers: int) -> str:
    table_flag = f"--table {table}" if table != "all" else ""
    return f"""#!/bin/bash
set -e
LOG_FILE="/var/log/supplemental_backfill.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Supplemental backfill starting $(date) ==="
echo "    table={table} standings_workers={standings_workers} rosters_workers={rosters_workers}"

dnf install -y python3.12 python3.12-pip
python3.12 -m pip install --quiet {PIP_PACKAGES}

aws s3 cp s3://{S3_BUCKET}/{CODE_S3_KEY} /tmp/mlb_supplemental.tar.gz
mkdir -p /home/ec2-user/mlb
tar -xzf /tmp/mlb_supplemental.tar.gz -C /home/ec2-user/mlb
mkdir -p /home/ec2-user/mlb/data/logs
cd /home/ec2-user/mlb

python3.12 data_curation/scripts/download_supplemental_history.py \\
    --start-year {start_year} \\
    --end-year {end_year} \\
    --standings-workers {standings_workers} \\
    --rosters-workers {rosters_workers} \\
    {table_flag}
EXIT_CODE=$?
echo "=== Script exited with code $EXIT_CODE at $(date) ==="

STAMP=$(date +%Y%m%d_%H%M%S)
aws s3 cp "$LOG_FILE" s3://{S3_BUCKET}/artifacts/logs/supplemental_backfill_${{STAMP}}.log || true

shutdown -h now
"""


def _launch_instance(ec2, start_year: int, end_year: int, table: str,
                     standings_workers: int, rosters_workers: int) -> str:
    resp = ec2.run_instances(
        ImageId=AMI_ID,
        InstanceType=INSTANCE_TYPE,
        MinCount=1, MaxCount=1,
        KeyName=KEY_PAIR,
        SecurityGroupIds=[SECURITY_GROUP],
        SubnetId=SUBNET_ID,
        IamInstanceProfile={"Name": IAM_PROFILE},
        UserData=_user_data(start_year, end_year, table, standings_workers, rosters_workers),
        InstanceInitiatedShutdownBehavior="terminate",
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name",    "Value": f"supplemental-backfill-{table}"},
                {"Key": "Purpose", "Value": "supplemental_ingest"},
            ],
        }],
    )
    return resp["Instances"][0]["InstanceId"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch EC2 for supplemental Gumbo backfill")
    parser.add_argument("--start-year",  type=int, default=2015)
    parser.add_argument("--end-year",    type=int, default=None)
    parser.add_argument("--table",
                        choices=["all", "standings", "rosters", "stats", "splits", "venues"],
                        default="all")
    parser.add_argument("--standings-workers", type=int, default=30)
    parser.add_argument("--rosters-workers",   type=int, default=10)
    parser.add_argument("--skip-upload", action="store_true",
                        help="Reuse existing S3 tarball; skip code upload")
    args = parser.parse_args()

    end_year = args.end_year or datetime.now().year

    if not args.skip_upload:
        _upload_code()

    ec2 = boto3.client("ec2", region_name=S3_REGION)
    iid = _launch_instance(ec2, args.start_year, end_year, args.table,
                           args.standings_workers, args.rosters_workers)

    print(f"\nLaunched: {iid}")
    print(f"Table: {args.table}  Years: {args.start_year}–{end_year}")
    print(f"Workers: standings={args.standings_workers}  rosters={args.rosters_workers}")
    print(f"Instance self-terminates on completion.")
    print()
    print("Monitor (run this locally):")
    print(f"  watch -n 30 \"aws ec2 describe-instances --instance-ids {iid} "
          f"--query 'Reservations[].Instances[].[InstanceId,State.Name,LaunchTime]' "
          f"--output table && echo '--- S3 file counts ---' && "
          f"for t in standings rosters pitcher_stats hitter_stats pitcher_splits hitter_splits; do "
          f"echo -n \\\"  \\$t: \\\"; "
          f"aws s3 ls s3://{S3_BUCKET}/data/ --recursive | grep \\\"\\$t\\\" | wc -l; done\"")
    print()
    print(f"Live log (once instance is ready, ~2min after launch):")
    print(f"  ssh -i ~/Documents/SENSITIVE/awstest.pem ec2-user@$(aws ec2 describe-instances "
          f"--instance-ids {iid} "
          f"--query 'Reservations[0].Instances[0].PublicIpAddress' --output text) "
          f"'tail -f /var/log/supplemental_backfill.log'")
    print()
    print(f"Final log on S3 (after completion):")
    print(f"  aws s3 ls s3://{S3_BUCKET}/artifacts/logs/ | grep supplemental")


if __name__ == "__main__":
    main()
