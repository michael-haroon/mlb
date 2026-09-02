"""Launch 10 EC2 instances in parallel, one per sizing target.

Each instance:
  1. Launches as c6i.2xlarge with terminate-on-shutdown
  2. Bootstraps Python 3.11 + deps via user-data
  3. Runs run_sizing_ec2.py --target <target> --self-terminate
  4. Uploads sizing_curve_{target}.json to S3
  5. Halts OS → EC2 terminates automatically

Usage (local):
    python scripts/launch_sizing_parallel.py
    python scripts/launch_sizing_parallel.py --poll     # just poll existing run
    python scripts/launch_sizing_parallel.py --targets home_win yrfi  # subset
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import boto3

AMI      = "ami-0bdc7d025135d7b49"   # AL2023 x86_64
ITYPE    = "c6i.2xlarge"              # 8 vCPU, 16GB — sufficient per target
KEY      = "awstest"
SG       = "sg-0583a4de608a95a41"
SUBNET   = "subnet-013362590293de96a"
PROFILE  = "read-write-mlb-s3"
BUCKET   = "mlb-265753586044-us-east-1-an"

ALL_TARGETS = [
    "home_win", "yrfi", "first_5_home_win", "extra_innings",
    "home_run_diff", "total_runs", "home_runs", "away_runs",
    "first_5_home_run_diff", "first_5_total_runs",
]

# User-data: runs on boot as root, installs deps, launches sizing, self-terminates
USER_DATA_TEMPLATE = """\
#!/bin/bash
set -e
exec > /var/log/sizing_boot.log 2>&1

dnf install -y python3.11 python3.11-pip git
pip3.11 install -q boto3 catboost lightgbm numpy optuna pandas pyarrow \\
    scikit-learn scipy xgboost ydf

# Pull project from S3 — tarball has mlb/ prefix, extract one level up
aws s3 cp s3://{bucket}/artifacts/code/mlb_code.tar.gz /tmp/mlb_code.tar.gz
tar -xzf /tmp/mlb_code.tar.gz -C /home/ec2-user

cd /home/ec2-user/mlb
python3.11 scripts/run_sizing_ec2.py --target {target} --self-terminate \\
    > /var/log/sizing_{target}.log 2>&1
"""


def pack_and_upload_code(s3) -> None:
    """Tar the project (excluding artifacts/data) and upload to S3."""
    import subprocess
    import tempfile

    proj = Path(__file__).resolve().parents[1]
    tarball = Path(tempfile.mktemp(suffix=".tar.gz"))

    excludes = [
        "--exclude=__pycache__", "--exclude=*.pyc",
        "--exclude=.git", "--exclude=pregame/artifacts",
        "--exclude=data", "--exclude=catboost_info",
        "--exclude=.claude", "--exclude=tmp",
    ]
    cmd = ["tar", "-czf", str(tarball)] + excludes + ["-C", str(proj.parent), proj.name]
    subprocess.run(cmd, check=True)

    key = "artifacts/code/mlb_code.tar.gz"
    print(f"Uploading code tarball to s3://{BUCKET}/{key} ({tarball.stat().st_size // 1024}KB)...")
    s3.upload_file(str(tarball), BUCKET, key)
    tarball.unlink()
    print("Code uploaded.")


def launch_instance(ec2, target: str) -> str:
    user_data = USER_DATA_TEMPLATE.format(bucket=BUCKET, target=target)
    resp = ec2.run_instances(
        ImageId=AMI,
        InstanceType=ITYPE,
        KeyName=KEY,
        SecurityGroupIds=[SG],
        SubnetId=SUBNET,
        IamInstanceProfile={"Name": PROFILE},
        InstanceInitiatedShutdownBehavior="terminate",
        UserData=user_data,
        MinCount=1, MaxCount=1,
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": f"mlb-sizing-{target}"},
                {"Key": "SizingTarget", "Value": target},
            ],
        }],
    )
    return resp["Instances"][0]["InstanceId"]


def poll_instances(ec2, instance_ids: list[str], targets: list[str]) -> None:
    id_to_target = dict(zip(instance_ids, targets))
    done = set()
    print(f"\nMonitoring {len(instance_ids)} instances...")
    print(f"{'Target':<28} {'InstanceId':<22} {'State'}")
    print("-" * 65)

    # EC2 is eventually consistent — new instances may not be visible immediately
    time.sleep(15)

    while len(done) < len(instance_ids):
        try:
            resp = ec2.describe_instances(InstanceIds=instance_ids)
        except ec2.exceptions.ClientError as exc:
            if "InvalidInstanceID.NotFound" in str(exc):
                time.sleep(10)
                continue
            raise
        states = {}
        for res in resp["Reservations"]:
            for inst in res["Instances"]:
                states[inst["InstanceId"]] = inst["State"]["Name"]

        for iid in instance_ids:
            state = states.get(iid, "unknown")
            target = id_to_target[iid]
            if state == "terminated" and iid not in done:
                done.add(iid)
                print(f"  {target:<26} {iid:<22} TERMINATED ✓")
            elif iid not in done:
                pass  # still running — print summary line below

        running = [iid for iid in instance_ids if iid not in done]
        if running:
            running_targets = [id_to_target[i] for i in running]
            print(f"\r  Running ({len(running_targets)}): {', '.join(running_targets)}  ", end="", flush=True)

        if len(done) < len(instance_ids):
            time.sleep(30)

    print(f"\n\nAll {len(instance_ids)} instances terminated.")


def check_s3_results(s3, targets: list[str]) -> None:
    print("\n=== S3 Results ===")
    all_ok = True
    for target in targets:
        key = f"artifacts/sizing/sizing_curve_{target}.json"
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            data = json.loads(obj["Body"].read())
            pf = data.get("per_family", {})
            ydf = pf.get("ydf_oblique_gbt", {})
            ydf_s = ydf.get("optimal_S", "MISSING")
            n_fams = len(pf)
            print(f"  {target:<28} {n_fams} families | YDF S*={ydf_s}")
        except Exception as e:
            print(f"  {target:<28} MISSING or error: {e}")
            all_ok = False

    if all_ok:
        print("\nAll sizing artifacts present on S3 with YDF S* values.")
    else:
        print("\nSome artifacts missing — check instance logs.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll", action="store_true",
                        help="Poll existing instances by tag instead of launching new ones")
    parser.add_argument("--targets", nargs="+", default=ALL_TARGETS,
                        help="Subset of targets (default: all 10)")
    parser.add_argument("--check-s3", action="store_true",
                        help="Just verify S3 results, no launching")
    args = parser.parse_args()

    ec2 = boto3.client("ec2", region_name="us-east-1")
    s3  = boto3.client("s3",  region_name="us-east-1")

    if args.check_s3:
        check_s3_results(s3, args.targets)
        return

    if args.poll:
        resp = ec2.describe_instances(
            Filters=[
                {"Name": "tag:Name", "Values": [f"mlb-sizing-{t}" for t in args.targets]},
                {"Name": "instance-state-name", "Values": ["pending","running","stopping","stopped"]},
            ]
        )
        instance_ids = [
            inst["InstanceId"]
            for res in resp["Reservations"]
            for inst in res["Instances"]
        ]
        running_targets = [
            next(t["Value"] for t in inst.get("Tags", []) if t["Key"] == "SizingTarget")
            for res in resp["Reservations"]
            for inst in res["Instances"]
        ]
        if not instance_ids:
            print("No running sizing instances found.")
            check_s3_results(s3, args.targets)
            return
        poll_instances(ec2, instance_ids, running_targets)
        check_s3_results(s3, args.targets)
        return

    # ── Launch phase ──────────────────────────────────────────────────────
    pack_and_upload_code(s3)

    print(f"\nLaunching {len(args.targets)} instances ({ITYPE})...")
    instance_ids = []
    for target in args.targets:
        iid = launch_instance(ec2, target)
        instance_ids.append(iid)
        print(f"  {target:<28} {iid}")

    # Save manifest so --poll can recover it
    manifest = {"targets": args.targets, "instance_ids": instance_ids}
    manifest_path = Path(__file__).parent / ".sizing_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest saved to {manifest_path}")

    poll_instances(ec2, instance_ids, args.targets)
    check_s3_results(s3, args.targets)


if __name__ == "__main__":
    main()
