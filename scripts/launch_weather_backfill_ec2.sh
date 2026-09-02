#!/bin/bash
# Launch EC2 t3.medium for one-time weather backfill (2015–present).
# Instance self-terminates when done; log uploaded to S3.
set -e
python3 scripts/run_weather_backfill_ec2.py "$@"
