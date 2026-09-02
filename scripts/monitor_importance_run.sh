#!/bin/bash
# Monitor forward-only LOYO importance instances.
# First 5 iterations: 1-minute interval. After that: hourly.
# Press Ctrl-C to stop.

set -e

BUCKET="mlb-265753586044-us-east-1-an"
TAG_FILTER="Name=tag:Purpose,Values=importance_forward_only_loyo"
QUERY='Reservations[*].Instances[*].{ID:InstanceId,Name:Tags[?Key==`Name`]|[0].Value,State:State.Name,Launch:LaunchTime}'

check() {
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
    aws ec2 describe-instances \
        --filters "$TAG_FILTER" \
        --query "$QUERY" \
        --output table 2>/dev/null

    echo "--- S3 outputs so far ---"
    aws s3 ls "s3://${BUCKET}/artifacts/importance/" 2>/dev/null | awk '{print $NF}' | sort | \
        while read dir; do
            count=$(aws s3 ls "s3://${BUCKET}/artifacts/importance/${dir}" 2>/dev/null | wc -l | tr -d ' ')
            echo "  ${dir}: ${count} files"
        done
    echo ""
}

iteration=0
while true; do
    check
    iteration=$((iteration + 1))
    if [ "$iteration" -lt 6 ]; then
        sleep 60
    else
        sleep 3600
    fi
done
