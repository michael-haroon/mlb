#!/bin/bash
# Check cap test status from local machine (polls S3 artifacts).
# Usage: bash scripts/check_cap_test_local.sh

echo "=== Cap Test Status — $(date) ==="

echo ""
echo "CAPPED (home_win, cap=20):"
aws s3 ls s3://mlb-265753586044-us-east-1-an/artifacts/importance_cap_test/capped_20/home_win/ 2>/dev/null | tail -5 || echo "  (no artifacts yet)"

echo ""
echo "UNCAPPED (home_runs):"
aws s3 ls s3://mlb-265753586044-us-east-1-an/artifacts/importance_cap_test/uncapped/home_runs/ 2>/dev/null | tail -5 || echo "  (no artifacts yet)"

# Compare cluster maps if both exist
echo ""
CAPPED=$(aws s3 cp s3://mlb-265753586044-us-east-1-an/artifacts/importance_cap_test/capped_20/home_win/clustering_meta.json /tmp/_cap_meta.json 2>/dev/null && echo y || echo n)
UNCAPPED=$(aws s3 cp s3://mlb-265753586044-us-east-1-an/artifacts/importance_cap_test/uncapped/home_runs/clustering_meta.json /tmp/_uncap_meta.json 2>/dev/null && echo y || echo n)

if [ "$CAPPED" = "y" ]; then
    echo "CAPPED META:"
    cat /tmp/_cap_meta.json
    echo ""
fi
if [ "$UNCAPPED" = "y" ]; then
    echo "UNCAPPED META:"
    cat /tmp/_uncap_meta.json
    echo ""
fi

if [ "$CAPPED" = "y" ] && [ "$UNCAPPED" = "y" ]; then
    echo "=== COMPARISON ==="
    bash scripts/check_cap_test.sh remote
fi
