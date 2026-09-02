#!/bin/bash
# Check status of capped vs uncapped importance runs.
# Run on EC2 or locally (locally checks S3 for uploaded artifacts).
#
# Usage:
#   bash scripts/check_cap_test.sh          # on EC2 (checks logs + S3)
#   bash scripts/check_cap_test.sh remote   # from local (checks S3 only)

echo "=========================================="
echo "  CAP TEST STATUS — $(date)"
echo "=========================================="
echo ""

if [ "$1" = "remote" ]; then
    echo "--- S3 artifacts (capped/home_win) ---"
    aws s3 ls s3://mlb-265753586044-us-east-1-an/artifacts/importance_cap_test/capped_20/home_win/ 2>/dev/null || echo "  (none yet)"
    echo ""
    echo "--- S3 artifacts (uncapped/home_runs) ---"
    aws s3 ls s3://mlb-265753586044-us-east-1-an/artifacts/importance_cap_test/uncapped/home_runs/ 2>/dev/null || echo "  (none yet)"
    echo ""

    # Try to download and compare cluster maps if both exist
    CAPPED_MAP=$(aws s3 cp s3://mlb-265753586044-us-east-1-an/artifacts/importance_cap_test/capped_20/home_win/cluster_map.json /tmp/cluster_map_capped.json 2>/dev/null && echo "yes" || echo "no")
    UNCAPPED_MAP=$(aws s3 cp s3://mlb-265753586044-us-east-1-an/artifacts/importance_cap_test/uncapped/home_runs/cluster_map.json /tmp/cluster_map_uncapped.json 2>/dev/null && echo "yes" || echo "no")

    if [ "$CAPPED_MAP" = "yes" ] && [ "$UNCAPPED_MAP" = "yes" ]; then
        echo "--- CLUSTER MAP COMPARISON ---"
        CAPPED_K=$(python3 -c "import json; d=json.load(open('/tmp/cluster_map_capped.json')); print(len(d))" 2>/dev/null)
        UNCAPPED_K=$(python3 -c "import json; d=json.load(open('/tmp/cluster_map_uncapped.json')); print(len(d))" 2>/dev/null)
        echo "  Capped clusters:   $CAPPED_K"
        echo "  Uncapped clusters:  $UNCAPPED_K"
        if [ "$CAPPED_K" = "$UNCAPPED_K" ]; then
            echo "  Cluster count: MATCH"
        else
            echo "  Cluster count: DIFFER (capped=$CAPPED_K, uncapped=$UNCAPPED_K)"
        fi
        # Compute ARI if both available
        python3 -c "
import json
from sklearn.metrics import adjusted_rand_score

c1 = json.load(open('/tmp/cluster_map_capped.json'))
c2 = json.load(open('/tmp/cluster_map_uncapped.json'))

# Build label arrays (features should be same set)
all_feats_1 = sorted(f for members in c1.values() for f in members)
all_feats_2 = sorted(f for members in c2.values() for f in members)

if all_feats_1 != all_feats_2:
    print(f'  WARNING: feature sets differ ({len(all_feats_1)} vs {len(all_feats_2)})')
else:
    labels1 = {}
    for cid, members in c1.items():
        for m in members:
            labels1[m] = int(cid)
    labels2 = {}
    for cid, members in c2.items():
        for m in members:
            labels2[m] = int(cid)
    l1 = [labels1[f] for f in all_feats_1]
    l2 = [labels2[f] for f in all_feats_1]
    ari = adjusted_rand_score(l1, l2)
    print(f'  ARI (Adjusted Rand Index): {ari:.6f}')
    if ari > 0.999:
        print('  VERDICT: IDENTICAL clustering')
    elif ari > 0.95:
        print('  VERDICT: Near-identical (minor path difference)')
    else:
        print(f'  VERDICT: DIFFERENT clustering (ARI={ari:.4f})')
" 2>/dev/null || echo "  (could not compute ARI — missing sklearn?)"
    elif [ "$CAPPED_MAP" = "yes" ]; then
        echo "  Capped clustering DONE, uncapped still running."
    elif [ "$UNCAPPED_MAP" = "yes" ]; then
        echo "  Uncapped clustering DONE, capped still running."
    else
        echo "  Neither clustering has uploaded results yet."
    fi
    exit 0
fi

# On EC2: check logs directly
echo "--- Capped (home_win, cap=20) ---"
if [ -f /tmp/importance_capped.log ]; then
    LAST_LINE=$(tail -1 /tmp/importance_capped.log)
    LINE_COUNT=$(wc -l < /tmp/importance_capped.log)
    echo "  Log lines: $LINE_COUNT"
    echo "  Last: $LAST_LINE"
    grep -c "CLUSTERING COMPLETE\|IMPORTANCE COMPLETE\|DONE:" /tmp/importance_capped.log > /dev/null 2>&1 && \
        grep "CLUSTERING COMPLETE\|IMPORTANCE COMPLETE\|DONE:" /tmp/importance_capped.log | tail -3
else
    echo "  (no log file yet)"
fi

echo ""
echo "--- Uncapped (home_runs) ---"
if [ -f /tmp/importance_uncapped.log ]; then
    LAST_LINE=$(tail -1 /tmp/importance_uncapped.log)
    LINE_COUNT=$(wc -l < /tmp/importance_uncapped.log)
    echo "  Log lines: $LINE_COUNT"
    echo "  Last: $LAST_LINE"
    grep -c "CLUSTERING COMPLETE\|IMPORTANCE COMPLETE\|DONE:" /tmp/importance_uncapped.log > /dev/null 2>&1 && \
        grep "CLUSTERING COMPLETE\|IMPORTANCE COMPLETE\|DONE:" /tmp/importance_uncapped.log | tail -3
else
    echo "  (no log file yet)"
fi

echo ""
echo "--- tmux sessions ---"
tmux list-sessions 2>/dev/null || echo "  (no tmux sessions)"
