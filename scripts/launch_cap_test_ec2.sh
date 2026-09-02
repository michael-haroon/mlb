#!/bin/bash
# Launch two tmux sessions on EC2: one capped (home_win), one uncapped (home_runs).
# Run this AFTER ssh'ing into the c8g.24xlarge instance.
#
# Prerequisites on EC2:
#   - pip install boto3 pyarrow scikit-learn joblib pandas numpy scipy
#   - Code synced to ~/mlb/ via rsync
#
# Usage:
#   bash launch_cap_test_ec2.sh

set -e
cd ~/mlb

echo "=== Launching capped (home_win, cap=20) in tmux session 'capped' ==="
tmux new-session -d -s capped \
  "python3.11 scripts/run_importance_ec2_cap_test.py --target home_win --cap-clusters 20 2>&1 | tee /tmp/importance_capped.log"

echo "=== Launching uncapped (home_runs) in tmux session 'uncapped' ==="
tmux new-session -d -s uncapped \
  "python3.11 scripts/run_importance_ec2_cap_test.py --target home_runs 2>&1 | tee /tmp/importance_uncapped.log"

echo ""
echo "Both running. Monitor with:"
echo "  tmux attach -t capped"
echo "  tmux attach -t uncapped"
echo "  tail -f /tmp/importance_capped.log"
echo "  tail -f /tmp/importance_uncapped.log"
echo ""
echo "Hourly status: bash scripts/check_cap_test.sh"
