#!/bin/bash
# Promote deep_learning/feature_store_staging/ -> deep_learning/feature_store/ in S3.
#
# WHY THIS IS A SCRIPT AND NOT A `SYNC`:
# the live prefix holds artifacts that staging does NOT contain -- weather_asof/ (12 season
# parquets), weather_asof_pre_qc/, weather_asof_norm.json, wx_hour_offset/, and
# rating_sequences.{npz,json}. A `aws s3 sync --delete` would erase all of them. Promotion is
# therefore an explicit per-file copy of exactly the 15 core tables plus the 2 json sidecars
# that build_feature_store actually produces. Everything else is left untouched by design.
#
# WHAT CHANGES (verified 2026-08-30 by cmp_store.py / diff_games.py):
#   +916 games  season 2026 (closes the 2026-06-20 -> 2026-08-30 gap that voided every artifact)
#   -1279 games season 2020 (SKIP_SEASONS; the live store predates that filter being applied)
#   net -363 games; game_meta gains 4 cols (division/wild_card games_back)
#
# All copies are S3->S3 server-side, so this is fast from anywhere and moves no bytes through
# the machine running it.
#
# Usage:
#   bash deep_learning/promote_feature_store.sh            # dry run, prints the plan
#   bash deep_learning/promote_feature_store.sh --apply    # back up, then promote
set -euo pipefail

BUCKET="${BUCKET:-mlb-265753586044-us-east-1-an}"
LIVE="s3://$BUCKET/deep_learning/feature_store"
STG="s3://$BUCKET/deep_learning/feature_store_staging"
BAK="s3://$BUCKET/deep_learning/feature_store_bak_$(date -u +%Y%m%d)"
APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

CORE="game_meta game_targets player_batting_targets player_pitching_targets team_games
player_batting_history player_pitching_history player_bios pitch_sequences runner_states
batted_balls live_snapshots weather_features venue_dimensions daily_stats"
SIDECARS="manifest.json market_specs.json"

say() { echo "$(date -u +%H:%M:%SZ) $*"; }

# Preflight: refuse to promote a staging prefix that is missing tables. A partial promotion
# leaves the store internally inconsistent (game_meta from one build, pitches from another),
# which is far worse than not promoting at all.
say "preflight: checking staging completeness"
missing=0
for t in $CORE; do
  aws s3 ls "$STG/$t.parquet" >/dev/null 2>&1 || { echo "  MISSING $t.parquet"; missing=1; }
done
for j in $SIDECARS; do
  aws s3 ls "$STG/$j" >/dev/null 2>&1 || { echo "  MISSING $j"; missing=1; }
done
[ "$missing" -eq 1 ] && { say "ABORT: staging is incomplete"; exit 1; }
say "preflight OK — 15 tables + 2 sidecars present in staging"

# Record what will survive, so the post-check can prove nothing was collaterally deleted.
say "derived artifacts that MUST survive (not in staging):"
aws s3 ls "$LIVE/" --recursive 2>/dev/null | awk '{print $4}' \
  | grep -E 'weather_asof|rating_sequences|wx_hour_offset' \
  | sed 's|.*feature_store/||' | cut -d/ -f1 | sort | uniq -c | sed 's/^/  /'
SURVIVE_BEFORE=$(aws s3 ls "$LIVE/" --recursive 2>/dev/null | awk '{print $4}' \
  | grep -cE 'weather_asof|rating_sequences|wx_hour_offset' || true)
say "  ($SURVIVE_BEFORE objects must still be there afterwards)"

if [ "$APPLY" -eq 0 ]; then
  say "DRY RUN — would back up to $BAK then copy:"
  for t in $CORE; do echo "    $STG/$t.parquet -> $LIVE/$t.parquet"; done
  for j in $SIDECARS; do echo "    $STG/$j -> $LIVE/$j"; done
  say "re-run with --apply to execute"
  exit 0
fi

say "1/3 backing up the ${BAK##*/} copies being replaced"
for t in $CORE; do aws s3 cp "$LIVE/$t.parquet" "$BAK/$t.parquet" --only-show-errors; done
for j in $SIDECARS; do aws s3 cp "$LIVE/$j" "$BAK/$j" --only-show-errors; done
say "  backup holds $(aws s3 ls "$BAK/" --recursive | wc -l | tr -d ' ') objects"

say "2/3 promoting staging -> live"
for t in $CORE; do aws s3 cp "$STG/$t.parquet" "$LIVE/$t.parquet" --only-show-errors; done
for j in $SIDECARS; do aws s3 cp "$STG/$j" "$LIVE/$j" --only-show-errors; done

say "3/3 verifying the derived artifacts survived"
SURVIVE_AFTER=$(aws s3 ls "$LIVE/" --recursive 2>/dev/null | awk '{print $4}' \
  | grep -cE 'weather_asof|rating_sequences|wx_hour_offset' || true)
if [ "$SURVIVE_AFTER" -lt "$SURVIVE_BEFORE" ]; then
  say "ANOMALY: derived artifacts dropped $SURVIVE_BEFORE -> $SURVIVE_AFTER; restore from $BAK"
  exit 1
fi
say "PROMOTED — derived artifacts intact ($SURVIVE_AFTER objects). Rollback: copy $BAK/* back."
