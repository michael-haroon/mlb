#!/bin/bash
# Runs ON a data box (never locally) once every shard's chain_build_weather_asof.sh has
# reported CHAIN COMPLETE. Turns 12 independently-built season tensors into a training-
# ready artifact: repairs any season the shards skipped, fits the shared standardizer,
# then runs every verifier over the WHOLE artifact rather than per-shard slices.
#
# Usage:  nohup bash finalize_weather_asof.sh [parity_date] >/dev/null 2>&1 &
# Log:    ~/finalize_asof.log
#
# Why this exists as its own stage rather than at the tail of each shard's chain: the
# standardizer is fit across all seasons at once, so it cannot run until the last season
# lands, and each shard only ever verified the two seasons it owned. A per-shard gate
# passing 12 times does NOT imply the artifact is complete -- a shard that dies before
# arming its chain leaves a hole no other shard checks. Season completeness is therefore
# re-checked here against the build population.
#
# Ordering is load-bearing: norm-stats must precede the loader-contract and parity
# verifiers, because both z-score through weather_asof_norm.json and silently fall back
# to raw units when it is absent (see _load_weather_asof_artifacts' warning path).
set -u

PARITY_DATE="${1:-}"
LOG=/home/ec2-user/finalize_asof.log
REPO=/home/ec2-user/mlb
PY=$(command -v python3.11 || echo "$HOME/miniconda3/envs/pred/bin/python")
BUCKET=mlb-265753586044-us-east-1-an
PREFIX=deep_learning/feature_store/weather_asof

exec >>"$LOG" 2>&1
echo "=== finalize start $(date -u +%FT%TZ) py=$PY ==="

fail() { echo "ABORT: $*"; exit 1; }

cd "$REPO" || fail "no repo at $REPO"

# --- Preflight -------------------------------------------------------------
( cd "$REPO/deep_learning" && "$PY" -m mlb_dl.build_weather_asof --help ) >/dev/null 2>&1 \
  || fail "builder not importable from $REPO/deep_learning"
for v in verify_weather_asof_artifact.py verify_asof_loader_contracts.py \
         verify_asof_train_live_parity.py verify_weather_archives.py; do
  [ -f "data_curation/scripts/$v" ] || fail "verifier missing: $v"
done
echo "preflight ok"

# --- Stage 1: repair any season the shards skipped --------------------------
# Two tiers, because they have different consequences. REQUIRED = population seasons
# holding TRAIN games: omitting one biases the standardizer, so a hole there aborts the
# run. The rest only affect artifact coverage, and 2026's HRRR backfill is deliberately
# deferred, so a hole there is reported and the run continues. REQUIRED is derived exactly
# as build_norm_stats derives it, so the repair loop and the guard cannot disagree.
read -r REQUIRED_LINE ALL_LINE < <("$PY" - <<'PYEOF'
import sys
sys.path.insert(0, "deep_learning")
import pandas as pd
from mlb_dl.build_weather_asof import (FS_PREFIX, POP_GAME_TYPES, POP_MIN_DATE,
                                       TRAIN_END_DATE, _read_parquet)
gm = _read_parquet(f"{FS_PREFIX}/game_meta.parquet",
                   columns=["game_pk", "game_date", "game_type_code"])
gm["game_date"] = pd.to_datetime(gm["game_date"])
pop = gm[(gm["game_date"] >= POP_MIN_DATE) & gm["game_type_code"].isin(POP_GAME_TYPES)]
req = sorted(pop[pop["game_date"] < TRAIN_END_DATE]["game_date"].dt.year.unique())
alls = sorted(pop["game_date"].dt.year.unique())
print(",".join(map(str, req)), ",".join(map(str, alls)))
PYEOF
) || fail "could not derive seasons from game_meta"
REQUIRED=${REQUIRED_LINE//,/ }
ALL_SEASONS=${ALL_LINE//,/ }
echo "required (train) seasons: $REQUIRED"
echo "all population seasons:   $ALL_SEASONS"

PRESENT=$(aws s3 ls "s3://$BUCKET/$PREFIX/" 2>/dev/null \
          | sed -n 's/.*season=\([0-9]*\)\.parquet/\1/p' | sort -u | tr '\n' ' ')
echo "present seasons:          $PRESENT"

build_season() {
  local S=$1
  echo "--- repair: season $S missing, gating archive $(date -u +%FT%TZ) ---"
  # Same gate the shard chains use. A season whose HRRR archive is incomplete must stay
  # missing rather than get built from partial input.
  "$PY" data_curation/scripts/verify_weather_archives.py \
    completeness --year "$S" --sample 400 || return 1
  ( cd "$REPO/deep_learning" && \
    flock -n "/tmp/wx_asof_build_$S.lock" \
      "$PY" -m mlb_dl.build_weather_asof build --season "$S" --workers 6 ) || return 1
  echo "REPAIR BUILD OK $S $(date -u +%FT%TZ)"
}

DEFERRED=""
for S in $ALL_SEASONS; do
  case " $PRESENT " in *" $S "*) continue;; esac
  if build_season "$S"; then
    PRESENT="$PRESENT $S"
    continue
  fi
  case " $REQUIRED " in
    *" $S "*) fail "required train season $S is absent and could not be built; its HRRR
       archive needs a --force backfill rerun before the standardizer can be fit" ;;
    *) echo "WARN: non-train season $S absent and not buildable (backfill deferred);
       continuing — it cannot affect train-only norm stats"
       DEFERRED="$DEFERRED $S" ;;
  esac
done

# --- Stage 2: fit the shared standardizer ----------------------------------
# Refuses to run on a partial artifact and records the seasons it saw, so a sidecar built
# before a repair is distinguishable from one built after.
echo "--- norm-stats $(date -u +%FT%TZ) ---"
( cd "$REPO/deep_learning" && "$PY" -m mlb_dl.build_weather_asof norm-stats ) \
  || fail "norm-stats failed (partial artifact, or S3 read error)"

# --- Stage 3: verify the whole artifact ------------------------------------
# Every present season, not the per-shard slices the chains already checked. Passing
# --seasons explicitly bypasses the verifier's own EXPECTED_SEASONS gate, which is what we
# want here: stage 1 already decided which holes are fatal, and that gate would otherwise
# abort on the deferred season while telling us nothing new.
echo "--- artifact audit, seasons:$PRESENT $(date -u +%FT%TZ) ---"
"$PY" data_curation/scripts/verify_weather_asof_artifact.py --seasons $PRESENT \
  || fail "artifact audit failed"

echo "--- loader contracts, all seasons $(date -u +%FT%TZ) ---"
"$PY" data_curation/scripts/verify_asof_loader_contracts.py --all \
  || fail "loader contracts failed"

# --- Stage 4: train/live parity --------------------------------------------
# Needs the sidecar from stage 2: the live path z-scores through it, so running this
# first would compare standardized training tensors against raw live ones.
if [ -n "$PARITY_DATE" ]; then
  echo "--- train/live parity $PARITY_DATE $(date -u +%FT%TZ) ---"
  "$PY" data_curation/scripts/verify_asof_train_live_parity.py \
    --date "$PARITY_DATE" --n-games 20 || fail "train/live parity failed"
else
  echo "SKIP parity: no date given (pass one as \$1 once a slate with archived obs is chosen)"
fi

if [ -n "$DEFERRED" ]; then
  echo "NOTE: seasons$DEFERRED have no artifact (backfill deferred). Training and eval
        cover only the seasons listed above; rerun this script after those backfills."
fi
echo "=== FINALIZE COMPLETE $(date -u +%FT%TZ) ==="
