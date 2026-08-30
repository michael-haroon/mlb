#!/bin/bash
# Waits for the as-of weather artifact to be COMPLETE in S3, then runs the weather A/B.
# Runs ON the GPU box.
#
# Usage:  nohup bash deep_learning/await_artifact_then_ab.sh >/dev/null 2>&1 &
# Log:    ~/await_ab.log   (the A/B itself logs to ~/weather_ab.log)
#
# WHY A WAITER RATHER THAN A HUMAN HANDOFF:
# the finalizer runs on a data box, which holds no SSH key for this box and so cannot
# trigger anything here. The two stages therefore communicate only through S3. Chaining
# them removes ~90min of idle GPU between "artifact finished" and "someone noticed", on
# a run whose two arms take ~10h.
#
# READINESS IS THREE CONDITIONS, NOT ONE:
#   1. weather_asof_norm.json exists. Necessary but NOT sufficient on its own.
#   2. Its "seasons" field covers every required train season. build_norm_stats refuses to
#      fit on a partial artifact, but a sidecar written by an OLDER build predating that
#      guard would look identical while encoding shifted z-scores. The field makes the two
#      distinguishable, so it is checked rather than assumed.
#   3. All population seasons have a season parquet. The sidecar is fit on train seasons
#      only, so it can legitimately exist while a val/test season is still missing -- and a
#      missing val/test season is fatal HERE for a different reason: the test split is 41%
#      2026 games, so losing one drops append coverage to ~59% and trips its 95% gate.
set -uo pipefail

LOG=/home/ec2-user/await_ab.log
REPO=/home/ec2-user/mlb
PY=$HOME/miniconda3/envs/pred/bin/python
INTERVAL=${INTERVAL:-240}
MAX_WAIT_MIN=${MAX_WAIT_MIN:-420}

exec >>"$LOG" 2>&1
echo "=== await artifact -> A/B start $(date -u +%FT%TZ) ==="
cd "$REPO" || { echo "ABORT: no repo at $REPO"; exit 1; }

if pgrep -f "train_unifie[d]" >/dev/null; then
  echo "ABORT: training already on the GPU; refusing to queue a run that would contend."
  exit 1
fi

deadline=$(( $(date +%s) + MAX_WAIT_MIN * 60 ))
while :; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "ABORT: ${MAX_WAIT_MIN}min deadline reached; artifact never became complete."
    exit 1
  fi

  # One python call does all three checks against S3 and prints a single verdict line, so
  # the log stays readable over a multi-hour wait.
  verdict=$("$PY" - <<'PYEOF' 2>&1 | tail -1
import json
import sys

sys.path.insert(0, "deep_learning")
import boto3
import pandas as pd
from mlb_dl.build_weather_asof import (FS_PREFIX, POP_GAME_TYPES, POP_MIN_DATE,
                                       S3_BUCKET, TRAIN_END_DATE, _read_parquet)

s3 = boto3.client("s3")
gm = _read_parquet(f"{FS_PREFIX}/game_meta.parquet",
                   columns=["game_pk", "game_date", "game_type_code"])
gm["game_date"] = pd.to_datetime(gm["game_date"])
pop = gm[(gm["game_date"] >= POP_MIN_DATE) & gm["game_type_code"].isin(POP_GAME_TYPES)]
# int() rather than the raw numpy scalars: these are compared against ints parsed from the
# sidecar's JSON and printed into a log that a human reads, and np.int32 reprs as
# "np.int32(2016)" there.
required_train = sorted(int(y) for y in
                        pop[pop["game_date"] < TRAIN_END_DATE]["game_date"].dt.year.unique())
all_seasons = sorted(int(y) for y in pop["game_date"].dt.year.unique())

r = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=f"{FS_PREFIX}/weather_asof/season=")
present = sorted({int(k["Key"].split("season=")[1].split(".")[0])
                  for k in r.get("Contents", [])})
missing = [s for s in all_seasons if s not in present]

try:
    body = s3.get_object(Bucket=S3_BUCKET,
                         Key=f"{FS_PREFIX}/weather_asof_norm.json")["Body"].read()
    fit_on = json.loads(body).get("seasons")
except s3.exceptions.NoSuchKey:
    print(f"WAIT no sidecar yet; seasons missing: {missing}")
    sys.exit(0)

if fit_on is None:
    print("STALE sidecar has no 'seasons' field -- predates the completeness guard; "
          "rerun build_weather_asof norm-stats")
    sys.exit(0)
short = [s for s in required_train if s not in fit_on]
if short:
    print(f"STALE sidecar fit on {fit_on}, missing required train seasons {short}")
elif missing:
    print(f"WAIT sidecar ok, but season parquets still missing: {missing}")
else:
    print(f"READY sidecar fit on {fit_on}; all {len(all_seasons)} season parquets present")
PYEOF
)
  echo "$(date -u +%H:%M:%SZ) $verdict"
  case "$verdict" in
    READY*) break ;;
    STALE*) echo "ABORT: sidecar is present but not trustworthy; not training on it."; exit 1 ;;
  esac
  sleep "$INTERVAL"
done

echo "=== artifact complete, launching A/B $(date -u +%FT%TZ) ==="
bash deep_learning/ec2_weather_ab.sh
rc=$?
echo "=== A/B exit=$rc $(date -u +%FT%TZ) ==="
exit "$rc"
