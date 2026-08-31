#!/bin/bash
# Renames the VOID prepared-tensor set in S3 so its name states what it is.
#
# S3 has no rename: this is a server-side copy of 158 GiB followed by a delete of the source.
# The delete is GATED on a per-file verification of the copy and will not run otherwise --
# there is no confirmed versioning on this bucket, so a bad delete is unrecoverable.
#
# WHAT IS BEING RENAMED AND WHY:
# s3://.../deep_learning/prepared_tensors/ holds the artifact produced by the 1950-train
# population bug (manifest: prepared_at 2026-08-27T19:36:16, train.n_games 157,150,
# train.n_samples 1,717,887). The corrected set is 21,384 train games / 315,791 samples and
# now lives at deep_learning/prepared_tensors_20260831/. The bare path is a trap: the planned
# 4-box sweep was one `s5cmd sync` away from silently training every arm on the bug and
# producing plausible, meaningless numbers. Renaming makes the name self-documenting instead
# of relying on every future reader to check the manifest first.
#
# WHERE TO RUN IT: the copy and verify stages work from the GPU box, but the DELETE does not.
# The read-write-mlb-s3 instance role has no s3:DeleteObject (observed 2026-08-31: every one of
# the 118 objects failed AccessDenied after a fully verified copy). That is a sensible role
# scope, so the fix is not to widen it -- run the copy+verify on the box and the final delete
# from a local admin identity. The script is safe to stop after verification for exactly this
# reason: it never deletes anything it has not diffed name-by-name first.
#
# Usage:  nohup bash rename_void_tensors_s3.sh >/dev/null 2>&1 &
# Log:    ~/rename_void.log
set -uo pipefail

BUCKET=${BUCKET:-mlb-265753586044-us-east-1-an}
SRC_PFX=${SRC_PFX:-deep_learning/prepared_tensors}
DST_PFX=${DST_PFX:-deep_learning/prepared_tensors_VOID_1950train_20260827}
SRC="s3://$BUCKET/$SRC_PFX"
DST="s3://$BUCKET/$DST_PFX"
LOG=/home/ec2-user/rename_void.log

exec >>"$LOG" 2>&1
echo "=== rename start $(date -u +%FT%TZ) ==="
echo "SRC $SRC"
echo "DST $DST"

# --- confirm the source really is the void set -----------------------------
# Guard against ever pointing this at the corrected set. The train game count is the single
# field that distinguishes the two populations, so it is checked explicitly rather than
# trusting the prefix name.
ng=$(aws s3 cp "$SRC/manifest.json" - 2>/dev/null \
     | python3 -c "import json,sys; print(json.load(sys.stdin)['splits']['train']['n_games'])" 2>/dev/null)
if [ "${ng:-0}" != "157150" ]; then
  echo "ABORT: $SRC/manifest.json reports train.n_games=${ng:-<none>}, expected 157150."
  echo "       This is NOT the void set. Refusing to touch it."
  exit 2
fi
echo "confirmed void set: train.n_games=157150"

# A partially-copied destination is expected and safe to complete into: DST is a fresh unique
# prefix, so unlike the upload script there is no risk of mixing two different builds here.
# Any object already present is checked against the source in the verify stage below.
n_dst=$(aws s3 ls "$DST/" --recursive 2>/dev/null | wc -l)
[ "$n_dst" -gt 0 ] && echo "destination already holds $n_dst objects; completing the copy"

# --- server-side copy ------------------------------------------------------
# TWO PASSES, and the s5cmd one is allowed to fail.
# S3's CopyObject rejects any source over 5 GiB (hard API limit: 5368709120 bytes) and
# s5cmd v2.2.2 does NOT fall back to multipart UploadPartCopy -- it just errors the object.
# Observed 2026-08-31: 116/118 objects copied, and train/ctx_obs.npy + train/ctx_seqs.npy
# both failed with "copy source is larger than the maximum allowable size". `aws s3 sync`
# does issue UploadPartCopy, so it finishes what s5cmd cannot, and it skips objects already
# present at the same size instead of re-copying the 116.
echo "--- copy pass 1: s5cmd (fast, fails >5GiB) $(date -u +%FT%TZ) ---"
if command -v s5cmd >/dev/null; then
  s5cmd cp "$SRC/*" "$DST/" || echo "s5cmd reported errors (expected for >5GiB objects)"
fi
# --copy-props none is REQUIRED, not tidiness. By default the v2 CLI preserves tags on a
# multipart copy, which makes it call GetObjectTagging -- an action the read-write-mlb-s3 role
# does not have. Observed 2026-08-31: both >5GiB objects failed with AccessDenied on
# s3:GetObjectTagging after transferring 45.7 GiB. These objects carry no tags or user
# metadata worth preserving, so dropping the props copy is a no-op on content.
echo "--- copy pass 2: aws s3 sync (multipart-capable, completes the rest) ---"
aws s3 sync "$SRC/" "$DST/" --copy-props none \
  || { echo "RENAME FAILED: aws sync nonzero"; exit 1; }

# --- verify per file BEFORE deleting anything ------------------------------
# Name-by-name, not a sum: equal totals can hide two offsetting errors, and every filename is
# identical between the two builds so a partial copy is otherwise invisible.
aws s3 ls "$SRC/" --recursive | awk -v p="$SRC_PFX/" '{n=$4; sub("^"p,"",n); print n, $3}' | sort > /tmp/void_src.txt
aws s3 ls "$DST/" --recursive | awk -v p="$DST_PFX/" '{n=$4; sub("^"p,"",n); print n, $3}' | sort > /tmp/void_dst.txt
echo "src objects=$(wc -l < /tmp/void_src.txt) dst objects=$(wc -l < /tmp/void_dst.txt)"
if ! diff /tmp/void_src.txt /tmp/void_dst.txt; then
  echo "RENAME FAILED: copy is not faithful (diff above). SOURCE LEFT INTACT."
  exit 1
fi
awk '{s+=$2} END {printf "verified %d objects, %d bytes copied\n", NR, s}' /tmp/void_dst.txt

# --- only now remove the source -------------------------------------------
echo "--- delete source $(date -u +%FT%TZ) ---"
if command -v s5cmd >/dev/null; then
  s5cmd rm "$SRC/*" || { echo "RENAME FAILED: delete nonzero (copy is safe at $DST)"; exit 1; }
else
  aws s3 rm "$SRC/" --recursive || { echo "RENAME FAILED: delete nonzero (copy is safe at $DST)"; exit 1; }
fi

left=$(aws s3 ls "$SRC/" --recursive 2>/dev/null | wc -l)
echo "objects remaining under the old path: $left"
[ "$left" -eq 0 ] || { echo "RENAME FAILED: $left objects still at $SRC"; exit 1; }
echo "=== RENAME COMPLETE $(date -u +%FT%TZ) ==="
echo "deep_learning/prepared_tensors/ is now EMPTY. Correct set: prepared_tensors_20260831/"
