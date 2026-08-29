"""
Compare derived home_runs/away_runs (from ensemble blends of home_run_diff + total_runs)
against direct OOF models for those targets.

home_runs_pred  = (total_runs_blend + home_run_diff_blend) / 2
away_runs_pred  = (total_runs_blend - home_run_diff_blend) / 2

Same for first_5 variants.
"""
import io, sys
sys.path.insert(0, "/Users/michaelharoon/Projects/prediction_markets/mlb")

import boto3
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

s3 = boto3.client("s3")
BUCKET = "mlb-265753586044-us-east-1-an"

def huber(y, p, delta=1.35):
    r = y - p
    return np.mean(np.where(np.abs(r) <= delta, 0.5 * r**2, delta * (np.abs(r) - 0.5 * delta)))

def load_npy(key):
    return np.load(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()))

def load_pkl(key):
    import pickle
    return pickle.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())

# ── ground truth ──────────────────────────────────────────────────────────────
obj = s3.get_object(Bucket=BUCKET, Key="artifacts/features/game_features.parquet")
df_full = pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
trainable_all = df_full[df_full["target_status"] == "trainable"].reset_index(drop=True)
full_train_idx = np.where((df_full["target_status"] == "trainable").values)[0]
season_full = df_full["season"].values[full_train_idx]
no2020_mask = season_full != 2020
trainable = trainable_all[trainable_all["season"] != 2020].reset_index(drop=True)

def align_oof(arr):
    n_full = len(full_train_idx)
    if len(arr) >= n_full:
        arr = arr[:n_full]
    else:
        arr = np.pad(arr, (0, n_full - len(arr)), constant_values=np.nan)
    return arr[no2020_mask]

# ── list OOF keys ─────────────────────────────────────────────────────────────
ALL_TARGETS = [
    "home_run_diff", "total_runs", "home_runs", "away_runs",
    "first_5_home_run_diff", "first_5_total_runs",
]
paginator = s3.get_paginator("list_objects_v2")
oof_keys = {}
for page in paginator.paginate(Bucket=BUCKET, Prefix="artifacts/models/oof_"):
    for obj2 in page.get("Contents", []):
        k = obj2["Key"]
        fname = k.split("/")[-1]
        if not fname.endswith("_A.npy"):
            continue
        stem = fname[4:-6]
        for t in sorted(ALL_TARGETS, key=len, reverse=True):
            if stem.startswith(t + "_"):
                oof_keys.setdefault(t, {})[stem[len(t)+1:]] = k
                break

# ── build ensemble blend for a target ────────────────────────────────────────
def ensemble_blend(target):
    """Return (blend_array, valid_mask) aligned to trainable-no-2020."""
    pkl = load_pkl(f"artifacts/models/ensemble_{target}_A.pkl")
    member_bundles = pkl["member_bundles"]
    weights = np.array(pkl["weights"])
    y_all = trainable[target].values

    raw_oofs, valid_w = [], []
    for mb, w in zip(member_bundles, weights):
        if w < 1e-6:
            continue
        key2 = (oof_keys.get(target) or {}).get(mb["family"])
        if not key2:
            continue
        try:
            arr = align_oof(load_npy(key2))
        except Exception:
            continue
        raw_oofs.append(arr)
        valid_w.append(w)

    if not raw_oofs:
        return None, None

    w_arr = np.array(valid_w); w_arr /= w_arr.sum()
    valid = ~np.isnan(y_all)
    for o in raw_oofs:
        valid &= ~np.isnan(o)
    blend = np.column_stack([o[valid] for o in raw_oofs]) @ w_arr
    return blend, valid

# ── metrics helper ────────────────────────────────────────────────────────────
def report(label, y, p):
    mae  = mean_absolute_error(y, p)
    rmse = np.sqrt(mean_squared_error(y, p))
    r2   = r2_score(y, p)
    hl   = huber(y, p)
    print(f"  {label:<30}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  Huber={hl:.4f}")

# ── FULL-GAME comparison ──────────────────────────────────────────────────────
print("\n=== Full-game: home_runs & away_runs ===\n")

rd_blend, rd_valid = ensemble_blend("home_run_diff")
tr_blend, tr_valid = ensemble_blend("total_runs")

# Intersection of valid rows for both blend targets
rd_valid_full = np.zeros(len(trainable), dtype=bool); rd_valid_full[rd_valid] = True  # re-expand
# Actually valid masks are already index-aligned to trainable-no-2020; find joint valid
joint_valid = rd_valid & tr_valid  # both valid masks are same length (trainable-no-2020)

# Re-run blends on joint_valid subset
def ensemble_blend_masked(target, mask):
    pkl = load_pkl(f"artifacts/models/ensemble_{target}_A.pkl")
    member_bundles = pkl["member_bundles"]
    weights = np.array(pkl["weights"])
    y_all = trainable[target].values

    raw_oofs, valid_w = [], []
    for mb, w in zip(member_bundles, weights):
        if w < 1e-6:
            continue
        key2 = (oof_keys.get(target) or {}).get(mb["family"])
        if not key2:
            continue
        try:
            arr = align_oof(load_npy(key2))
        except Exception:
            continue
        raw_oofs.append(arr)
        valid_w.append(w)

    if not raw_oofs:
        return None, None

    w_arr = np.array(valid_w); w_arr /= w_arr.sum()
    valid = mask.copy()
    for o in raw_oofs:
        valid &= ~np.isnan(o)
    blend = np.column_stack([o[valid] for o in raw_oofs]) @ w_arr
    return blend, valid

# Compute joint mask (rows where both rd and tr targets are non-null)
y_rd_all = trainable["home_run_diff"].values
y_tr_all = trainable["total_runs"].values
y_hr_all = trainable["home_runs"].values
y_ar_all = trainable["away_runs"].values

base_mask = ~np.isnan(y_rd_all) & ~np.isnan(y_tr_all) & ~np.isnan(y_hr_all) & ~np.isnan(y_ar_all)

rd_b, rd_m = ensemble_blend_masked("home_run_diff", base_mask)
tr_b, tr_m = ensemble_blend_masked("total_runs", base_mask)

# final joint valid (both blends succeeded)
joint = rd_m & tr_m
rd_final = rd_b  # already on joint rows
tr_final = tr_b

# derived predictions
hr_derived = (tr_final + rd_final) / 2
ar_derived = (tr_final - rd_final) / 2

y_hr = y_hr_all[joint]
y_ar = y_ar_all[joint]

report("home_runs (DERIVED)", y_hr, hr_derived)
report("away_runs (DERIVED)", y_ar, ar_derived)

print()
# Direct OOF model metrics on same rows for comparison
for family, key in sorted((oof_keys.get("home_runs") or {}).items()):
    arr = align_oof(load_npy(key))[joint]
    valid2 = ~np.isnan(arr)
    if valid2.sum() < 50: continue
    report(f"home_runs direct [{family}]", y_hr[valid2], arr[valid2])

print()
for family, key in sorted((oof_keys.get("away_runs") or {}).items()):
    arr = align_oof(load_npy(key))[joint]
    valid2 = ~np.isnan(arr)
    if valid2.sum() < 50: continue
    report(f"away_runs direct [{family}]", y_ar[valid2], arr[valid2])

# ── FIRST-5 comparison ────────────────────────────────────────────────────────
print("\n=== First-5: home_runs & away_runs ===\n")

f5_rd_col = "first_5_home_run_diff"
f5_tr_col = "first_5_total_runs"

if f5_rd_col in trainable.columns and f5_tr_col in trainable.columns:
    y_f5rd = trainable[f5_rd_col].values
    y_f5tr = trainable[f5_tr_col].values
    y_f5hr = trainable["first_5_home_runs"].values if "first_5_home_runs" in trainable.columns else None
    y_f5ar = trainable["first_5_away_runs"].values if "first_5_away_runs" in trainable.columns else None

    if y_f5hr is not None and y_f5ar is not None:
        f5_base = ~np.isnan(y_f5rd) & ~np.isnan(y_f5tr) & ~np.isnan(y_f5hr) & ~np.isnan(y_f5ar)
        f5rd_b, f5rd_m = ensemble_blend_masked(f5_rd_col, f5_base)
        f5tr_b, f5tr_m = ensemble_blend_masked(f5_tr_col, f5_base)
        f5joint = f5rd_m & f5tr_m

        f5hr_derived = (f5tr_b + f5rd_b) / 2
        f5ar_derived = (f5tr_b - f5rd_b) / 2

        report("first_5_home_runs (DERIVED)", y_f5hr[f5joint], f5hr_derived)
        report("first_5_away_runs (DERIVED)", y_f5ar[f5joint], f5ar_derived)
    else:
        print("  first_5_home_runs / first_5_away_runs not in feature store columns — skipping")
else:
    print("  first_5 ensemble targets not available — skipping")

print()
