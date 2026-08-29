"""
Step 1: Measure OOF residual correlation between total_runs and home_run_diff.
Tells us whether the independence formula overstates or understates derived-target spread.
Also checks first_5 pair.
"""
import io, sys
sys.path.insert(0, "/Users/michaelharoon/Projects/prediction_markets/mlb")

import boto3
import numpy as np
import pyarrow.parquet as pq

s3 = boto3.client("s3")
BUCKET = "mlb-265753586044-us-east-1-an"

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
ALL_TARGETS = ["home_run_diff", "total_runs", "first_5_home_run_diff", "first_5_total_runs"]
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

def ensemble_blend_array(target):
    """Return aligned ensemble blend for a target (full trainable-no-2020 length, NaN where invalid)."""
    pkl = load_pkl(f"artifacts/models/ensemble_{target}_A.pkl")
    member_bundles = pkl["member_bundles"]
    weights = np.array(pkl["weights"])
    n = len(trainable)

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
        return None

    w_arr = np.array(valid_w); w_arr /= w_arr.sum()
    # Build full-length blend (NaN where any member is NaN)
    matrix = np.column_stack(raw_oofs)  # shape (n, k)
    blend = matrix @ w_arr
    # Mask rows where any input was NaN
    any_nan = np.isnan(matrix).any(axis=1)
    blend[any_nan] = np.nan
    return blend

def analyze_pair(t1, t2, col1, col2, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    b1 = ensemble_blend_array(t1)
    b2 = ensemble_blend_array(t2)
    y1 = trainable[col1].values
    y2 = trainable[col2].values

    joint = ~np.isnan(b1) & ~np.isnan(b2) & ~np.isnan(y1) & ~np.isnan(y2)
    n = joint.sum()
    print(f"  Joint valid rows: {n:,}")

    r1 = y1[joint] - b1[joint]   # total_runs residuals
    r2 = y2[joint] - b2[joint]   # run_diff residuals

    corr = np.corrcoef(r1, r2)[0, 1]
    cov  = np.cov(r1, r2)[0, 1]
    std1 = r1.std()
    std2 = r2.std()

    print(f"\n  Residual std  [{t1}]: {std1:.4f}")
    print(f"  Residual std  [{t2}]: {std2:.4f}")
    print(f"  Covariance   (r1, r2): {cov:.4f}")
    print(f"  Correlation  (r1, r2): {corr:.4f}")

    # Variance of derived target under independence vs actual
    var_indep  = (std1**2 + std2**2) / 4
    var_actual = (std1**2 + std2**2 + 2 * cov) / 4
    sigma_indep  = np.sqrt(var_indep)
    sigma_actual = np.sqrt(var_actual)
    pct_error = 100 * (sigma_indep - sigma_actual) / sigma_actual

    print(f"\n  σ(derived) — independence assumption: {sigma_indep:.4f}")
    print(f"  σ(derived) — with actual covariance:  {sigma_actual:.4f}")
    print(f"  Independence formula error: {pct_error:+.1f}% ({'understates' if pct_error < 0 else 'overstates'} spread)")

    # Empirical residuals for derived target
    r_home = (r1 + r2) / 2  # home_runs_residual
    r_away = (r1 - r2) / 2  # away_runs_residual
    print(f"\n  Empirical σ(home_derived residual): {r_home.std():.4f}  (should match σ_actual above)")
    print(f"  Empirical σ(away_derived residual): {r_away.std():.4f}")

    # Seasons breakdown for stationarity check
    seasons = trainable["season"].values[joint]
    print(f"\n  Per-season residual correlation (r_total, r_diff):")
    for yr in sorted(np.unique(seasons)):
        m = seasons == yr
        if m.sum() < 30:
            continue
        c = np.corrcoef(r1[m], r2[m])[0, 1]
        print(f"    {yr}: r={c:.3f}  (n={m.sum()})")

# Full-game pair
analyze_pair("total_runs", "home_run_diff",
             "total_runs", "home_run_diff",
             "Full-game: total_runs vs home_run_diff")

# First-5 pair
if "first_5_home_run_diff" in oof_keys and "first_5_total_runs" in oof_keys:
    analyze_pair("first_5_total_runs", "first_5_home_run_diff",
                 "first_5_total_runs", "first_5_home_run_diff",
                 "First-5: first_5_total_runs vs first_5_home_run_diff")
