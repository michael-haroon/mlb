"""
Per-season OOF residual correlation between total_runs and home_run_diff (full-game and F5).
Shows correlation stability and the σ-derived impact of independence assumption vs actual covariance.
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
    pkl = load_pkl(f"artifacts/models/ensemble_{target}_A.pkl")
    member_bundles, weights = pkl["member_bundles"], np.array(pkl["weights"])
    raw_oofs, valid_w = [], []
    for mb, w in zip(member_bundles, weights):
        if w < 1e-6: continue
        key2 = (oof_keys.get(target) or {}).get(mb["family"])
        if not key2: continue
        try: arr = align_oof(load_npy(key2))
        except Exception: continue
        raw_oofs.append(arr); valid_w.append(w)
    if not raw_oofs: return None
    w_arr = np.array(valid_w); w_arr /= w_arr.sum()
    matrix = np.column_stack(raw_oofs)
    blend = matrix @ w_arr
    blend[np.isnan(matrix).any(axis=1)] = np.nan
    return blend

def season_table(t_tr, t_rd, col_tr, col_rd, label):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  {'Season':>6}  {'n':>5}  {'corr(r_tr,r_rd)':>16}  {'σ_indep':>8}  {'σ_actual':>9}  {'Δ%':>6}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*16}  {'-'*8}  {'-'*9}  {'-'*6}")

    b_tr = ensemble_blend_array(t_tr)
    b_rd = ensemble_blend_array(t_rd)
    y_tr = trainable[col_tr].values
    y_rd = trainable[col_rd].values
    seasons = trainable["season"].values

    joint = ~np.isnan(b_tr) & ~np.isnan(b_rd) & ~np.isnan(y_tr) & ~np.isnan(y_rd)
    r_tr_all = y_tr[joint] - b_tr[joint]
    r_rd_all = y_rd[joint] - b_rd[joint]
    seas_all  = seasons[joint]

    season_corrs = []
    for yr in sorted(np.unique(seas_all)):
        m = seas_all == yr
        n = m.sum()
        if n < 30: continue
        r1, r2 = r_tr_all[m], r_rd_all[m]
        corr = np.corrcoef(r1, r2)[0, 1]
        cov  = np.cov(r1, r2)[0, 1]
        v1, v2 = r1.var(), r2.var()
        sig_indep  = np.sqrt((v1 + v2) / 4)
        sig_actual = np.sqrt((v1 + v2 + 2 * cov) / 4)
        delta_pct  = 100 * (sig_indep - sig_actual) / sig_actual
        season_corrs.append(corr)
        print(f"  {yr:>6}  {n:>5}  {corr:>+16.3f}  {sig_indep:>8.4f}  {sig_actual:>9.4f}  {delta_pct:>+5.1f}%")

    # Summary
    arr = np.array(season_corrs)
    print(f"\n  Pooled corr:  {np.corrcoef(r_tr_all, r_rd_all)[0,1]:+.3f}")
    print(f"  Season range: [{arr.min():+.3f}, {arr.max():+.3f}]  "
          f"std={arr.std():.3f}  sign-flips={int((arr[:-1] * arr[1:] < 0).sum())}/{len(arr)-1} consecutive pairs")

season_table("total_runs", "home_run_diff",
             "total_runs", "home_run_diff",
             "Full-game: corr(r_total_runs, r_home_run_diff) by season")

season_table("first_5_total_runs", "first_5_home_run_diff",
             "first_5_total_runs", "first_5_home_run_diff",
             "First-5: corr(r_f5_total, r_f5_home_run_diff) by season")
