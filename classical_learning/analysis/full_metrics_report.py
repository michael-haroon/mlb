"""
Comprehensive metrics for all component models + ensembles.
Classification: LL, AUC, Brier, ECE
Regression: MAE, RMSE, R2, Huber loss
Outputs markdown to stdout.
"""
import io, pickle, sys
sys.path.insert(0, "/Users/michaelharoon/Projects/prediction_markets/mlb")

import boto3
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import (
    log_loss, roc_auc_score, brier_score_loss,
    mean_absolute_error, mean_squared_error, r2_score
)

s3 = boto3.client("s3")
BUCKET = "mlb-265753586044-us-east-1-an"

CLF = ["extra_innings", "first_5_home_win", "home_win", "yrfi"]
REG = ["away_runs", "first_5_home_run_diff", "first_5_total_runs",
       "home_run_diff", "home_runs", "total_runs"]
ALL_TARGETS = CLF + REG

# ── helpers ───────────────────────────────────────────────────────────────────
def ece(y_true, y_pred, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(y_true)
    err = 0.0
    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() == 0:
            continue
        err += mask.sum() / total * abs(y_true[mask].mean() - y_pred[mask].mean())
    return err

def huber(y_true, y_pred, delta=1.35):
    r = y_true - y_pred
    return np.mean(np.where(np.abs(r) <= delta, 0.5 * r**2, delta * (np.abs(r) - 0.5 * delta)))

def loyo_ece(seasons, raw_oofs, iso_cals, w_arr, y_all, valid, n_bins=10):
    """Honest LOYO ECE for the calibrated ensemble blend.

    For each holdout year Y:
      1. Fit per-model isotonic on OOF[season != Y]
      2. Apply to OOF[season == Y]
      3. Blend with fixed weights
      4. Fit CalibrationBundle isotonic on calibrated blend[season != Y]
      5. Apply to calibrated blend[season == Y]
      6. Compute ECE on year Y

    Returns mean ECE across all holdout years.
    """
    from sklearn.isotonic import IsotonicRegression

    seasons_valid = seasons[valid]
    y_valid = y_all[valid]
    raw_matrix = np.column_stack([o[valid] for o in raw_oofs])

    holdout_years = [yr for yr in np.unique(seasons_valid) if yr != 2020]
    year_eces = []

    for yr in holdout_years:
        train_mask = seasons_valid != yr
        test_mask  = seasons_valid == yr

        if test_mask.sum() < 20:
            continue

        # Step 1+2: per-model isotonic fit on train, apply to test
        cal_test_cols = []
        cal_train_cols = []
        for i in range(raw_matrix.shape[1]):
            col = raw_matrix[:, i]
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(col[train_mask], y_valid[train_mask])
            cal_test_cols.append(iso.predict(col[test_mask]))
            cal_train_cols.append(iso.predict(col[train_mask]))

        cal_test  = np.column_stack(cal_test_cols)  @ w_arr
        cal_train = np.column_stack(cal_train_cols) @ w_arr

        # Step 3+4: CalibrationBundle isotonic fit on train blend, apply to test blend
        iso_bundle = IsotonicRegression(out_of_bounds="clip")
        iso_bundle.fit(np.clip(cal_train, 0.01, 0.99), y_valid[train_mask])
        p_test = iso_bundle.predict(np.clip(cal_test, 0.01, 0.99))

        year_eces.append(ece(y_valid[test_mask], p_test, n_bins=n_bins))

    return float(np.mean(year_eces)) if year_eces else float("nan")

def load_npy(key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return np.load(io.BytesIO(obj["Body"].read()))

def load_pkl(key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pickle.loads(obj["Body"].read())

# ── ground truth ──────────────────────────────────────────────────────────────
obj = s3.get_object(Bucket=BUCKET, Key="artifacts/features/game_features.parquet")
df_full = pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
trainable_all = df_full[df_full["target_status"] == "trainable"].reset_index(drop=True)

# build season mask index for full trainable (for OOF alignment pre-2020-removal)
full_train_idx = np.where((df_full["target_status"] == "trainable").values)[0]
season_full = df_full["season"].values[full_train_idx]
no2020_mask = season_full != 2020

trainable = trainable_all[trainable_all["season"] != 2020].reset_index(drop=True)

# ── list OOF keys ─────────────────────────────────────────────────────────────
paginator = s3.get_paginator("list_objects_v2")
oof_keys = {}
for page in paginator.paginate(Bucket=BUCKET, Prefix="artifacts/models/oof_"):
    for obj2 in page.get("Contents", []):
        k = obj2["Key"]
        fname = k.split("/")[-1]  # oof_<target>_<family>_A.npy
        if not fname.endswith("_A.npy"):
            continue
        stem = fname[4:-6]  # strip oof_ and _A.npy
        for t in sorted(ALL_TARGETS, key=len, reverse=True):
            if stem.startswith(t + "_"):
                family = stem[len(t)+1:]
                oof_keys.setdefault(t, {})[family] = k
                break

def align_oof(arr):
    """Align OOF array to trainable-no-2020 length."""
    n_full = len(full_train_idx)
    if len(arr) >= n_full:
        arr = arr[:n_full]
    else:
        arr = np.pad(arr, (0, n_full - len(arr)), constant_values=np.nan)
    return arr[no2020_mask]

# ── build report ──────────────────────────────────────────────────────────────
lines = []

def md_table(headers, rows):
    """Render a markdown table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    header = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    out = [header, sep]
    for row in rows:
        out.append("| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |")
    return "\n".join(out)

# ── CLASSIFICATION ────────────────────────────────────────────────────────────
lines.append("# MLB Pregame Model Metrics Report\n")
lines.append("_All metrics computed on held-out OOF predictions (2015–2026, 2020 excluded). Ensemble ECE uses LOYO calibration evaluation (calibrator never sees holdout year)._\n")
lines.append("---\n")
lines.append("## Classification Targets\n")
lines.append("**Metrics:** Log-Loss (lower=better) · AUC-ROC (higher=better) · Brier Score (lower=better) · ECE (lower=better)\n")

for target in CLF:
    lines.append(f"\n### {target.replace('_', ' ').title()}\n")
    y_all = trainable[target].values if target in trainable.columns else None
    if y_all is None:
        lines.append("_target not found_\n")
        continue

    # Load ensemble pickle for ensemble metrics + weights
    try:
        pkl = load_pkl(f"artifacts/models/ensemble_{target}_A.pkl")
    except Exception as e:
        lines.append(f"_ensemble pkl not found: {e}_\n")
        pkl = None

    rows = []
    ensemble_member_families = set()
    ensemble_weights = {}
    if pkl:
        for mb, w in zip(pkl["member_bundles"], pkl["weights"]):
            ensemble_member_families.add(mb["family"])
            ensemble_weights[mb["family"]] = w

    # Component models
    component_rows = []
    for family, key in sorted((oof_keys.get(target) or {}).items()):
        try:
            arr = align_oof(load_npy(key))
        except Exception:
            continue
        valid = ~np.isnan(y_all) & ~np.isnan(arr)
        if valid.sum() < 50:
            continue
        y, p = y_all[valid], arr[valid]
        p_clip = np.clip(p, 1e-7, 1 - 1e-7)
        try:
            auc = roc_auc_score(y, p_clip)
        except Exception:
            auc = float("nan")
        ll = log_loss(y, p_clip)
        bs = brier_score_loss(y, p_clip)
        ec = ece(y, p_clip)
        in_ens = "✓" if family in ensemble_member_families else ""
        w_str = f"{ensemble_weights.get(family, 0):.3f}" if family in ensemble_member_families else "—"
        component_rows.append((family, f"{ll:.5f}", f"{auc:.4f}", f"{bs:.5f}", f"{ec:.5f}", in_ens, w_str))

    # Sort by AUC desc
    component_rows.sort(key=lambda r: float(r[2]), reverse=True)

    headers = ["Model", "Log-Loss", "AUC-ROC", "Brier", "ECE", "In Ens", "Weight"]
    lines.append(md_table(headers, component_rows))
    lines.append("")

    # Ensemble row
    if pkl:
        member_bundles = pkl["member_bundles"]
        weights = np.array(pkl["weights"])
        cal_bundle = pkl["calibration"]

        raw_oofs, iso_cals, valid_w = [], [], []
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
            iso_cals.append(mb.get("isotonic_calibrator"))
            valid_w.append(w)

        if raw_oofs:
            w_arr = np.array(valid_w); w_arr /= w_arr.sum()
            valid = ~np.isnan(y_all)
            for o in raw_oofs:
                valid &= ~np.isnan(o)
            y = y_all[valid]
            cal_matrix = np.column_stack([
                (iso.predict(o[valid]) if iso is not None else o[valid])
                for o, iso in zip(raw_oofs, iso_cals)
            ])
            blend = cal_matrix @ w_arr
            blend_cal = cal_bundle.isotonic.predict(np.clip(blend, 0.01, 0.99))
            p_clip = np.clip(blend_cal, 1e-7, 1 - 1e-7)
            try:
                auc = roc_auc_score(y, p_clip)
            except Exception:
                auc = float("nan")
            ll = log_loss(y, p_clip)
            bs = brier_score_loss(y, p_clip)
            # Honest ECE: LOYO — calibrators fit on all years except holdout year
            seasons_arr = trainable["season"].values
            ec = loyo_ece(seasons_arr, raw_oofs, iso_cals, w_arr, y_all, valid)
            n_nonzero = sum(1 for w in pkl["weights"] if w >= 0.01)
            lines.append(f"\n**Ensemble** ({n_nonzero} non-zero members): "
                         f"Log-Loss=**{ll:.5f}** · AUC=**{auc:.4f}** · Brier=**{bs:.5f}** · ECE(LOYO)=**{ec:.5f}**\n")

# ── REGRESSION ────────────────────────────────────────────────────────────────
lines.append("\n---\n")
lines.append("## Regression Targets\n")
lines.append("**Metrics:** MAE (lower=better) · RMSE (lower=better) · R² (higher=better) · Huber Loss δ=1.35 (lower=better)\n")

for target in REG:
    lines.append(f"\n### {target.replace('_', ' ').title()}\n")
    y_all = trainable[target].values if target in trainable.columns else None
    if y_all is None:
        lines.append("_target not found_\n")
        continue

    try:
        pkl = load_pkl(f"artifacts/models/ensemble_{target}_A.pkl")
    except Exception as e:
        lines.append(f"_ensemble pkl not found: {e}_\n")
        pkl = None

    ensemble_member_families = set()
    ensemble_weights = {}
    if pkl:
        for mb, w in zip(pkl["member_bundles"], pkl["weights"]):
            ensemble_member_families.add(mb["family"])
            ensemble_weights[mb["family"]] = w

    component_rows = []
    for family, key in sorted((oof_keys.get(target) or {}).items()):
        try:
            arr = align_oof(load_npy(key))
        except Exception:
            continue
        valid = ~np.isnan(y_all) & ~np.isnan(arr)
        if valid.sum() < 50:
            continue
        y, p = y_all[valid], arr[valid]
        mae = mean_absolute_error(y, p)
        rmse = np.sqrt(mean_squared_error(y, p))
        r2 = r2_score(y, p)
        hl = huber(y, p)
        in_ens = "✓" if family in ensemble_member_families else ""
        w_str = f"{ensemble_weights.get(family, 0):.3f}" if family in ensemble_member_families else "—"
        component_rows.append((family, f"{mae:.4f}", f"{rmse:.4f}", f"{r2:.4f}", f"{hl:.4f}", in_ens, w_str))

    # Sort by MAE asc
    component_rows.sort(key=lambda r: float(r[1]))

    headers = ["Model", "MAE", "RMSE", "R²", "Huber", "In Ens", "Weight"]
    lines.append(md_table(headers, component_rows))
    lines.append("")

    # Ensemble
    if pkl:
        member_bundles = pkl["member_bundles"]
        weights = np.array(pkl["weights"])

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

        if raw_oofs:
            w_arr = np.array(valid_w); w_arr /= w_arr.sum()
            valid = ~np.isnan(y_all)
            for o in raw_oofs:
                valid &= ~np.isnan(o)
            y = y_all[valid]
            blend = np.column_stack([o[valid] for o in raw_oofs]) @ w_arr
            mae = mean_absolute_error(y, blend)
            rmse = np.sqrt(mean_squared_error(y, blend))
            r2 = r2_score(y, blend)
            hl = huber(y, blend)
            n_nonzero = sum(1 for w in pkl["weights"] if w >= 0.01)
            lines.append(f"\n**Ensemble** ({n_nonzero} non-zero members): "
                         f"MAE=**{mae:.4f}** · RMSE=**{rmse:.4f}** · R²=**{r2:.4f}** · Huber=**{hl:.4f}**\n")

print("\n".join(lines))
