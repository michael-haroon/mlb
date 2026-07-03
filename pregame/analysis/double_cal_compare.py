"""
Compare three post-blend calibration scenarios on OOF data:
  A. Current (broken): raw blend → existing CalibrationBundle isotonic
  B. Drop bundle:      per-model-isotonic blend → no post-blend calibration
  C. Refit bundle:     per-model-isotonic blend → new isotonic fit on calibrated blend

Only classification targets are compared (regression uses Student-t, not affected).
"""
import io, pickle, sys
sys.path.insert(0, "/Users/michaelharoon/Projects/prediction_markets/mlb")
import boto3
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression

s3 = boto3.client("s3")
BUCKET = "mlb-265753586044-us-east-1-an"
CLF_TARGETS = ["extra_innings", "first_5_home_win", "home_win", "yrfi"]

# ── ECE helper ────────────────────────────────────────────────────────────────
def ece(y_true, y_pred, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(y_true)
    err = 0.0
    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_pred[mask].mean()
        err += mask.sum() / total * abs(acc - conf)
    return err

# ── Metrics helper ─────────────────────────────────────────────────────────────
def metrics(y_true, y_pred, label):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    ll = log_loss(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    bs = brier_score_loss(y_true, y_pred)
    ec = ece(y_true, y_pred)
    print(f"    {label:<25} LL={ll:.5f}  AUC={auc:.4f}  Brier={bs:.5f}  ECE={ec:.5f}")
    return {"log_loss": ll, "auc": auc, "brier": bs, "ece": ec}

# ── Load helpers ───────────────────────────────────────────────────────────────
def load_pkl(key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pickle.loads(obj["Body"].read())

def load_npy(key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return np.load(io.BytesIO(obj["Body"].read()))

# ── Load ground truth from parquet ────────────────────────────────────────────
import pyarrow.parquet as pq
obj = s3.get_object(Bucket=BUCKET, Key="artifacts/features/game_features.parquet")
df = pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
trainable = df[df["target_status"] == "trainable"].reset_index(drop=True)
# Exclude 2020 — same filter as ensemble pipeline
mask_2020 = trainable["season"] != 2020
trainable = trainable[mask_2020].reset_index(drop=True)

print("=" * 72)
print("DOUBLE-CALIBRATION COMPARISON: raw-blend vs drop-bundle vs refit-bundle")
print("=" * 72)

summary = {}

for target in CLF_TARGETS:
    print(f"\n{'─'*72}")
    print(f"  {target.upper()}")
    print(f"{'─'*72}")

    pkl = load_pkl(f"artifacts/models/ensemble_{target}_A.pkl")
    member_bundles = pkl["member_bundles"]
    weights = np.array(pkl["weights"])
    cal_bundle = pkl["calibration"]  # existing CalibrationBundle (fitted on raw blend)

    y_all = trainable[target].values if target in trainable.columns else None
    if y_all is None:
        print(f"  SKIP: {target} not in trainable columns")
        continue

    n = len(y_all)

    # ── Load raw OOF per member ────────────────────────────────────────────────
    raw_oofs = []
    iso_cals = []
    valid_weights = []

    for mb, w in zip(member_bundles, weights):
        if w < 1e-6:
            continue  # skip zero-weight members
        family = mb["family"]
        iso_cal = mb.get("isotonic_calibrator")

        key = f"artifacts/models/oof_{target}_{family}_A.npy"
        try:
            arr = load_npy(key)
        except Exception as e:
            print(f"  WARNING: could not load OOF for {family}: {e}")
            continue

        # Align length to trainable (excluding 2020)
        # The OOF arrays were saved before 2020 masking — need same mask
        full_mask = (df["target_status"] == "trainable").values
        full_idx = np.where(full_mask)[0]
        season_full = df["season"].values[full_idx]
        no2020 = season_full != 2020

        if len(arr) >= len(full_idx):
            arr_aligned = arr[:len(full_idx)][no2020]
        else:
            arr_padded = np.pad(arr, (0, len(full_idx) - len(arr)), constant_values=np.nan)
            arr_aligned = arr_padded[no2020]

        raw_oofs.append(arr_aligned)
        iso_cals.append(iso_cal)
        valid_weights.append(w)

    if not raw_oofs:
        print(f"  SKIP: no OOF arrays loaded")
        continue

    w_arr = np.array(valid_weights)
    w_arr = w_arr / w_arr.sum()

    # ── Build valid mask (no NaN in y or any OOF) ─────────────────────────────
    valid = ~np.isnan(y_all)
    for oof in raw_oofs:
        valid &= ~np.isnan(oof)

    y = y_all[valid]
    raw_matrix = np.column_stack([o[valid] for o in raw_oofs])

    # ── Per-model isotonic calibrated OOFs ───────────────────────────────────
    cal_matrix = np.zeros_like(raw_matrix)
    for i, (iso_cal, raw_col) in enumerate(zip(iso_cals, raw_oofs)):
        col = raw_col[valid]
        if iso_cal is not None:
            cal_matrix[:, i] = iso_cal.predict(col)
        else:
            cal_matrix[:, i] = col  # no per-model cal (shouldn't happen for clf)

    # ── Scenario A: Current (broken) ──────────────────────────────────────────
    # cli.py blended raw OOFs, fitted CalibrationBundle on that
    # predict.py now applies per-model isotonic THEN post-blend cal_bundle
    # → cal_bundle receives calibrated blend but was fitted on raw blend
    raw_blend = raw_matrix @ w_arr
    cal_blend = cal_matrix @ w_arr

    # What actually happens at inference (the bug):
    # per-model isotonic applied → calibrated blend → cal_bundle (fitted on raw blend)
    current_output = cal_bundle.isotonic.predict(np.clip(cal_blend, 0.01, 0.99))

    # ── Scenario B: Drop bundle ────────────────────────────────────────────────
    # per-model isotonic applied → calibrated blend → no post-blend step
    drop_output = cal_blend

    # ── Scenario C: Refit bundle ───────────────────────────────────────────────
    # per-model isotonic applied → calibrated blend → NEW isotonic fit on calibrated blend
    iso_refit = IsotonicRegression(out_of_bounds="clip")
    iso_refit.fit(cal_blend, y)
    refit_output = iso_refit.predict(np.clip(cal_blend, 0.01, 0.99))

    # ── Also show what raw blend → cal_bundle gives (the "intended" behavior) ──
    raw_then_cal = cal_bundle.isotonic.predict(np.clip(raw_blend, 0.01, 0.99))

    print(f"  n_valid={valid.sum()}  n_members={len(valid_weights)}")
    print()
    r_a = metrics(y, current_output, "A: current (broken)")
    r_raw = metrics(y, raw_then_cal, "   raw blend + cal_bundle")
    r_b = metrics(y, drop_output, "B: drop bundle")
    r_c = metrics(y, refit_output, "C: refit bundle")

    # Best on log_loss
    best_ll = min(r_a["log_loss"], r_b["log_loss"], r_c["log_loss"])
    winner = "A" if r_a["log_loss"] == best_ll else ("B" if r_b["log_loss"] == best_ll else "C")
    print(f"\n  >> Winner on LL: {winner}  (best={best_ll:.5f})")
    print(f"     A vs B delta LL: {r_a['log_loss'] - r_b['log_loss']:+.5f}")
    print(f"     A vs C delta LL: {r_a['log_loss'] - r_c['log_loss']:+.5f}")
    print(f"     B vs C delta LL: {r_b['log_loss'] - r_c['log_loss']:+.5f}")

    summary[target] = {"A": r_a, "B": r_b, "C": r_c, "winner": winner}

print(f"\n{'='*72}")
print("SUMMARY TABLE  (lower LL = better)")
print(f"{'='*72}")
print(f"{'Target':<25} {'A (current)':<14} {'B (drop)':<14} {'C (refit)':<14} {'Winner'}")
print(f"{'-'*72}")
for t, s in summary.items():
    print(f"{t:<25} {s['A']['log_loss']:.5f}        {s['B']['log_loss']:.5f}        {s['C']['log_loss']:.5f}        {s['winner']}")

print()
print("ECE SUMMARY  (lower = better calibration)")
print(f"{'-'*72}")
print(f"{'Target':<25} {'A ECE':<14} {'B ECE':<14} {'C ECE':<14}")
print(f"{'-'*72}")
for t, s in summary.items():
    print(f"{t:<25} {s['A']['ece']:.5f}        {s['B']['ece']:.5f}        {s['C']['ece']:.5f}")
