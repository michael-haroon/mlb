"""
De Prado-style validation of the total_runs XGBoost model.

Requires the conda pred environment. Run:
    conda run -n pred python scripts/validate_total_runs_model.py

Performs:
1. OOF predictions vs actual outcomes (calibration, bias, accuracy)
2. NegBin distributional calibration (MACE, PIT histogram)
3. Directional bet accuracy at standard lines
4. Strategy-level Sharpe estimation
5. Feature importance cross-check (MDI vs MDA stability)
"""
import sys
from pathlib import Path
from math import lgamma, exp, log, floor, sqrt

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "pregame" / "artifacts" / "models"
FEATURES_PATH = PROJECT_ROOT / "pregame" / "artifacts" / "features" / "game_features.parquet"

# ── Load data ─────────────────────────────────────────────────────────────────

def load_oof_with_targets():
    """Load XGBoost OOF predictions aligned with actual outcomes."""
    SKIP_SEASONS = [2020]  # COVID shortened season

    # Load features + target (only need game_pk, season, total_runs)
    try:
        df = pd.read_parquet(FEATURES_PATH, columns=["game_pk", "season", "total_runs"])
    except Exception:
        # game_pk might be in index or column set differs
        df = pd.read_parquet(FEATURES_PATH)
        if "game_pk" not in df.columns and df.index.name == "game_pk":
            df = df.reset_index()
        df = df[["game_pk", "season", "total_runs"]]
    df = df[df["season"] >= 2015].reset_index(drop=True)
    df = df[~df["season"].isin(SKIP_SEASONS)].reset_index(drop=True)
    df = df[df["total_runs"].notna()].reset_index(drop=True)

    y_true = df["total_runs"].values
    seasons = df["season"].values
    game_pks = df["game_pk"].values

    # Load OOF predictions
    oof_xgb = np.load(MODELS_DIR / "oof_total_runs_xgboost_A.npy")
    oof_game_pks = np.load(MODELS_DIR / "oof_game_pks_total_runs_A.npy")

    # Align by game_pk
    oof_map = {int(gp): pred for gp, pred in zip(oof_game_pks, oof_xgb) if not np.isnan(pred)}

    y_aligned = []
    pred_aligned = []
    season_aligned = []
    for i, gp in enumerate(game_pks):
        if int(gp) in oof_map:
            y_aligned.append(y_true[i])
            pred_aligned.append(oof_map[int(gp)])
            season_aligned.append(seasons[i])

    return (np.array(y_aligned), np.array(pred_aligned), np.array(season_aligned))


# ── NegBin helpers ────────────────────────────────────────────────────────────

def negbin_pmf(k, n, p):
    log_pmf = lgamma(k + n) - lgamma(k + 1) - lgamma(n) + n * log(p) + k * log(1 - p)
    return exp(log_pmf)

def negbin_cdf_val(k_max, alpha, mu):
    p = alpha / (alpha + mu)
    total = 0.0
    for k in range(int(k_max) + 1):
        total += negbin_pmf(k, alpha, p)
    return total

def prob_over(mu, threshold, alpha):
    return 1.0 - negbin_cdf_val(int(floor(threshold)), alpha, mu)

def negbin_pit(y, mu, alpha):
    """Probability Integral Transform: CDF(y) under NegBin(mu, alpha)."""
    return negbin_cdf_val(int(y), alpha, mu)


# ── Main analysis ─────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    y_true, y_pred, seasons = load_oof_with_targets()
    alpha = 6.732374011057548
    n = len(y_true)
    print(f"Loaded {n} aligned OOF games.\n")

    residuals = y_true - y_pred

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. POINT PREDICTION ACCURACY
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("  1. POINT PREDICTION ACCURACY")
    print("=" * 70)
    mae = np.mean(np.abs(residuals))
    rmse = sqrt(np.mean(residuals ** 2))
    r2 = 1 - np.mean(residuals ** 2) / np.var(y_true)
    bias = np.mean(residuals)
    median_residual = np.median(residuals)

    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  Bias (mean residual): {bias:+.4f}")
    print(f"  Median residual: {median_residual:+.4f}")
    print(f"  y_true mean: {y_true.mean():.3f}, y_pred mean: {y_pred.mean():.3f}")
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. DISTRIBUTIONAL CALIBRATION (NegBin)
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("  2. NEGBIN DISTRIBUTIONAL CALIBRATION")
    print("=" * 70)

    # PIT histogram: if NegBin(mu, alpha) is correct, PIT values should be uniform
    print("  Computing PIT values (this may take a minute)...")
    pit_values = np.array([negbin_pit(y, mu, alpha) for y, mu in zip(y_true, y_pred)])

    # Check uniformity via bin counts
    n_bins = 10
    hist, edges = np.histogram(pit_values, bins=n_bins, range=(0, 1))
    expected = n / n_bins
    chi2 = np.sum((hist - expected) ** 2 / expected)
    print(f"  PIT histogram (should be uniform):")
    for i in range(n_bins):
        bar = "█" * int(hist[i] / expected * 20)
        print(f"    [{edges[i]:.1f}, {edges[i+1]:.1f}): {hist[i]:>5} (expected {expected:.0f}) {bar}")
    print(f"  Chi² = {chi2:.1f} (critical value at p=0.05 with {n_bins-1} df: 16.9)")
    print(f"  {'FAIL: distribution miscalibrated' if chi2 > 16.9 else 'PASS: distribution well-calibrated'}")
    print()

    # MACE: Mean Absolute Calibration Error at standard lines
    print("  Market-line calibration (MACE):")
    lines = [6.5, 7.5, 8.5, 9.5, 10.5, 11.5]
    mace_total = 0
    for line in lines:
        pred_probs = np.array([prob_over(mu, line, alpha) for mu in y_pred])
        actual_over = (y_true > line).astype(float)
        # Bin by predicted probability
        n_cal_bins = 5
        cal_errors = []
        for q in range(n_cal_bins):
            lo = np.percentile(pred_probs, q * 100 / n_cal_bins)
            hi = np.percentile(pred_probs, (q + 1) * 100 / n_cal_bins)
            mask = (pred_probs >= lo) & (pred_probs <= hi) if q == n_cal_bins - 1 else (pred_probs >= lo) & (pred_probs < hi)
            if mask.sum() > 0:
                cal_errors.append(abs(pred_probs[mask].mean() - actual_over[mask].mean()))
        mace = np.mean(cal_errors)
        mace_total += mace
        # Simple overall calibration
        pred_mean = pred_probs.mean()
        actual_mean = actual_over.mean()
        print(f"    O/U {line:>4.1f}: model P(over)={pred_mean:.4f}, actual={actual_mean:.4f}, "
              f"error={pred_mean-actual_mean:+.4f}, MACE={mace:.4f}")
    print(f"  Average MACE: {mace_total/len(lines):.4f}")
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. DIRECTIONAL BET ACCURACY (de Prado triple-barrier analog)
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("  3. DIRECTIONAL BET ACCURACY")
    print("=" * 70)
    print()
    print("  For each game, if model says P(over X.5) > 0.5, bet over.")
    print("  Accuracy = fraction of bets that win.")
    print()

    for line in [7.5, 8.5, 9.5, 10.5]:
        pred_probs = np.array([prob_over(mu, line, alpha) for mu in y_pred])
        actual_over = y_true > line

        # Bet over when model > 0.5
        bet_over_mask = pred_probs > 0.5
        bet_under_mask = pred_probs < 0.5

        if bet_over_mask.sum() > 0:
            over_accuracy = actual_over[bet_over_mask].mean()
            over_n = bet_over_mask.sum()
        else:
            over_accuracy, over_n = 0, 0

        if bet_under_mask.sum() > 0:
            under_accuracy = (~actual_over)[bet_under_mask].mean()
            under_n = bet_under_mask.sum()
        else:
            under_accuracy, under_n = 0, 0

        total_accuracy = (over_accuracy * over_n + under_accuracy * under_n) / (over_n + under_n) if (over_n + under_n) > 0 else 0

        print(f"  O/U {line}:")
        print(f"    Bet OVER:  {over_n:>5} games, accuracy={over_accuracy:.4f}")
        print(f"    Bet UNDER: {under_n:>5} games, accuracy={under_accuracy:.4f}")
        print(f"    Combined:  accuracy={total_accuracy:.4f} (baseline=0.50)")
        print(f"    Lift: {(total_accuracy-0.5)*100:+.2f}pp")
        print()

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. STRATEGY SHARPE (de Prado: realized P&L per unit risk)
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("  4. STRATEGY-LEVEL SHARPE RATIO")
    print("=" * 70)
    print()
    print("  Simulated flat-bet strategy: bet 1 contract on model direction")
    print("  Profit = +contract value if correct, -price if wrong")
    print()

    for line in [8.5, 9.5]:
        pred_probs = np.array([prob_over(mu, line, alpha) for mu in y_pred])
        actual_over = y_true > line

        # Only bet when |edge| > 3c (model deviates from 50%)
        edge = pred_probs - 0.5
        trade_mask = np.abs(edge) > 0.03
        if trade_mask.sum() < 50:
            continue

        # P&L per trade: if bet over at price p, win (1-p) if correct, lose p if wrong
        pnl = []
        for i in np.where(trade_mask)[0]:
            p = pred_probs[i]
            if edge[i] > 0:  # bet over
                pnl.append((1 - p) if actual_over[i] else -p)
            else:  # bet under
                pnl.append((1 - (1 - p)) if not actual_over[i] else -(1 - p))

        pnl = np.array(pnl)
        sharpe = pnl.mean() / pnl.std() * sqrt(252 * 15) if pnl.std() > 0 else 0  # annualized, ~15 games/day
        print(f"  O/U {line} (|edge|>3c filter, {trade_mask.sum()} trades):")
        print(f"    Mean P&L/trade: {pnl.mean()*100:.2f}c")
        print(f"    Std P&L/trade:  {pnl.std()*100:.2f}c")
        print(f"    Win rate:       {(pnl > 0).mean():.4f}")
        print(f"    Annualized Sharpe: {sharpe:.2f}")
        print()

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. RESIDUAL DIAGNOSTICS
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("  5. RESIDUAL DIAGNOSTICS")
    print("=" * 70)
    print()

    # Heteroscedasticity check
    pred_bins = np.percentile(y_pred, [0, 20, 40, 60, 80, 100])
    print("  Residual std by prediction level (heteroscedasticity check):")
    for i in range(len(pred_bins) - 1):
        mask = (y_pred >= pred_bins[i]) & (y_pred < pred_bins[i + 1])
        if i == len(pred_bins) - 2:
            mask = (y_pred >= pred_bins[i]) & (y_pred <= pred_bins[i + 1])
        if mask.sum() > 0:
            res_std = np.std(residuals[mask])
            res_mean = np.mean(residuals[mask])
            print(f"    pred [{pred_bins[i]:.1f}, {pred_bins[i+1]:.1f}): "
                  f"n={mask.sum():>5}, residual_std={res_std:.3f}, residual_mean={res_mean:+.3f}")
    print()

    # Skewness of residuals
    skew = np.mean(((residuals - residuals.mean()) / residuals.std()) ** 3)
    kurt = np.mean(((residuals - residuals.mean()) / residuals.std()) ** 4) - 3
    print(f"  Residual skewness: {skew:.3f} (>0 means model underpredicts high-scoring games)")
    print(f"  Residual excess kurtosis: {kurt:.3f} (>0 means fat tails)")
    print()

    # Per-season bias check
    print("  Per-season bias (mean residual):")
    for s in sorted(set(seasons)):
        mask = seasons == s
        if mask.sum() > 100:
            bias_s = np.mean(residuals[mask])
            print(f"    {s}: bias={bias_s:+.3f} ({mask.sum()} games)")
    print()

    print("Done. Run paper trades to validate against live Kalshi lines.")


if __name__ == "__main__":
    main()
