from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class BinaryMetrics:
    target_name: str
    n_samples: int
    brier_score: float
    log_loss: float
    auc_roc: float
    accuracy: float
    mean_predicted: float
    mean_actual: float
    calibration_bins: list[dict] = field(default_factory=list)
    ece: float = 0.0

    def to_dict(self) -> dict:
        return {
            "target_name": self.target_name,
            "n_samples": self.n_samples,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "auc_roc": self.auc_roc,
            "accuracy": self.accuracy,
            "mean_predicted": self.mean_predicted,
            "mean_actual": self.mean_actual,
            "ece": self.ece,
            "calibration_bins": self.calibration_bins,
        }


@dataclass
class GaussianMetrics:
    target_name: str
    n_samples: int
    mae: float
    rmse: float
    mean_nll: float
    coverage_50: float
    coverage_90: float
    mean_sigma: float
    crps: float

    def to_dict(self) -> dict:
        return {
            "target_name": self.target_name,
            "n_samples": self.n_samples,
            "mae": self.mae,
            "rmse": self.rmse,
            "mean_nll": self.mean_nll,
            "coverage_50": self.coverage_50,
            "coverage_90": self.coverage_90,
            "mean_sigma": self.mean_sigma,
            "crps": self.crps,
        }


@dataclass
class SeasonReport:
    season: int
    n_games: int
    binary_metrics: list[BinaryMetrics] = field(default_factory=list)
    gaussian_metrics: list[GaussianMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "season": self.season,
            "n_games": self.n_games,
            "binary": [m.to_dict() for m in self.binary_metrics],
            "gaussian": [m.to_dict() for m in self.gaussian_metrics],
        }


def compute_binary_metrics(
    actuals: np.ndarray,
    predicted_probs: np.ndarray,
    target_name: str,
    n_bins: int = 10,
) -> BinaryMetrics:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    actuals = actuals.astype("float64")
    predicted_probs = np.clip(predicted_probs.astype("float64"), 1e-7, 1 - 1e-7)
    n = len(actuals)

    brier = float(brier_score_loss(actuals, predicted_probs))
    ll = float(log_loss(actuals, predicted_probs))

    unique_classes = np.unique(actuals)
    if len(unique_classes) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(actuals, predicted_probs))

    preds_binary = (predicted_probs >= 0.5).astype("float64")
    accuracy = float((preds_binary == actuals).mean())

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    calibration_bins = []
    ece_sum = 0.0
    for i in range(n_bins):
        mask = (predicted_probs >= bins[i]) & (predicted_probs < bins[i + 1])
        if i == n_bins - 1:
            mask = mask | (predicted_probs == bins[i + 1])
        bin_count = int(mask.sum())
        if bin_count == 0:
            continue
        bin_mean_pred = float(predicted_probs[mask].mean())
        bin_mean_actual = float(actuals[mask].mean())
        calibration_bins.append({
            "bin_lower": float(bins[i]),
            "bin_upper": float(bins[i + 1]),
            "count": bin_count,
            "mean_predicted": bin_mean_pred,
            "mean_actual": bin_mean_actual,
            "abs_error": abs(bin_mean_pred - bin_mean_actual),
        })
        ece_sum += bin_count * abs(bin_mean_pred - bin_mean_actual)

    ece = ece_sum / max(n, 1)

    return BinaryMetrics(
        target_name=target_name,
        n_samples=n,
        brier_score=brier,
        log_loss=ll,
        auc_roc=auc,
        accuracy=accuracy,
        mean_predicted=float(predicted_probs.mean()),
        mean_actual=float(actuals.mean()),
        calibration_bins=calibration_bins,
        ece=ece,
    )


def compute_gaussian_metrics(
    actuals: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    target_name: str,
) -> GaussianMetrics:
    from scipy.stats import norm

    actuals = actuals.astype("float64")
    mu = mu.astype("float64")
    sigma = np.clip(sigma.astype("float64"), 1e-6, None)
    n = len(actuals)

    residuals = actuals - mu
    mae = float(np.abs(residuals).mean())
    rmse = float(np.sqrt((residuals ** 2).mean()))

    nll = 0.5 * ((residuals / sigma) ** 2 + 2 * np.log(sigma) + np.log(2 * np.pi))
    mean_nll = float(nll.mean())

    z_50 = 0.6745
    z_90 = 1.6449
    coverage_50 = float(((actuals >= mu - z_50 * sigma) & (actuals <= mu + z_50 * sigma)).mean())
    coverage_90 = float(((actuals >= mu - z_90 * sigma) & (actuals <= mu + z_90 * sigma)).mean())

    z = residuals / sigma
    crps_per = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    crps = float(crps_per.mean())

    return GaussianMetrics(
        target_name=target_name,
        n_samples=n,
        mae=mae,
        rmse=rmse,
        mean_nll=mean_nll,
        coverage_50=coverage_50,
        coverage_90=coverage_90,
        mean_sigma=float(sigma.mean()),
        crps=crps,
    )


def evaluate_pregame_model(model, loader, device, game_targets_df) -> list[SeasonReport]:
    """Run the pregame model over a DataLoader and produce per-season metrics."""

    import torch

    model.eval()
    all_preds = {
        "home_win_logit": [], "yrfi_logit": [],
        "total_runs_mu": [], "total_runs_sigma": [],
        "home_run_diff_mu": [], "home_run_diff_sigma": [],
    }
    all_targets = {"home_win": [], "yrfi": [], "total_runs": [], "home_run_diff": []}
    all_game_pks = []

    with torch.no_grad():
        for batch in loader:
            batch_dev = _to_device(batch, device)
            preds = model(batch_dev)
            for key in all_preds:
                all_preds[key].append(preds[key].cpu().numpy())
            for key in all_targets:
                all_targets[key].append(batch["targets"][key].numpy())
            all_game_pks.append(batch["game_pk"].numpy())

    for key in all_preds:
        all_preds[key] = np.concatenate(all_preds[key])
    for key in all_targets:
        all_targets[key] = np.concatenate(all_targets[key])
    all_game_pks = np.concatenate(all_game_pks)

    import pandas as pd
    pk_to_season = dict(zip(
        game_targets_df["game_pk"].astype("int64"),
        game_targets_df["season"].astype("int64"),
    ))
    seasons = np.array([pk_to_season.get(int(pk), -1) for pk in all_game_pks])
    unique_seasons = sorted(set(seasons) - {-1})

    reports = []
    for season in unique_seasons:
        mask = seasons == season
        n_games = int(mask.sum())
        if n_games == 0:
            continue

        report = SeasonReport(season=int(season), n_games=n_games)

        home_win_probs = _sigmoid(all_preds["home_win_logit"][mask])
        report.binary_metrics.append(
            compute_binary_metrics(all_targets["home_win"][mask], home_win_probs, "home_win")
        )

        yrfi_probs = _sigmoid(all_preds["yrfi_logit"][mask])
        report.binary_metrics.append(
            compute_binary_metrics(all_targets["yrfi"][mask], yrfi_probs, "yrfi")
        )

        report.gaussian_metrics.append(
            compute_gaussian_metrics(
                all_targets["total_runs"][mask],
                all_preds["total_runs_mu"][mask],
                all_preds["total_runs_sigma"][mask],
                "total_runs",
            )
        )
        report.gaussian_metrics.append(
            compute_gaussian_metrics(
                all_targets["home_run_diff"][mask],
                all_preds["home_run_diff_mu"][mask],
                all_preds["home_run_diff_sigma"][mask],
                "home_run_diff",
            )
        )

        reports.append(report)

    return reports


def save_evaluation(reports: list[SeasonReport], output_dir: str) -> str:
    """Write evaluation JSON for later plotting/analysis."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "evaluation.json"
    data = {
        "seasons": [r.to_dict() for r in reports],
        "summary": _build_summary(reports),
    }
    path.write_text(json.dumps(data, indent=2, default=_json_default))
    return str(path)


def _build_summary(reports: list[SeasonReport]) -> dict:
    """Aggregate across all seasons for a quick overall view."""
    all_binary = {}
    all_gaussian = {}

    for r in reports:
        for m in r.binary_metrics:
            all_binary.setdefault(m.target_name, []).append(m)
        for m in r.gaussian_metrics:
            all_gaussian.setdefault(m.target_name, []).append(m)

    summary = {"binary": {}, "gaussian": {}}

    for name, metrics_list in all_binary.items():
        total_n = sum(m.n_samples for m in metrics_list)
        summary["binary"][name] = {
            "n_samples": total_n,
            "mean_brier": _weighted_mean([m.brier_score for m in metrics_list], [m.n_samples for m in metrics_list]),
            "mean_log_loss": _weighted_mean([m.log_loss for m in metrics_list], [m.n_samples for m in metrics_list]),
            "mean_auc": _weighted_mean([m.auc_roc for m in metrics_list], [m.n_samples for m in metrics_list]),
            "mean_accuracy": _weighted_mean([m.accuracy for m in metrics_list], [m.n_samples for m in metrics_list]),
            "mean_ece": _weighted_mean([m.ece for m in metrics_list], [m.n_samples for m in metrics_list]),
        }

    for name, metrics_list in all_gaussian.items():
        total_n = sum(m.n_samples for m in metrics_list)
        summary["gaussian"][name] = {
            "n_samples": total_n,
            "mean_mae": _weighted_mean([m.mae for m in metrics_list], [m.n_samples for m in metrics_list]),
            "mean_rmse": _weighted_mean([m.rmse for m in metrics_list], [m.n_samples for m in metrics_list]),
            "mean_nll": _weighted_mean([m.mean_nll for m in metrics_list], [m.n_samples for m in metrics_list]),
            "mean_coverage_50": _weighted_mean([m.coverage_50 for m in metrics_list], [m.n_samples for m in metrics_list]),
            "mean_coverage_90": _weighted_mean([m.coverage_90 for m in metrics_list], [m.n_samples for m in metrics_list]),
            "mean_crps": _weighted_mean([m.crps for m in metrics_list], [m.n_samples for m in metrics_list]),
        }

    return summary


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _weighted_mean(values: list, weights: list) -> float:
    values = [v for v, w in zip(values, weights) if np.isfinite(v)]
    weights = [w for v, w in zip(values, weights) if np.isfinite(v)]
    total_w = sum(weights)
    if total_w == 0:
        return float("nan")
    return sum(v * w for v, w in zip(values, weights)) / total_w


def _to_device(value, device):
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_device(v, device) for v in value]
    return value


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
