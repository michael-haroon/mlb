"""
Workflow 1: EDA on raw PITCHES table from S3.

Analyses:
1. Missing value heatmap per column by season
2. Row counts, games, anomalies
3. Physics distributions (velocity, spin, trajectory, etc.)
4. Categorical cardinality and trends
5. Temporal patterns (inning fatigue, count distributions)
6. Seasonal drift (KS distance matrices)
7. Correlation structure
8. Timestamp quality
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .data_sources import ParquetCatalog, season_range

log = logging.getLogger(__name__)


def _setup_logging(output_dir: Path) -> None:
    if log.handlers:
        return
    log.setLevel(logging.DEBUG)

    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_dir / "raw_eda.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
    log.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(sh)


def run_raw_eda(source_uri: str, output_dir: str, seasons: list[int] | None = None) -> dict:
    """Orchestrate raw PITCHES EDA."""
    import numpy as np
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _setup_logging(output_dir)
    t0 = time.time()

    log.info("Starting raw PITCHES EDA — source=%s, output=%s", source_uri, output_dir)

    catalog = ParquetCatalog(source_uri)

    # Load raw pitches
    log.info("Loading PITCHES table ...")
    pitches = catalog.read_table("pitches", seasons=seasons)
    log.info("Loaded %d pitch rows across %d games", len(pitches), pitches["game_pk"].nunique() if len(pitches) > 0 else 0)

    if len(pitches) == 0:
        log.warning("No pitch data found")
        return {"error": "no_data", "output_dir": str(output_dir)}

    # Convert numeric columns to float64 for analysis
    num_cols = pitches.select_dtypes(include=["number"]).columns
    pitches[num_cols] = pitches[num_cols].astype("float64")

    # --- Analysis 1: Missing value heatmap by season ---
    log.info("Analyzing missing values by season ...")
    missing_by_season = compute_missing_heatmap_data(pitches)

    # --- Analysis 2: Row counts, games, anomalies ---
    log.info("Analyzing row counts and game anomalies ...")
    game_stats = analyze_game_counts(pitches)

    # --- Analysis 3: Physics distributions ---
    log.info("Computing physics distribution stats ...")
    physics_cols = [
        "release_speed", "spin_rate", "pfx_x", "pfx_z",
        "extension", "launch_angle", "exit_velocity"
    ]
    physics_cols = [c for c in physics_cols if c in pitches.columns]
    physics_stats = compute_column_stats_batch(pitches, physics_cols)

    # --- Analysis 4: Categorical cardinality ---
    log.info("Analyzing categorical cardinality ...")
    categorical_cols = ["pitch_type", "event_type", "hit_trajectory"]
    categorical_cols = [c for c in categorical_cols if c in pitches.columns]
    cat_cardinality = analyze_categorical(pitches, categorical_cols)

    # --- Analysis 5: Temporal patterns ---
    log.info("Analyzing temporal patterns ...")
    temporal_patterns = analyze_temporal_patterns(pitches)

    # --- Analysis 6: Seasonal drift ---
    log.info("Analyzing seasonal drift ...")
    seasonal_drift = analyze_seasonal_drift(pitches, physics_cols)

    # --- Analysis 7: Correlation structure ---
    log.info("Analyzing correlation structure ...")
    correlations = analyze_correlations(pitches, physics_cols)

    # --- Analysis 8: Timestamp quality ---
    log.info("Analyzing timestamp quality ...")
    timestamp_quality = analyze_timestamp_quality(pitches)

    # --- Generate plots ---
    log.info("Generating plots ...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_missing_heatmap(missing_by_season, output_dir / "missing_heatmap.png")
    plot_physics_distributions(pitches, physics_cols, output_dir / "distributions")
    plot_seasonal_distributions(pitches, physics_cols, output_dir / "seasonal")

    # --- Save CSVs ---
    log.info("Saving summary statistics ...")
    summary_stats = {
        "game_stats": game_stats,
        "physics_stats": physics_stats,
        "categorical_cardinality": cat_cardinality,
        "temporal_patterns": temporal_patterns,
        "seasonal_drift": seasonal_drift,
        "correlations": correlations,
        "timestamp_quality": timestamp_quality,
    }

    for name, df in summary_stats.items():
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            csv_path = output_dir / f"{name}.csv"
            df.to_csv(csv_path, index=False)
            log.info("Saved %s.csv", name)

    # --- Build HTML report ---
    log.info("Building HTML report ...")
    html_path = build_raw_eda_html_report(output_dir, summary_stats, list(pitches.columns))

    elapsed = time.time() - t0
    log.info("Raw EDA complete in %.1fs — report: %s", elapsed, html_path)

    return {
        "index_html": str(html_path),
        "game_stats_csv": str(output_dir / "game_stats.csv"),
        "physics_stats_csv": str(output_dir / "physics_stats.csv"),
        "output_dir": str(output_dir),
        "elapsed_secs": round(elapsed, 1),
    }


def compute_missing_heatmap_data(pitches) -> dict:
    """Compute % missing values per column by season."""
    import pandas as pd

    seasons = sorted(pitches["season"].dropna().unique().astype(int))
    result = {}

    for col in pitches.columns:
        col_missing = {}
        for s in seasons:
            s_data = pitches[pitches["season"] == s]
            pct_missing = round(100.0 * s_data[col].isna().sum() / len(s_data), 1)
            col_missing[int(s)] = pct_missing
        result[col] = col_missing

    return result


def analyze_game_counts(pitches):
    """Row counts, games per season, anomalies."""
    import pandas as pd

    data = []
    seasons = sorted(pitches["season"].dropna().unique().astype(int))

    for s in seasons:
        s_pitches = pitches[pitches["season"] == s]
        n_pitches = len(s_pitches)
        n_games = s_pitches["game_pk"].nunique()
        n_per_game = n_pitches / max(n_games, 1)

        # Find anomalies (games with very few pitches)
        game_counts = s_pitches.groupby("game_pk").size()
        anomalies = (game_counts < 10).sum()

        data.append({
            "season": s,
            "total_pitches": n_pitches,
            "num_games": n_games,
            "pitches_per_game_mean": round(n_per_game, 1),
            "anomalous_games": anomalies,
        })

    return pd.DataFrame(data)


def compute_column_stats_batch(pitches, cols):
    """Basic stats for numeric columns."""
    import numpy as np
    import pandas as pd
    from scipy import stats as spstats

    rows = []
    for col in cols:
        if col not in pitches.columns:
            continue
        values = pitches[col].dropna().to_numpy(dtype="float64")
        values = values[np.isfinite(values)]

        if len(values) < 3:
            continue

        row = {
            "col_name": col,
            "count": len(values),
            "mean": round(float(np.mean(values)), 4),
            "std": round(float(np.std(values)), 4),
            "min": round(float(np.min(values)), 4),
            "p5": round(float(np.percentile(values, 5)), 4),
            "p50": round(float(np.percentile(values, 50)), 4),
            "p95": round(float(np.percentile(values, 95)), 4),
            "max": round(float(np.max(values)), 4),
            "skewness": round(float(spstats.skew(values)), 4),
            "kurtosis": round(float(spstats.kurtosis(values)), 4),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def analyze_categorical(pitches, cols):
    """Cardinality and frequency by season."""
    import pandas as pd

    data = []
    for col in cols:
        if col not in pitches.columns:
            continue

        unique_vals = pitches[col].nunique()
        top_5 = pitches[col].value_counts().head(5)

        data.append({
            "col_name": col,
            "num_unique": unique_vals,
            "top_value": top_5.index[0] if len(top_5) > 0 else None,
            "top_value_count": top_5.iloc[0] if len(top_5) > 0 else 0,
        })

    return pd.DataFrame(data)


def analyze_temporal_patterns(pitches):
    """Velocity by inning, count distributions, etc."""
    import numpy as np
    import pandas as pd

    data = []

    # Velocity by inning (fatigue validation)
    if "release_speed" in pitches.columns and "inning" in pitches.columns:
        for inning in sorted(pitches["inning"].dropna().unique()):
            inning_data = pitches[pitches["inning"] == inning]
            velocities = inning_data["release_speed"].dropna().to_numpy(dtype="float64")
            velocities = velocities[np.isfinite(velocities)]

            if len(velocities) > 0:
                data.append({
                    "pattern": f"velocity_inning_{int(inning)}",
                    "mean": round(float(np.mean(velocities)), 2),
                    "std": round(float(np.std(velocities)), 2),
                    "count": len(velocities),
                })

    # Count distributions
    if "balls" in pitches.columns and "strikes" in pitches.columns:
        for b in range(4):
            for s in range(3):
                count_subset = pitches[(pitches["balls"] == b) & (pitches["strikes"] == s)]
                if len(count_subset) > 0:
                    data.append({
                        "pattern": f"count_{b}_{s}",
                        "count": len(count_subset),
                        "freq_pct": round(100.0 * len(count_subset) / len(pitches), 2),
                    })

    return pd.DataFrame(data)


def analyze_seasonal_drift(pitches, physics_cols):
    """KS distance matrices between seasons."""
    import numpy as np
    import pandas as pd
    from scipy import stats as spstats

    seasons = sorted(pitches["season"].dropna().unique().astype(int))
    data = []

    for col in physics_cols:
        if col not in pitches.columns:
            continue
        if len(seasons) < 2:
            continue

        for s1_idx, s1 in enumerate(seasons):
            v1 = pitches[pitches["season"] == s1][col].dropna().to_numpy(dtype="float64")
            v1 = v1[np.isfinite(v1)]

            for s2 in seasons[s1_idx + 1:]:
                v2 = pitches[pitches["season"] == s2][col].dropna().to_numpy(dtype="float64")
                v2 = v2[np.isfinite(v2)]

                if len(v1) >= 2 and len(v2) >= 2:
                    ks_stat, ks_pval = spstats.ks_2samp(v1, v2)
                    data.append({
                        "col_name": col,
                        "season_1": s1,
                        "season_2": s2,
                        "ks_stat": round(float(ks_stat), 4),
                        "ks_pvalue": round(float(ks_pval), 4),
                    })

    return pd.DataFrame(data)


def analyze_correlations(pitches, physics_cols):
    """Spearman correlation among physics features."""
    import numpy as np
    import pandas as pd
    import warnings

    physics_cols = [c for c in physics_cols if c in pitches.columns]
    if len(physics_cols) < 2:
        return pd.DataFrame()

    subset = pitches[physics_cols].dropna()
    if len(subset) < 10:
        return pd.DataFrame()

    data = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, c1 in enumerate(physics_cols):
            for c2 in physics_cols[i + 1:]:
                corr = subset[[c1, c2]].corr(method="spearman").iloc[0, 1]
                data.append({
                    "var_1": c1,
                    "var_2": c2,
                    "spearman_rho": round(float(corr), 4),
                })

    return pd.DataFrame(data)


def analyze_timestamp_quality(pitches):
    """Timestamp validity, inter-pitch Δt distribution."""
    import numpy as np
    import pandas as pd

    data = []

    if "pitch_start_time" in pitches.columns:
        valid = pitches["pitch_start_time"].notna().sum()
        total = len(pitches)
        data.append({
            "metric": "pitch_start_time_valid_pct",
            "value": round(100.0 * valid / max(total, 1), 2),
        })

    return pd.DataFrame(data)


def plot_missing_heatmap(missing_data: dict, output_path: Path):
    """Heatmap of missing % by season."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    if not missing_data:
        return

    df = pd.DataFrame(missing_data).T
    df = df.fillna(0)

    fig, ax = plt.subplots(figsize=(14, max(8, len(df) * 0.35)))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="RdYlGn_r", vmin=0, vmax=100, ax=ax,
                cbar_kws={"label": "% Missing"})
    ax.set_title("Missing Values by Season (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Season")
    ax.set_ylabel("Column")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close("all")
    log.info("Missing heatmap saved: %s", output_path)


def plot_physics_distributions(pitches, cols: list, output_dir: Path):
    """Histograms + KDE for each physics column."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import warnings

    output_dir.mkdir(parents=True, exist_ok=True)

    for col in cols:
        if col not in pitches.columns:
            continue

        values = pitches[col].dropna().to_numpy(dtype="float64")
        values = values[np.isfinite(values)]

        if len(values) < 10:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        clipped = np.clip(values, np.percentile(values, 1), np.percentile(values, 99))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import seaborn as sns
                sns.histplot(clipped, bins=50, stat="density", alpha=0.5, ax=ax, color="steelblue")
                sns.kdeplot(clipped, ax=ax, color="navy", linewidth=2)
            except Exception:
                ax.hist(clipped, bins=50, density=True, alpha=0.5, color="steelblue")

        ax.set_title(f"{col} Distribution", fontsize=11, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")

        output_file = output_dir / f"{col}.png"
        fig.savefig(output_file, dpi=100, bbox_inches="tight")
        plt.close("all")

    log.info("Distribution plots saved: %s", output_dir)


def plot_seasonal_distributions(pitches, cols: list, output_dir: Path):
    """Violin plots by season for each physics column."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import warnings

    output_dir.mkdir(parents=True, exist_ok=True)

    for col in cols:
        if col not in pitches.columns:
            continue

        seasons = sorted(pitches["season"].dropna().unique().astype(int))
        if len(seasons) < 2:
            continue

        season_data = []
        for s in seasons:
            s_values = pitches[pitches["season"] == s][col].dropna().to_numpy(dtype="float64")
            s_values = s_values[np.isfinite(s_values)]
            if len(s_values) > 0:
                # Subsample if too large
                if len(s_values) > 5000:
                    s_values = np.random.choice(s_values, 5000, replace=False)
                for v in s_values:
                    season_data.append({"season": str(s), "value": v})

        if not season_data:
            continue

        df_long = pd.DataFrame(season_data)

        fig, ax = plt.subplots(figsize=(12, 6))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import seaborn as sns
                sns.violinplot(data=df_long, x="season", y="value", ax=ax, inner="box")
            except Exception:
                import seaborn as sns
                sns.boxplot(data=df_long, x="season", y="value", ax=ax)

        ax.set_title(f"{col} by Season", fontsize=11, fontweight="bold")
        ax.set_xlabel("Season")
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=45)

        output_file = output_dir / f"{col}_by_season.png"
        fig.savefig(output_file, dpi=100, bbox_inches="tight")
        plt.close("all")

    log.info("Seasonal plots saved: %s", output_dir)


def build_raw_eda_html_report(output_dir: Path, stats: dict, columns: list) -> Path:
    """Build HTML index for raw EDA."""
    import html as html_mod

    body = f"""
<h1>MLB Raw PITCHES EDA Report</h1>
<p>Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>

<h2>Summary</h2>
<p>Total columns in PITCHES table: {len(columns)}</p>

<h2>Missing Values Heatmap</h2>
<p>See <a href="missing_heatmap.png">missing_heatmap.png</a> for seasonal patterns in data availability.</p>

<h2>Analysis Outputs</h2>
<ul>
"""

    for name in stats:
        if isinstance(stats[name], dict) and len(stats[name]) > 0:
            csv_file = f"{name}.csv"
            body += f'  <li><a href="{csv_file}">{name}</a></li>\n'

    body += """
</ul>

<h2>Plots</h2>
<ul>
  <li><a href="distributions/">Physics distributions (histograms)</a></li>
  <li><a href="seasonal/">Seasonal distributions (violin plots)</a></li>
</ul>
"""

    html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Raw PITCHES EDA</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; }}
    h1 {{ border-bottom: 2px solid #333; }}
    h2 {{ border-bottom: 1px solid #999; margin-top: 30px; }}
    a {{ color: #0066cc; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
{body}
</body>
</html>"""

    html_path = output_dir / "index.html"
    html_path.write_text(html_content)
    return html_path
