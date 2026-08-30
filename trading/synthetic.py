"""
pregame/trading/synthetic.py
-----------------------------
Construct synthetic feature rows for upcoming games.

The old approach (_lookup_game_row) found the last time the exact (home, away)
matchup occurred — meaning team-rolling features could be months stale.

This module builds a fresh feature vector by:
1. Finding each team's most recent game on the correct side
2. Extracting the starting pitcher's features from their last start
3. Filling game-context from today's GUMBO schedule data
4. Filling weather from Open-Meteo forecast at scheduled first-pitch hour
5. Recomputing all derived features (diffs, sums, probs)
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR

logger = logging.getLogger(__name__)

# Columns that are team-rolling features (exist per-side, carry forward).
# These are extracted from the team's most recent game on the target side.
_TEAM_FEATURE_PREFIXES = (
    "_all_ewma_", "_all_roll10_", "_all_roll20_",
    "_ewma_", "_roll5_", "_roll10_", "_roll20_",
    "_win_streak", "_days_rest", "_games_last_7d",
    "_elo", "_srs", "_wolfe", "_pythag_1st", "_pythag_2nd",
    "_bsr_offense", "_bsr_defense", "_bsr_game",
    "_team_woba_vs_", "_team_pitchmix_matchup_score_",
    "_sp_bbpct_", "_sp_fip_", "_sp_kbb_", "_sp_kpct_",
    "_sp_tto_release_", "_sp_tto_velo_",
)

# SP features keyed by pitcher identity (not team)
_SP_FEATURE_PREFIXES = ("sp_{side}_season_era", "sp_{side}_season_whip", "sp_{side}_is_lefty")

# Game-context columns filled from GUMBO or venue lookup
_CONTEXT_COLUMNS = (
    "venue_id", "park_factor", "air_density_index",
    "temp_f", "is_dome", "is_night_game", "is_doubleheader",
    "is_same_league", "is_same_division",
)

# Weather feature columns produced by engineer_weather_features
_WEATHER_FEATURE_COLUMNS = (
    "air_density", "air_density_ratio", "wind_toward_cf", "wind_crossfield",
    "wind_speed", "wind_gusts", "precip_6h", "precip_24h",
    "vpd", "humidity", "wet_bulb_f", "temperature_f",
    "air_density_anomaly", "temperature_f_anomaly", "humidity_anomaly",
    "wind_speed_anomaly", "surface_pressure_anomaly", "wind_toward_cf_open",
)

# Module-level cache for weather artifacts (loaded once per process)
_weather_cache: dict = {}

# Venue elevation table (from feature_engineering.py)
_VENUE_ELEVATIONS_FT: dict[int, int] = {
    19: 5280, 15: 1082, 5325: 551, 7: 750, 3312: 815,
    2889: 466, 2602: 480, 4705: 1050, 2700: 1555,
}


def build_synthetic_row(
    away_team: str,
    home_team: str,
    features: pd.DataFrame,
    game_info: Optional[dict] = None,
) -> Optional[pd.DataFrame]:
    """Construct a feature vector for an upcoming game.

    Parameters
    ----------
    away_team : str
        Away team abbreviation (e.g. "BOS").
    home_team : str
        Home team abbreviation (e.g. "NYY").
    features : pd.DataFrame
        Full game_features.parquet (sorted by game_date).
    game_info : dict, optional
        GUMBO schedule data for today's game. Keys:
        - venue_id: int
        - probable_pitcher_home_id: int/float
        - probable_pitcher_away_id: int/float
        - game_number: int (1 or 2 for doubleheader)
        - day_night: str ("day" or "night")
        - home_league_id: int
        - away_league_id: int
        - home_division_id: int
        - away_division_id: int

    Returns
    -------
    pd.DataFrame with one row, or None if insufficient data.
    """
    home_row = _find_latest_team_row(home_team, features, side="home")
    away_row = _find_latest_team_row(away_team, features, side="away")

    if home_row is None or away_row is None:
        missing = []
        if home_row is None:
            missing.append(f"{home_team} (home)")
        if away_row is None:
            missing.append(f"{away_team} (away)")
        logger.warning(f"Cannot build synthetic row — no history for: {', '.join(missing)}")
        return None

    row: dict = {}

    # 1. Extract team-rolling features from each team's last same-side game
    _extract_side_features(home_row, "home", row)
    _extract_side_features(away_row, "away", row)

    # 2. SP features from probable pitcher's last start
    if game_info:
        _fill_sp_features(
            game_info.get("probable_pitcher_home_id"),
            features, "home", row,
        )
        _fill_sp_features(
            game_info.get("probable_pitcher_away_id"),
            features, "away", row,
        )
    else:
        # Fall back to SP from the team's last game row
        for col in home_row.index:
            if col.startswith("sp_home"):
                row[col] = home_row[col]
        for col in away_row.index:
            if col.startswith("sp_away"):
                row[col] = away_row[col]

    # 3. Game-context features
    _fill_context(row, game_info, features)

    # 3b. Weather features from Open-Meteo forecast
    if game_info and game_info.get("venue_id") and game_info.get("game_datetime_utc"):
        _fill_weather_features(row, game_info)

    # 4. Identity columns
    row["home_team_abbr"] = home_team
    row["away_team_abbr"] = away_team
    row["home_team_id"] = home_row.get("home_team_id")
    row["away_team_id"] = away_row.get("away_team_id")
    row["home_league_id"] = home_row.get("home_league_id")
    row["away_league_id"] = away_row.get("away_league_id")
    row["home_division_id"] = home_row.get("home_division_id")
    row["away_division_id"] = away_row.get("away_division_id")

    # 5. Recompute all derived features
    _recompute_derived(row)

    return pd.DataFrame([row])


def _find_latest_team_row(
    team_abbr: str,
    features: pd.DataFrame,
    side: Optional[str] = None,
) -> Optional[pd.Series]:
    """Find a team's most recent game row.

    Parameters
    ----------
    side : "home", "away", or None (any side)
    """
    if side == "home":
        mask = features["home_team_abbr"] == team_abbr
    elif side == "away":
        mask = features["away_team_abbr"] == team_abbr
    else:
        mask = (
            (features["home_team_abbr"] == team_abbr)
            | (features["away_team_abbr"] == team_abbr)
        )

    rows = features[mask]
    if rows.empty:
        return None
    # Features are sorted by game_date in build_features_incremental
    return rows.iloc[-1]


def _extract_side_features(row: pd.Series, side: str, out: dict) -> None:
    """Copy team-rolling features from a row where team was on `side`."""
    prefix = f"{side}_"
    for col in row.index:
        if not col.startswith(prefix):
            continue
        suffix = col[len(prefix):]
        # Only copy rolling/derived team features, not raw game outcomes
        if _is_team_feature(suffix):
            out[col] = row[col]


def _is_team_feature(suffix: str) -> bool:
    """Return True if this suffix represents a team-rolling feature."""
    # Rolling stats
    if suffix.startswith(("roll5_", "roll10_", "roll20_")):
        return True
    # Unified rolling
    if suffix.startswith(("all_ewma_", "all_roll10_", "all_roll20_")):
        return True
    # EWMA
    if suffix.startswith("ewma_"):
        return True
    # Ratings and momentum
    if suffix in (
        "elo", "srs", "wolfe", "pythag_1st", "pythag_2nd",
        "bsr_offense", "bsr_defense", "bsr_game",
        "win_streak", "days_rest", "games_last_7d",
    ):
        return True
    # SP pitch-level rolling
    if suffix.startswith(("sp_bbpct_", "sp_fip_", "sp_kbb_", "sp_kpct_", "sp_tto_")):
        return True
    # Team platoon matchup
    if suffix.startswith(("team_woba_vs_", "team_pitchmix_")):
        return True
    return False


def _fill_sp_features(
    pitcher_id: Optional[float],
    features: pd.DataFrame,
    side: str,
    out: dict,
) -> None:
    """Find probable pitcher's last start and extract their features."""
    if pitcher_id is None or (isinstance(pitcher_id, float) and np.isnan(pitcher_id)):
        return

    # Search both sides for this pitcher's last start
    sp_col_home = "sp_home_id"
    sp_col_away = "sp_away_id"
    prob_col_home = "probable_pitcher_home_id"
    prob_col_away = "probable_pitcher_away_id"

    # Try probable_pitcher columns first (pre-game announcement), fall back to sp_id
    mask = pd.Series(False, index=features.index)
    for col in (prob_col_home, sp_col_home, prob_col_away, sp_col_away):
        if col in features.columns:
            mask = mask | (features[col] == pitcher_id)

    starts = features[mask]
    if starts.empty:
        logger.debug(f"No prior starts found for pitcher {pitcher_id}")
        return

    last_start = starts.iloc[-1]

    # Determine which side the pitcher was on in their last start
    was_home = False
    for col in (prob_col_home, sp_col_home):
        if col in last_start.index and last_start.get(col) == pitcher_id:
            was_home = True
            break

    source_side = "home" if was_home else "away"

    # Extract SP features and remap to target side
    out[f"sp_{side}_season_era"] = last_start.get(f"sp_{source_side}_season_era", np.nan)
    out[f"sp_{side}_season_whip"] = last_start.get(f"sp_{source_side}_season_whip", np.nan)

    # SP is_lefty
    sp_hand_col = f"sp_{source_side}_hand"
    if sp_hand_col in last_start.index:
        out[f"sp_{side}_is_lefty"] = float(last_start[sp_hand_col] == "L")
    else:
        is_lefty_col = f"sp_{source_side}_is_lefty"
        if is_lefty_col in last_start.index:
            out[f"sp_{side}_is_lefty"] = last_start[is_lefty_col]

    out[f"probable_pitcher_{side}_id"] = pitcher_id


def _fill_context(row: dict, game_info: Optional[dict], features: pd.DataFrame) -> None:
    """Fill game-context features from GUMBO schedule data."""
    if game_info is None:
        # Minimal defaults — model handles NaN for these
        row["park_factor"] = np.nan
        row["air_density_index"] = 1.0
        row["temp_f"] = np.nan
        row["is_dome"] = np.nan
        row["is_night_game"] = np.nan
        row["is_doubleheader"] = 0.0
        row["is_same_league"] = np.nan
        row["is_same_division"] = np.nan
        return

    venue_id = game_info.get("venue_id")
    row["venue_id"] = venue_id

    # Park factor: use historical expanding mean for this venue
    if venue_id is not None and "venue_id" in features.columns:
        venue_games = features[features["venue_id"] == venue_id]
        if not venue_games.empty and "park_factor" in venue_games.columns:
            valid_pf = venue_games["park_factor"].dropna()
            row["park_factor"] = valid_pf.iloc[-1] if not valid_pf.empty else np.nan
        else:
            row["park_factor"] = np.nan
    else:
        row["park_factor"] = np.nan

    # Air density from venue elevation
    if venue_id is not None:
        elev = _VENUE_ELEVATIONS_FT.get(int(venue_id), 0)
        row["air_density_index"] = float((1.0 - 6.8756e-6 * elev) ** 4.2558)
    else:
        row["air_density_index"] = 1.0

    # Weather/time
    row["temp_f"] = game_info.get("temp_f", np.nan)
    row["is_dome"] = game_info.get("is_dome", np.nan)
    row["is_night_game"] = float(game_info.get("day_night", "night") == "night")
    row["is_doubleheader"] = float(game_info.get("game_number", 1) > 1)

    # League/division match
    h_league = game_info.get("home_league_id") or row.get("home_league_id")
    a_league = game_info.get("away_league_id") or row.get("away_league_id")
    h_div = game_info.get("home_division_id") or row.get("home_division_id")
    a_div = game_info.get("away_division_id") or row.get("away_division_id")

    if h_league is not None and a_league is not None:
        row["is_same_league"] = float(h_league == a_league)
    else:
        row["is_same_league"] = np.nan

    if h_div is not None and a_div is not None and h_league == a_league:
        row["is_same_division"] = float(h_div == a_div)
    else:
        row["is_same_division"] = 0.0


def _recompute_derived(row: dict) -> None:
    """Recompute all cross-team derived features from assembled home/away values."""

    # --- Differentials and sums for all rolling features ---
    home_keys = [k for k in row if k.startswith("home_roll") or k.startswith("home_all_")]
    for h_key in home_keys:
        suffix = h_key[len("home_"):]
        a_key = f"away_{suffix}"
        if a_key in row:
            h_val = row[h_key]
            a_val = row[a_key]
            if h_val is not None and a_val is not None:
                try:
                    row[f"diff_{suffix}"] = float(h_val) - float(a_val)
                    row[f"sum_{suffix}"] = float(h_val) + float(a_val)
                except (TypeError, ValueError):
                    pass

    # --- EWMA differentials ---
    for stat in ("avg", "obp", "slg", "ops", "era", "whip", "k9", "fip"):
        h_key = f"home_ewma_{stat}"
        a_key = f"away_ewma_{stat}"
        if h_key in row and a_key in row:
            try:
                h, a = float(row[h_key]), float(row[a_key])
                row[f"diff_ewma_{stat}"] = h - a
                row[f"sum_ewma_{stat}"] = h + a
            except (TypeError, ValueError):
                pass

    # --- Rating-derived features ---
    h_elo = _safe_float(row.get("home_elo"))
    a_elo = _safe_float(row.get("away_elo"))
    if h_elo is not None and a_elo is not None:
        row["elo_diff"] = h_elo - a_elo
        row["elo_sum"] = h_elo + a_elo
        # Home advantage = 24 Elo points (from ratings.py DEFAULT_PARAMS)
        row["elo_prob"] = 1.0 / (1.0 + 10.0 ** ((a_elo - (h_elo + 24.0)) / 400.0))

    h_srs = _safe_float(row.get("home_srs"))
    a_srs = _safe_float(row.get("away_srs"))
    if h_srs is not None and a_srs is not None:
        row["srs_diff"] = h_srs - a_srs
        row["srs_sum"] = h_srs + a_srs

    h_wolfe = _safe_float(row.get("home_wolfe"))
    a_wolfe = _safe_float(row.get("away_wolfe"))
    if h_wolfe is not None and a_wolfe is not None:
        row["wolfe_diff"] = h_wolfe - a_wolfe
        row["wolfe_sum"] = h_wolfe + a_wolfe
        # Bradley-Terry probability
        denom = h_wolfe + a_wolfe
        row["wolfe_prob"] = h_wolfe / denom if denom > 0 else 0.5

    for tier in ("1st", "2nd"):
        h_p = _safe_float(row.get(f"home_pythag_{tier}"))
        a_p = _safe_float(row.get(f"away_pythag_{tier}"))
        if h_p is not None and a_p is not None:
            row[f"pythag_{tier}_diff"] = h_p - a_p
            row[f"pythag_{tier}_sum"] = h_p + a_p

    # BaseRuns
    for stat in ("offense", "defense"):
        h_b = _safe_float(row.get(f"home_bsr_{stat}"))
        a_b = _safe_float(row.get(f"away_bsr_{stat}"))
        if h_b is not None and a_b is not None:
            row[f"bsr_{stat}_diff"] = h_b - a_b
            row[f"bsr_{stat}_sum"] = h_b + a_b

    # --- SP differentials ---
    h_era = _safe_float(row.get("sp_home_season_era"))
    a_era = _safe_float(row.get("sp_away_season_era"))
    if h_era is not None and a_era is not None:
        row["sp_era_diff"] = a_era - h_era
        row["sp_era_sum"] = a_era + h_era

    h_whip = _safe_float(row.get("sp_home_season_whip"))
    a_whip = _safe_float(row.get("sp_away_season_whip"))
    if h_whip is not None and a_whip is not None:
        row["sp_whip_diff"] = a_whip - h_whip
        row["sp_whip_sum"] = a_whip + h_whip

    # --- Consensus probability ---
    prob_cols = [v for k, v in row.items() if k.endswith("_prob") and k != "consensus_home_win_prob"
                 and isinstance(v, (int, float)) and not np.isnan(v)]
    if prob_cols:
        row["consensus_home_win_prob"] = float(np.mean(prob_cols))
        row["consensus_home_win_std"] = float(np.std(prob_cols)) if len(prob_cols) > 1 else 0.0
        row["consensus_prob"] = row["consensus_home_win_prob"]

    # --- Interaction terms ---
    elo_prob = row.get("elo_prob")
    same_league = row.get("is_same_league")
    same_div = row.get("is_same_division")
    if elo_prob is not None and same_league is not None:
        try:
            row["elo_prob_x_same_league"] = float(elo_prob) * float(same_league)
        except (TypeError, ValueError):
            pass
    if elo_prob is not None and same_div is not None:
        try:
            row["elo_prob_x_same_division"] = float(elo_prob) * float(same_div)
        except (TypeError, ValueError):
            pass


def _safe_float(val) -> Optional[float]:
    """Convert to float, returning None for NaN/None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Weather features (inference-time forecast lookup)
# ══════════════════════════════════════════════════════════════════════════════

def _load_weather_artifacts() -> tuple[Optional[dict], Optional[pd.DataFrame]]:
    """Load cached park azimuths and climatology. Returns (azimuths, climatology)."""
    if "azimuths" in _weather_cache and "climatology" in _weather_cache:
        return _weather_cache["azimuths"], _weather_cache["climatology"]

    features_dir = ARTIFACTS_DIR / "features"
    azimuth_path = features_dir / "park_azimuths.json"
    climatology_path = features_dir / "weather_climatology.parquet"

    azimuths = None
    climatology = None

    if azimuth_path.exists():
        with open(azimuth_path) as f:
            azimuths = {int(k): v for k, v in json.load(f).items()}
        logger.debug(f"Loaded park azimuths for {len(azimuths)} venues")
    else:
        logger.warning(f"No park azimuths at {azimuth_path} — wind features unavailable")

    if climatology_path.exists():
        climatology = pd.read_parquet(climatology_path)
        logger.debug(f"Loaded climatology: {len(climatology)} venue-month entries")
    else:
        logger.warning(f"No climatology at {climatology_path} — anomaly features unavailable")

    _weather_cache["azimuths"] = azimuths
    _weather_cache["climatology"] = climatology
    return azimuths, climatology


def _load_forecast_from_s3(venue_id: int, date_str: str) -> Optional[pd.DataFrame]:
    """Load forecast parquet from S3 for a single venue/date."""
    import boto3

    s3_bucket = "mlb-265753586044-us-east-1-an"
    forecast_key = f"data/weather/source=forecast/venue_id={venue_id}/date={date_str}.parquet"

    try:
        s3 = boto3.client("s3", region_name="us-east-1")
        buf = io.BytesIO()
        s3.download_fileobj(s3_bucket, forecast_key, buf)
        buf.seek(0)
        df = pd.read_parquet(buf)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["venue_id"] = venue_id
        return df
    except Exception as e:
        logger.warning(f"Could not load forecast for venue {venue_id} date {date_str}: {e}")
        return None


def _load_ensemble_from_s3(venue_id: int, date_str: str) -> Optional[pd.DataFrame]:
    """Load ensemble parquet from S3 for uncertainty signal."""
    import boto3

    s3_bucket = "mlb-265753586044-us-east-1-an"
    ensemble_key = f"data/weather/source=ensemble/venue_id={venue_id}/date={date_str}.parquet"

    try:
        s3 = boto3.client("s3", region_name="us-east-1")
        buf = io.BytesIO()
        s3.download_fileobj(s3_bucket, ensemble_key, buf)
        buf.seek(0)
        df = pd.read_parquet(buf)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["venue_id"] = venue_id
        return df
    except Exception:
        return None


def _fill_weather_features(row: dict, game_info: dict) -> None:
    """Fill weather features from Open-Meteo forecast at scheduled first-pitch hour."""
    from classical_learning.engineering.weather import (
        join_forecast_to_game,
        engineer_weather_features,
    )

    venue_id = int(game_info["venue_id"])
    game_dt_utc = pd.Timestamp(game_info["game_datetime_utc"], tz="UTC")
    game_hour_utc = game_dt_utc.floor("h")
    date_str = game_dt_utc.strftime("%Y-%m-%d")

    # Load cached artifacts
    azimuths, climatology = _load_weather_artifacts()
    if azimuths is None and climatology is None:
        logger.info("Weather artifacts not available — skipping weather features")
        return

    # Load forecast for this venue/date
    forecast_df = _load_forecast_from_s3(venue_id, date_str)
    if forecast_df is None or forecast_df.empty:
        logger.info(f"No forecast data for venue {venue_id} on {date_str}")
        return

    ensemble_df = _load_ensemble_from_s3(venue_id, date_str)

    # Get weather values at game hour
    weather_vals = join_forecast_to_game(
        venue_id, game_hour_utc, forecast_df, ensemble_df
    )
    if not weather_vals:
        logger.info(f"No forecast match within ±1h of {game_hour_utc} for venue {venue_id}")
        return

    # Build single-row DataFrame for engineer_weather_features
    weather_vals["venue_id"] = venue_id
    weather_vals["game_datetime_utc"] = game_dt_utc.isoformat()
    game_row = pd.DataFrame([weather_vals])

    # Engineer all derived features (physics, anomalies, interactions)
    engineered = engineer_weather_features(
        game_row,
        climatology if climatology is not None else pd.DataFrame(),
        azimuths if azimuths is not None else {},
    )

    # Copy weather feature columns into the synthetic row
    for col in _WEATHER_FEATURE_COLUMNS:
        if col in engineered.columns:
            val = engineered[col].iloc[0]
            row[col] = val if pd.notna(val) else np.nan

    # Also update air_density_index from the physics-based value
    if "air_density_ratio" in row and pd.notna(row.get("air_density_ratio")):
        row["air_density_index"] = row["air_density_ratio"]

    # temperature_f from forecast overrides GUMBO temp_f
    if "temperature_f" in row and pd.notna(row.get("temperature_f")):
        row["temp_f"] = row["temperature_f"]

    logger.debug(f"Weather features filled for venue {venue_id}: "
                 f"air_density={row.get('air_density', 'N/A'):.4f}, "
                 f"wind_cf={row.get('wind_toward_cf', 'N/A')}")
