from __future__ import annotations

from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import re

from .data_sources import ParquetCatalog
from .targets import (
    build_game_targets,
    build_player_batting_targets,
    build_player_pitching_targets,
    market_specs,
)


# ---------------------------------------------------------------------------
# MLB regime-change dates used to build structural-break binary features.
# These mark rule changes that create measurable distributional shifts in the
# historical data.  Each flag is set to 1.0 for all games on/after the cutoff.
# ---------------------------------------------------------------------------
MLB_REGIME_CHANGES = {
    # 2020: Relief pitcher 3-batter minimum (ARP rule)
    # Effect: deeper relief appearances → lower BABIP variance per appearance
    "rule_3batter_minimum": "2020-01-01",
    # 2022: Universal Designated Hitter adopted by both leagues
    # Effect: pitchers no longer bat in NL → structural increase in run totals
    "rule_universal_dh": "2022-01-01",
    # 2023: Shift ban + pitch clock introduced
    # Effect: higher BABIP, faster pace, measurable change in game duration
    "rule_shift_ban_pitch_clock": "2023-01-01",
}


PITCH_META_COLUMNS = [
    "game_pk",
    "season",
    "game_date",
    "game_datetime_utc",
    "game_number",
    "game_type_code",
    "double_header",
    "tiebreaker",
    "series_description",
    "series_game_number",
    "games_in_series",
    "game_status_detail",
    "game_status_code",
    "venue_id",
    "venue_name",
    "venue_city",
    "venue_state",
    "venue_latitude",
    "venue_longitude",
    "venue_timezone",
    "venue_tz_offset",
    "venue_capacity",
    "venue_surface",
    "venue_roof_type",
    "home_team_id",
    "home_team_name",
    "home_team_abbr",
    "home_league_id",
    "home_division_id",
    "home_wins",
    "home_losses",
    "home_win_pct",
    "home_games_played",
    "away_team_id",
    "away_team_name",
    "away_team_abbr",
    "away_league_id",
    "away_division_id",
    "away_wins",
    "away_losses",
    "away_win_pct",
    "away_games_played",
    "start_time",
    "day_night",
    "weather_condition",
    "weather_temp",
    "weather_wind",
    "attendance",
    "game_duration_minutes",
    "umpire_hp",
    "umpire_1b",
    "umpire_2b",
    "umpire_3b",
    "probable_pitcher_home_id",
    "probable_pitcher_away_id",
    "review_home_challenges_used",
    "review_away_challenges_used",
    "flag_no_hitter",
    "flag_perfect_game",
    "flag_away_team_no_hitter",
    "flag_home_team_no_hitter",
]

BATTING_SUM_COLUMNS = [
    "game_ab",
    "game_runs",
    "game_hits",
    "game_doubles",
    "game_triples",
    "game_hr",
    "game_rbi",
    "game_bb",
    "game_ibb",
    "game_so",
    "game_sb",
    "game_cs",
    "game_hbp",
    "game_sac",
    "game_sf",
    "game_gidp",
    "game_lob",
]

PITCHING_SUM_COLUMNS = [
    "game_innings_pitched",
    "game_hits",
    "game_runs",
    "game_earned_runs",
    "game_bb",
    "game_so",
    "game_hr",
    "game_hbp",
    "game_pitches_thrown",
    "game_strikes_thrown",
    "game_balls_thrown",
    "game_strikes_looking",
    "game_strikes_swinging",
]

PITCH_SEQUENCE_NUMERIC_COLUMNS = [
    "inning",
    "is_top_inning",
    "pre_on_first_id",
    "pre_on_second_id",
    "pre_on_third_id",
    "post_on_first_id",
    "post_on_second_id",
    "post_on_third_id",
    "is_scoring_play",
    "rbi_count",
    "score_home",
    "score_away",
    "cum_balls",
    "cum_strikes",
    "cum_outs",
    "pitch_sequence_index",
    "is_pitch",
    "pitch_number",
    "pitch_count_balls",
    "pitch_count_strikes",
    "pitch_count_outs",
    "is_in_play",
    "is_strike",
    "is_ball",
    "release_speed",
    "end_speed",
    "strike_zone_top",
    "strike_zone_bottom",
    "type_confidence",
    "plate_time",
    "extension",
    "coord_px",
    "coord_pz",
    "coord_x0",
    "coord_y0",
    "coord_z0",
    "coord_vx0",
    "coord_vy0",
    "coord_vz0",
    "coord_ax",
    "coord_ay",
    "coord_az",
    "pfx_x",
    "pfx_z",
    "break_angle",
    "break_length",
    "break_y",
    "spin_rate",
    "spin_direction",
    "zone_location",
    "hit_launch_speed",
    "hit_launch_angle",
    "hit_total_distance",
    "hit_coord_x",
    "hit_coord_y",
    "weather_temp",
    "venue_id",
]

PITCH_SEQUENCE_CATEGORICAL_COLUMNS = [
    "batter_id",
    "pitcher_id",
    "pitch_type",
    "pitch_call",
    "bat_side_code",
    "pitch_hand_code",
    "half_inning",
    "event_type",
    "at_bat_event",
    "hit_trajectory",
    "hit_hardness",
]


# ---------------------------------------------------------------------------
# Regime-change flag injection
# ---------------------------------------------------------------------------

def _add_regime_flags(df, date_col: str = "game_date"):
    """Add binary float32 regime-change columns to any frame that has a date."""
    import pandas as pd

    if date_col not in df.columns:
        for flag in MLB_REGIME_CHANGES:
            df[flag] = 0.0
        return df

    dates = pd.to_datetime(df[date_col], errors="coerce")
    for flag, cutoff_str in MLB_REGIME_CHANGES.items():
        cutoff = pd.Timestamp(cutoff_str)
        df[flag] = (dates >= cutoff).astype("float32")
    return df


# ---------------------------------------------------------------------------
# Public frame builders
# ---------------------------------------------------------------------------

def build_game_meta_from_pitches(pitches_df):
    import pandas as pd

    if pitches_df.empty:
        return pd.DataFrame(columns=PITCH_META_COLUMNS)

    cols = [col for col in PITCH_META_COLUMNS if col in pitches_df.columns]
    meta = pitches_df[cols].drop_duplicates("game_pk").copy()
    if "game_date" in meta.columns:
        meta["game_date"] = pd.to_datetime(meta["game_date"], errors="coerce")
    meta = _add_regime_flags(meta, date_col="game_date")
    return meta


def build_team_game_frame(
    boxscore_batting_df,
    boxscore_pitching_df,
    game_meta_df,
    pitches_df=None,
    runners_df=None,
    hits_df=None,
):
    """Aggregate one completed game into one history row per team."""
    import numpy as np
    import pandas as pd

    bat = _aggregate_side_table(boxscore_batting_df, BATTING_SUM_COLUMNS, prefix="bat")
    pit = _aggregate_side_table(boxscore_pitching_df, PITCHING_SUM_COLUMNS, prefix="pit")
    pitch_aggs = _build_pitch_side_aggs(pitches_df)
    runner_aggs = _build_runner_side_aggs(runners_df, pitches_df)
    hit_aggs = _build_hit_side_aggs(hits_df)

    frames = [frame for frame in [bat, pit, pitch_aggs, runner_aggs, hit_aggs] if not frame.empty]
    if not frames:
        return pd.DataFrame()

    team_games = frames[0]
    for frame in frames[1:]:
        team_games = team_games.merge(frame, on=["game_pk", "season", "side"], how="outer")

    team_games = team_games.replace([np.inf, -np.inf], np.nan)

    if game_meta_df is not None and not game_meta_df.empty:
        meta_cols = [
            col
            for col in [
                "game_pk",
                "game_date",
                "home_team_id",
                "away_team_id",
                "venue_id",
                "day_night",
                "weather_temp",
                "probable_pitcher_home_id",
                "probable_pitcher_away_id",
                *list(MLB_REGIME_CHANGES.keys()),
            ]
            if col in game_meta_df.columns
        ]
        team_games = team_games.merge(
            game_meta_df[meta_cols].drop_duplicates("game_pk"),
            on="game_pk",
            how="left",
        )
        team_games["team_id"] = np.where(
            team_games["side"].eq("home"),
            team_games.get("home_team_id"),
            team_games.get("away_team_id"),
        )
        team_games["opponent_team_id"] = np.where(
            team_games["side"].eq("home"),
            team_games.get("away_team_id"),
            team_games.get("home_team_id"),
        )
        team_games["is_home"] = team_games["side"].eq("home").astype("float32")

    # Ensure regime flags are present even when game_meta_df had no date column.
    for flag in MLB_REGIME_CHANGES:
        if flag not in team_games.columns:
            team_games[flag] = 0.0

    if "game_date" in team_games.columns:
        team_games["game_date"] = pd.to_datetime(team_games["game_date"], errors="coerce")
        team_games = team_games.sort_values(["game_date", "game_pk", "side"]).reset_index(drop=True)

    return team_games


def build_pitch_sequence_frame(pitches_df, game_meta_df):
    import numpy as np
    import pandas as pd

    if pitches_df.empty:
        return pd.DataFrame()

    cols = list(dict.fromkeys(
        col
        for col in [
            "game_pk",
            "season",
            "play_index",
            "at_bat_index",
            "pitch_sequence_index",
            *PITCH_SEQUENCE_NUMERIC_COLUMNS,
            *PITCH_SEQUENCE_CATEGORICAL_COLUMNS,
        ]
        if col in pitches_df.columns
    ))
    seq = pitches_df[cols].copy()
    if "is_top_inning" in seq.columns:
        seq["batting_side"] = np.where(seq["is_top_inning"].astype(bool), "away", "home")
        seq["fielding_side"] = np.where(seq["is_top_inning"].astype(bool), "home", "away")
    else:
        seq["batting_side"] = "unknown"
        seq["fielding_side"] = "unknown"

    if {"score_home", "score_away", "batting_side"}.issubset(seq.columns):
        home_diff = pd.to_numeric(seq["score_home"], errors="coerce") - pd.to_numeric(
            seq["score_away"], errors="coerce"
        )
        seq["score_diff_batting"] = np.where(seq["batting_side"].eq("home"), home_diff, -home_diff)

    sort_cols = [col for col in ["game_pk", "play_index", "pitch_sequence_index"] if col in seq.columns]
    if sort_cols:
        seq = seq.sort_values(sort_cols).reset_index(drop=True)
    seq["sequence_index"] = seq.groupby("game_pk").cumcount()

    if game_meta_df is not None and not game_meta_df.empty:
        meta_cols = [
            col for col in ["game_pk", "game_date", "home_team_id", "away_team_id", *list(MLB_REGIME_CHANGES.keys())]
            if col in game_meta_df.columns and (col == "game_pk" or col not in seq.columns)
        ]
        seq = seq.merge(game_meta_df[meta_cols].drop_duplicates("game_pk"), on="game_pk", how="left")

    return seq.replace([np.inf, -np.inf], np.nan)


def build_runner_state_frame(runners_df, pitches_df, game_meta_df):
    import numpy as np
    import pandas as pd

    if runners_df.empty:
        return pd.DataFrame()

    out = runners_df.copy()
    if pitches_df is not None and not pitches_df.empty and "play_index" in pitches_df.columns:
        side_lookup = pitches_df[["game_pk", "play_index", "is_top_inning"]].drop_duplicates(
            ["game_pk", "play_index"]
        ).copy()
        side_lookup["side"] = np.where(side_lookup["is_top_inning"].astype(bool), "away", "home")
        out = out.merge(side_lookup[["game_pk", "play_index", "side"]], on=["game_pk", "play_index"], how="left")
    if game_meta_df is not None and not game_meta_df.empty:
        meta_cols = [col for col in ["game_pk", "game_date", "home_team_id", "away_team_id"]
                     if col in game_meta_df.columns and (col == "game_pk" or col not in out.columns)]
        out = out.merge(game_meta_df[meta_cols].drop_duplicates("game_pk"), on="game_pk", how="left")
    out["sequence_index"] = out.groupby("game_pk").cumcount()
    return out.replace([np.inf, -np.inf], np.nan)


def build_batted_ball_frame(hits_df, game_meta_df):
    import numpy as np
    import pandas as pd

    if hits_df.empty:
        return pd.DataFrame()

    out = hits_df.copy()
    if game_meta_df is not None and not game_meta_df.empty:
        meta_cols = [col for col in ["game_pk", "game_date", "home_team_id", "away_team_id", "venue_id"]
                     if col in game_meta_df.columns and (col == "game_pk" or col not in out.columns)]
        out = out.merge(game_meta_df[meta_cols].drop_duplicates("game_pk"), on="game_pk", how="left")
    out["sequence_index"] = out.groupby("game_pk").cumcount()
    return out.replace([np.inf, -np.inf], np.nan)


def build_player_bio_frame(players_df):
    import numpy as np
    import pandas as pd

    if players_df.empty:
        return pd.DataFrame()

    out = players_df.drop_duplicates("player_id", keep="last").copy()
    if "height" in out.columns:
        out["height_inches"] = out["height"].map(_height_to_inches)
    for col in ["birth_date", "mlb_debut_date"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce", format="mixed")
    return out.replace([np.inf, -np.inf], np.nan)


def build_player_history_frames(batting_df, pitching_df, game_meta_df, players_df):
    import numpy as np

    batting_targets = build_player_batting_targets(batting_df)
    pitching_targets = build_player_pitching_targets(pitching_df)
    bio = build_player_bio_frame(players_df)

    batting_history = _merge_player_history(batting_df, batting_targets, game_meta_df, bio)
    pitching_history = _merge_player_history(pitching_df, pitching_targets, game_meta_df, bio)
    return (
        batting_history.replace([np.inf, -np.inf], np.nan),
        pitching_history.replace([np.inf, -np.inf], np.nan),
        batting_targets,
        pitching_targets,
        bio,
    )


def build_live_snapshot_frame(live_state_dir: str | None):
    import pandas as pd

    if live_state_dir is None:
        return pd.DataFrame()

    root = Path(live_state_dir)
    if not root.exists():
        return pd.DataFrame()

    rows = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        linescore = payload.get("linescore") or {}
        rows.append(
            {
                "game_pk": payload.get("game_pk"),
                "season": payload.get("season"),
                "game_date": payload.get("game_date"),
                "abstract_state": payload.get("abstract_state"),
                "coded_state": payload.get("coded_state"),
                "detailed_state": payload.get("detailed_state"),
                "polled_at": payload.get("polled_at"),
                "poll_lag_ms": payload.get("poll_lag_ms"),
                "current_inning": linescore.get("current_inning"),
                "inning_half": linescore.get("inning_half"),
                "outs": linescore.get("outs"),
                "balls": linescore.get("balls"),
                "strikes": linescore.get("strikes"),
                "home_runs": linescore.get("home_runs"),
                "away_runs": linescore.get("away_runs"),
                "home_hits": linescore.get("home_hits"),
                "away_hits": linescore.get("away_hits"),
                "home_team_id": payload.get("home_team_id"),
                "away_team_id": payload.get("away_team_id"),
                "current_play_json": json.dumps(payload.get("current_play")),
                "weather_json": json.dumps(payload.get("weather")),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Feature store builder
# ---------------------------------------------------------------------------

def build_feature_store(
    source_uri: str,
    output_dir: str,
    seasons: list[int] | None = None,
    live_state_dir: str | None = "data/live_state",
) -> dict[str, str]:
    """Read curated Parquet files and write model-ready feature-store tables.

    Processes one season at a time (newest → oldest) and streams each aggregated
    chunk directly into PyArrow ParquetWriters, keeping RAM flat regardless of
    the number of seasons requested.

    Newest seasons are processed first so that modern Statcast-rich seasons set
    the output schema.  Older seasons with missing columns are backfilled with
    null arrays by _write_chunk before being appended.
    """
    import gc
    import logging
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    log = logging.getLogger("mlb_dl.feature_store")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    catalog = ParquetCatalog(source_uri)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if seasons is None:
        all_paths = catalog.resolve_table_paths("pitches", seasons=None)
        discovered: set[int] = set()
        for p in all_paths:
            m = re.search(r"season=(\d{4})", p)
            if m:
                discovered.add(int(m.group(1)))
        season_list: list[int] = sorted(discovered)
    else:
        season_list = sorted(seasons)

    if not season_list:
        raise RuntimeError("No seasons found — check source_uri and season range.")

    season_list_desc = list(reversed(season_list))
    log.info(f"Processing {len(season_list_desc)} seasons: {season_list_desc[0]} … {season_list_desc[-1]}")

    outputs = {
        "game_meta": str(output / "game_meta.parquet"),
        "game_targets": str(output / "game_targets.parquet"),
        "player_batting_targets": str(output / "player_batting_targets.parquet"),
        "player_pitching_targets": str(output / "player_pitching_targets.parquet"),
        "team_games": str(output / "team_games.parquet"),
        "player_batting_history": str(output / "player_batting_history.parquet"),
        "player_pitching_history": str(output / "player_pitching_history.parquet"),
        "player_bios": str(output / "player_bios.parquet"),
        "pitch_sequences": str(output / "pitch_sequences.parquet"),
        "runner_states": str(output / "runner_states.parquet"),
        "batted_balls": str(output / "batted_balls.parquet"),
        "live_snapshots": str(output / "live_snapshots.parquet"),
        "market_specs": str(output / "market_specs.json"),
        "manifest": str(output / "manifest.json"),
    }

    writers: dict[str, pq.ParquetWriter] = {}
    row_totals: dict[str, int] = {k: 0 for k in outputs if k not in ("market_specs", "manifest")}
    raw_table_summary: dict[str, int] = {}

    def _write_chunk(name: str, df: pd.DataFrame) -> None:
        """Append df to the named output Parquet file.

        use_dictionary=False prevents PyArrow's C++ dictionary builder from
        caching every unique string value for the lifetime of the writer,
        which accumulates unbounded heap across 10 seasons of pitch data.
        """
        if df.empty:
            return
        table = pa.Table.from_pandas(df, preserve_index=False)
        if name not in writers:
            writers[name] = pq.ParquetWriter(
                outputs[name],
                table.schema,
                use_dictionary=False,
            )
            writers[name].write_table(table)
        else:
            target = writers[name].schema
            current = {f.name: table.column(f.name) for f in table.schema}
            aligned_arrays = []
            for field in target:
                if field.name in current:
                    col = current[field.name]
                    aligned_arrays.append(
                        col.cast(field.type, safe=False) if col.type != field.type else col
                    )
                else:
                    aligned_arrays.append(pa.array([None] * len(table), type=field.type))
            writers[name].write_table(
                pa.table(
                    {field.name: arr for field, arr in zip(target, aligned_arrays)},
                    schema=target,
                )
            )
        row_totals[name] += len(df)

    log.info("Loading players table (all seasons)...")
    t0 = time.time()
    players_df = catalog.read_table("players", columns=None, seasons=None)
    log.info(f"  -> players: {len(players_df):,} rows ({time.time()-t0:.1f}s)")
    raw_table_summary["players"] = len(players_df)
    player_bios = build_player_bio_frame(players_df)
    del players_df
    gc.collect()

    season_tables = ["pitches", "linescore", "runners", "boxscore_batting", "boxscore_pitching", "hits"]

    t_total = time.time()
    for season in season_list_desc:
        log.info(f"--- Season {season} ---")
        t_season = time.time()

        raw: dict[str, pd.DataFrame] = {}

        def _load_season_table(tbl: str, _s: int = season) -> tuple[str, pd.DataFrame]:
            return tbl, catalog.read_table(tbl, columns=None, seasons=[_s])

        with ThreadPoolExecutor(max_workers=len(season_tables)) as pool:
            futures = {pool.submit(_load_season_table, tbl): tbl for tbl in season_tables}
            for fut in as_completed(futures):
                tbl = futures[fut]
                raw[tbl] = fut.result()[1]
                log.info(f"  loaded {tbl}: {len(raw[tbl]):,} rows")
        for tbl, df in raw.items():
            raw_table_summary[tbl] = raw_table_summary.get(tbl, 0) + len(df)

        game_meta = build_game_meta_from_pitches(raw["pitches"])
        log.info(f"  game_meta: {len(game_meta):,} games")
        _write_chunk("game_meta", game_meta)

        game_targets = build_game_targets(raw["linescore"], game_meta_df=game_meta)
        _write_chunk("game_targets", game_targets)
        del raw["linescore"], game_targets
        gc.collect()

        team_games = build_team_game_frame(
            raw["boxscore_batting"],
            raw["boxscore_pitching"],
            game_meta,
            pitches_df=raw["pitches"],
            runners_df=raw["runners"],
            hits_df=raw["hits"],
        )
        _write_chunk("team_games", team_games)
        del team_games
        gc.collect()

        bat_targets = build_player_batting_targets(raw["boxscore_batting"])
        pit_targets = build_player_pitching_targets(raw["boxscore_pitching"])
        bat_history = _merge_player_history(raw["boxscore_batting"], bat_targets, game_meta, player_bios)
        pit_history = _merge_player_history(raw["boxscore_pitching"], pit_targets, game_meta, player_bios)
        _write_chunk("player_batting_targets", bat_targets)
        _write_chunk("player_pitching_targets", pit_targets)
        _write_chunk("player_batting_history", bat_history)
        _write_chunk("player_pitching_history", pit_history)
        del raw["boxscore_batting"], raw["boxscore_pitching"]
        del bat_targets, pit_targets, bat_history, pit_history
        gc.collect()

        pitch_sequences = build_pitch_sequence_frame(raw["pitches"], game_meta)
        _write_chunk("pitch_sequences", pitch_sequences)
        del pitch_sequences
        gc.collect()

        runner_states = build_runner_state_frame(raw["runners"], raw["pitches"], game_meta)
        _write_chunk("runner_states", runner_states)
        del raw["runners"], runner_states
        gc.collect()

        batted_balls = build_batted_ball_frame(raw["hits"], game_meta)
        _write_chunk("batted_balls", batted_balls)
        del raw["pitches"], raw["hits"], batted_balls
        gc.collect()

        log.info(f"  season {season} done in {time.time()-t_season:.1f}s")

        # Force glibc to return freed pages to the OS.  The default
        # MALLOC_ARENA_MAX (8 arenas) and glibc's aggressive sbrk growth means
        # gc.collect() alone does not lower the process RSS after each season.
        # malloc_trim(0) is a no-op on macOS (different allocator) and is safe
        # to call unconditionally on Linux.
        if platform.system() == "Linux":
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
                log.info(f"  malloc_trim(0) called after season {season}")
            except Exception as exc:
                log.debug(f"  malloc_trim skipped: {exc}")

    _write_chunk("player_bios", player_bios)
    del player_bios
    gc.collect()

    log.info("Building live snapshot frame...")
    live_snapshots = build_live_snapshot_frame(live_state_dir)
    _write_chunk("live_snapshots", live_snapshots)

    for w in writers.values():
        w.close()

    for name, path in outputs.items():
        if name in ("market_specs", "manifest"):
            continue
        if not Path(path).exists():
            pd.DataFrame().to_parquet(path, index=False)

    Path(outputs["market_specs"]).write_text(json.dumps(market_specs(), indent=2))
    log.info(f"Feature store build complete in {time.time()-t_total:.1f}s")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_uri": source_uri,
        "seasons": season_list,
        "live_state_dir": live_state_dir,
        "raw_tables": raw_table_summary,
        "artifacts": {name: {"rows": row_totals[name]} for name in row_totals},
        "regime_changes": {k: str(v) for k, v in MLB_REGIME_CHANGES.items()},
        "leakage_policy": (
            "pregame samples only use rows with game_date < target game_date; "
            "live samples only slice pitch_sequences up to sequence_index."
        ),
    }
    Path(outputs["manifest"]).write_text(json.dumps(manifest, indent=2, default=str))
    return outputs


# ---------------------------------------------------------------------------
# Private aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_side_table(df, sum_columns: list[str], prefix: str):
    import pandas as pd

    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    for col in sum_columns:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    grouped = out.groupby(["game_pk", "season", "side"], sort=False)[sum_columns].sum()
    grouped = grouped.rename(columns={col: f"{prefix}_{col}" for col in sum_columns})
    grouped[f"{prefix}_players_used"] = out.groupby(
        ["game_pk", "season", "side"], sort=False
    )["player_id"].nunique()
    return grouped.reset_index()


def _build_pitch_side_aggs(pitches_df):
    import numpy as np
    import pandas as pd

    if pitches_df is None or pitches_df.empty:
        return pd.DataFrame()

    if "is_top_inning" not in pitches_df.columns:
        return pd.DataFrame()

    sum_cols = ["is_pitch", "is_strike", "is_ball", "is_in_play", "is_scoring_play"]
    mean_cols = [
        "release_speed", "end_speed", "spin_rate",
        "coord_px", "coord_pz", "pfx_x", "pfx_z",
        "hit_launch_speed", "hit_launch_angle", "hit_total_distance",
    ]
    agg_map = {}
    for col in sum_cols:
        if col in pitches_df.columns:
            agg_map[col] = "sum"
    for col in mean_cols:
        if col in pitches_df.columns:
            agg_map[col] = "mean"
    if not agg_map:
        return pd.DataFrame()

    keep = ["game_pk", "season", "is_top_inning", *agg_map.keys()]
    out = pitches_df[[c for c in keep if c in pitches_df.columns]].copy()
    out["side"] = np.where(out["is_top_inning"].astype(bool), "away", "home")
    for col in sum_cols:
        if col in out.columns:
            out[col] = out[col].astype("float32")
    for col in mean_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    grouped = out.groupby(["game_pk", "season", "side"], sort=False).agg(agg_map)
    grouped.columns = [f"pitch_{col}_{agg_map[col]}" for col in grouped.columns]
    return grouped.reset_index()


def _build_runner_side_aggs(runners_df, pitches_df):
    import numpy as np
    import pandas as pd

    if runners_df is None or runners_df.empty:
        return pd.DataFrame()

    sum_cols = ["is_out", "is_scoring_event", "rbi", "earned", "team_unearned"]
    agg_cols = {col: "sum" for col in sum_cols if col in runners_df.columns}
    if not agg_cols:
        return pd.DataFrame()

    keep = ["game_pk", "season", "play_index", *agg_cols.keys()]
    out = runners_df[[c for c in keep if c in runners_df.columns]].copy()

    if pitches_df is not None and not pitches_df.empty and "play_index" in pitches_df.columns:
        side_lookup = pitches_df[["game_pk", "play_index", "is_top_inning"]].drop_duplicates(
            ["game_pk", "play_index"]
        ).copy()
        side_lookup["side"] = np.where(side_lookup["is_top_inning"].astype(bool), "away", "home")
        if "side" in out.columns:
            out = out.drop(columns=["side"])
        out = out.merge(side_lookup[["game_pk", "play_index", "side"]], on=["game_pk", "play_index"], how="left")
    if "side" not in out.columns:
        return pd.DataFrame()

    for col in agg_cols:
        out[col] = out[col].astype("float32")

    grouped = out.groupby(["game_pk", "season", "side"], sort=False).agg(agg_cols)
    grouped.columns = [f"runner_{col}_{agg_cols[col]}" for col in grouped.columns]
    return grouped.reset_index()


def _build_hit_side_aggs(hits_df):
    import pandas as pd

    if hits_df is None or hits_df.empty or "side" not in hits_df.columns:
        return pd.DataFrame()
    out = hits_df.copy()
    for col in ["hit_x", "hit_y"]:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")
    grouped = out.groupby(["game_pk", "season", "side"], sort=False).agg(
        hit_spray_count=("hit_type", "count"),
        hit_x_mean=("hit_x", "mean"),
        hit_y_mean=("hit_y", "mean"),
    )
    return grouped.reset_index()


def _merge_player_history(raw_df, target_df, game_meta_df, bio_df):
    import numpy as np
    import pandas as pd

    if raw_df.empty:
        return pd.DataFrame()
    out = raw_df.copy()
    if not target_df.empty:
        # Always bring in these target-only columns even if they appear in raw_df;
        # drop them from out first to avoid _x/_y suffixes.
        force_from_target = {"target_status", "plate_appearances_est", "game_total_bases", "game_hits_runs_rbi"}
        out = out.drop(columns=[c for c in force_from_target if c in out.columns], errors="ignore")
        target_cols = [col for col in target_df.columns if col not in out.columns or col in force_from_target]
        out = out.merge(
            target_df[list(dict.fromkeys(["game_pk", "player_id", *target_cols]))].drop_duplicates(["game_pk", "player_id"]),
            on=["game_pk", "player_id"],
            how="left",
        )
    if game_meta_df is not None and not game_meta_df.empty:
        meta_cols = [
            col
            for col in [
                "game_pk", "game_date", "home_team_id", "away_team_id", "venue_id", "weather_temp",
                *list(MLB_REGIME_CHANGES.keys()),
            ]
            if col in game_meta_df.columns and (col == "game_pk" or col not in out.columns)
        ]
        out = out.merge(game_meta_df[meta_cols].drop_duplicates("game_pk"), on="game_pk", how="left")
        out["team_id"] = np.where(out["side"].eq("home"), out.get("home_team_id"), out.get("away_team_id"))
        out["opponent_team_id"] = np.where(out["side"].eq("home"), out.get("away_team_id"), out.get("home_team_id"))
        out["is_home"] = out["side"].eq("home").astype("float32")
    if bio_df is not None and not bio_df.empty:
        bio_cols = [
            col
            for col in [
                "player_id", "weight", "current_age", "strike_zone_top",
                "strike_zone_bottom", "height_inches", "bat_side", "pitch_hand", "position_code",
            ]
            if col in bio_df.columns
        ]
        out = out.merge(bio_df[bio_cols], on="player_id", how="left")
    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
        out = out.sort_values(["player_id", "game_date", "game_pk"]).reset_index(drop=True)
    return out.replace([np.inf, -np.inf], np.nan)


def _height_to_inches(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"unknown", "none", "nan"}:
        return None
    if "'" not in text:
        return None
    try:
        feet, inches = text.replace('"', "").split("'", 1)
        return float(feet) * 12.0 + float(inches.strip())
    except ValueError:
        return None


def _table_summary(tables: dict) -> dict:
    summary = {}
    for name, df in tables.items():
        summary[name] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
        }
    return summary
