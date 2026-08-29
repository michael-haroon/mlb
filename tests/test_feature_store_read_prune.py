"""`_load_feature_store` must prune pitch_sequences to the Statcast era AT READ TIME.

Why this is not merely an optimisation. The population filter lives in `_build_datasets`,
which runs *after* the whole feature store is in RAM. Measured on the g5.2xlarge
(30GB, no swap) on 2026-08-29:

    pitch_sequences, all 39.5M rows, downcast   11.45 GB
    peak during _build_datasets                 26.90 GB  -> 2GB available
    consequence                                 kernel could not fork sshd for 90+ min;
                                                the box became unobservable and neither
                                                SSH nor SSM could reach it.

Only 19.6% of games in the store are 2015+, so reading the full archive costs ~8.5GB to
build a frame that is then thrown away. Pruning on read is safe because it is a strict
SUPERSET of every split's lower bound: train is floored at `min_date` explicitly, and
val/test begin at `train_end`/`val_end`, both of which are > min_date. So no row that any
split would have used can be removed.

The game-type filter deliberately stays in `_build_datasets` only — duplicating it here
would create two places for the population policy to drift.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from deep_learning.mlb_dl import train_unified

MIN_DATE = pd.Timestamp("2015-01-01")


def _write_store(tmp_path, game_date_arrow_type):
    """Write a minimal feature store whose pitch game_date uses a given arrow type.

    The real store has shipped game_date as a string and as a timestamp at different
    times, and the read-time filter has to work for both or it will silently prune
    nothing (or everything).
    """
    dates = pd.to_datetime(
        ["1995-06-01", "2005-06-01", "2014-12-31", MIN_DATE, "2016-06-01", "2024-06-01"]
    )
    games = pd.DataFrame({
        "game_pk": np.arange(1, len(dates) + 1, dtype="int64"),
        "game_date": dates,
        "game_type_code": ["R"] * len(dates),
        "target_status": ["trainable"] * len(dates),
    })

    # 3 pitches per game, so pruning is observable in the row count.
    pitches = games.loc[games.index.repeat(3), ["game_pk", "game_date"]].reset_index(drop=True)
    pitches["play_index"] = np.tile(np.arange(3), len(dates)).astype("int64")
    pitches["pitch_sequence_index"] = pitches["play_index"]
    pitches["release_speed"] = 92.0

    tbl = pa.Table.from_pandas(pitches, preserve_index=False)
    field = tbl.schema.field("game_date")
    if game_date_arrow_type == "string":
        col = pa.array(pitches["game_date"].dt.strftime("%Y-%m-%d").to_list(), pa.string())
    elif game_date_arrow_type == "date32":
        col = pa.array(pitches["game_date"].dt.date.to_list(), pa.date32())
    else:
        col = tbl.column("game_date")
    tbl = tbl.set_column(tbl.schema.get_field_index("game_date"),
                         field.with_type(col.type), col)
    pq.write_table(tbl, tmp_path / "pitch_sequences.parquet")

    games.to_parquet(tmp_path / "game_targets.parquet", index=False)
    games.to_parquet(tmp_path / "game_meta.parquet", index=False)
    empty = pd.DataFrame({"game_pk": pd.Series(dtype="int64"),
                          "team_id": pd.Series(dtype="int64"),
                          "game_date": pd.Series(dtype="datetime64[ns]")})
    empty.to_parquet(tmp_path / "team_games.parquet", index=False)
    empty.to_parquet(tmp_path / "player_batting_history.parquet", index=False)
    return tmp_path


@pytest.mark.parametrize("arrow_type", ["timestamp", "string", "date32"])
def test_pitches_pruned_to_statcast_era_on_read(tmp_path, arrow_type):
    """9 pre-2015 pitch rows must never enter RAM; the 9 in-era rows must all survive."""
    store = _write_store(tmp_path, arrow_type)
    frames = train_unified._load_feature_store(str(store))
    pitches = frames["pitch_sequences"]

    dates = pd.to_datetime(pitches["game_date"].astype(str), errors="coerce")
    assert len(pitches) == 9, (
        f"expected 9 in-era pitch rows, got {len(pitches)}: the read-time prune did not "
        f"apply (18 = no pruning at all)"
    )
    assert dates.min() >= MIN_DATE
    # The floor is inclusive, so the game exactly on 2015-01-01 must be kept.
    assert (dates == MIN_DATE).sum() == 3


def test_prune_is_disableable(tmp_path):
    """Ablations need the full archive; min_date=None must restore it."""
    store = _write_store(tmp_path, "timestamp")
    frames = train_unified._load_feature_store(str(store), min_date=None)
    assert len(frames["pitch_sequences"]) == 18


def test_prune_does_not_touch_game_targets(tmp_path):
    """game_targets must stay whole: temporal_split_dates needs it to pick cut points, and
    `_build_datasets` owns the authoritative population filter."""
    store = _write_store(tmp_path, "timestamp")
    frames = train_unified._load_feature_store(str(store))
    assert len(frames["game_targets"]) == 6


def test_pruned_read_gives_the_same_splits_as_filtering_after_load(tmp_path):
    """The prune must be a no-op on RESULTS -- only on peak memory.

    This is the property that makes it safe: same splits, fewer bytes.
    """
    from deep_learning.mlb_dl.game_transformer_dataset import AblationConfig

    store = _write_store(tmp_path, "timestamp")
    train_end = pd.Timestamp("2020-01-01")
    val_end = pd.Timestamp("2023-01-01")

    captured = []

    class _Spy:
        def __init__(self, pitch_sequences=None, **kw):
            self.pitch_sequences = pitch_sequences
            captured.append(pitch_sequences)

        def __len__(self):
            return 0

    orig = train_unified.GameTransformerDataset
    train_unified.GameTransformerDataset = _Spy
    try:
        pruned = train_unified._load_feature_store(str(store))
        train_unified._build_datasets(pruned, AblationConfig(), train_end, val_end)
        pruned_pks = [sorted(set(c["game_pk"])) for c in captured]

        captured.clear()
        full = train_unified._load_feature_store(str(store), min_date=None)
        train_unified._build_datasets(full, AblationConfig(), train_end, val_end)
        full_pks = [sorted(set(c["game_pk"])) for c in captured]
    finally:
        train_unified.GameTransformerDataset = orig

    assert pruned_pks == full_pks, (
        f"read-time prune changed the splits: pruned={pruned_pks} full={full_pks}"
    )
