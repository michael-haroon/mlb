"""The Statcast-era floor must actually FILTER the train split, not merely pick boundaries.

Regression tests for a silent population bug found 2026-08-29. `_STATCAST_MIN_DATE` was
handed to `temporal_split_dates`, which applies `min_date` only to a *local* copy of the
dates in order to choose the 80th/90th-percentile cut points, and then returns
`(train_end, val_end)` -- two upper bounds and no floor. `_build_datasets` therefore did:

    train_mask = dates < train_end          # no lower bound

so the train split silently absorbed the entire pre-Statcast archive. Measured on the
shipped `prepared_tensors` before the fix:

    split    games     seasons              weather populated
    train  157,150     1950-2024 (75)             11.4%
    val      3,204     2024-2025                  91.1%
    test     3,198     2025-2026                  90.9%

i.e. the model was fit ~89% weather-blind on a 1950-median population and scored on a
2024+ population. Non-regular-season games (spring training, exhibition, WBC) were also
present in every split: 7.1% of train, 16.0% of val, 16.7% of test.

These tests exercise the real `_build_datasets` and spy on the dataset constructor, so they
assert the production split assignment rather than a reimplementation of it.
"""

import numpy as np
import pandas as pd
import pytest

from deep_learning.mlb_dl import train_unified
from deep_learning.mlb_dl.datasets import temporal_split_dates

MIN_DATE = pd.Timestamp("2015-01-01")


class _DatasetSpy:
    """Stands in for GameTransformerDataset, recording what each split was handed.

    Building a real GameTransformerDataset needs the full 39.5M-row feature store; the
    split assignment under test happens entirely inside `_build_datasets`, before the
    constructor is reached.
    """

    instances: list["_DatasetSpy"] = []

    def __init__(self, pitch_sequences=None, split_start=None, split_end=None, **kwargs):
        self.pitch_sequences = pitch_sequences
        self.split_start = split_start
        self.split_end = split_end
        self.game_targets = kwargs.get("game_targets")
        _DatasetSpy.instances.append(self)

    def __len__(self):
        return len(self.pitch_sequences) if self.pitch_sequences is not None else 0


def _make_frames(seasons=range(1990, 2027), dates_per_season=20, include_spring=True):
    """Minimal feature-store frames spanning pre- and post-Statcast eras.

    Regular-season dates sit in Apr-Sep; spring-training dates in March, so the game-type
    filter and the date floor are independently observable.
    """
    rows = []
    pk = 1
    for season in seasons:
        for i in range(dates_per_season):
            month = 4 + (i * 6) // dates_per_season
            day = 1 + (i % 28)
            rows.append({
                "game_pk": pk,
                "game_date": pd.Timestamp(f"{season}-{month:02d}-{day:02d}"),
                "game_type_code": "R",
            })
            pk += 1
        if include_spring:
            for i in range(3):
                rows.append({
                    "game_pk": pk,
                    "game_date": pd.Timestamp(f"{season}-03-{10 + i:02d}"),
                    "game_type_code": "S",
                })
                pk += 1

    games = pd.DataFrame(rows)

    # One pitch row per game is enough: the split is assigned by date, not by pitch count.
    pitches = games[["game_pk", "game_date"]].copy()
    rng = np.random.default_rng(0)
    pitches["release_speed"] = rng.normal(92.0, 3.0, len(pitches)).astype("float32")

    empty = pd.DataFrame({"game_pk": pd.Series(dtype="int64")})
    return {
        "pitch_sequences": pitches,
        "game_targets": games,
        "game_meta": games.copy(),
        "team_games": empty.copy(),
        "player_batting_history": empty.copy(),
    }


@pytest.fixture(autouse=True)
def _spy(monkeypatch):
    _DatasetSpy.instances = []
    monkeypatch.setattr(train_unified, "GameTransformerDataset", _DatasetSpy)
    yield
    _DatasetSpy.instances = []


def _build(frames=None, **kwargs):
    """Call _build_datasets exactly as production does: four positional args."""
    frames = frames if frames is not None else _make_frames()
    train_end, val_end = temporal_split_dates(frames["game_targets"], min_date=MIN_DATE)
    from deep_learning.mlb_dl.game_transformer_dataset import AblationConfig

    train_ds, val_ds, test_ds = train_unified._build_datasets(
        frames, AblationConfig(), train_end, val_end, **kwargs
    )
    return train_ds, val_ds, test_ds, train_end, val_end


class TestStatcastFloor:
    def test_train_split_excludes_pre_statcast_pitches(self):
        """The core defect: pre-2015 pitches must not reach the train dataset."""
        train_ds, _, _, _, _ = _build()
        dates = pd.to_datetime(train_ds.pitch_sequences["game_date"])
        assert not dates.empty, "train split is empty; fixture or split logic is wrong"
        assert dates.min() >= MIN_DATE, (
            f"train split reaches back to {dates.min().date()}, before the Statcast floor "
            f"{MIN_DATE.date()}: {(dates < MIN_DATE).sum()} of {len(dates)} pitch rows leak"
        )

    def test_train_dataset_receives_a_split_start_floor(self):
        """`split_start` is what filters game_targets (game_transformer_dataset.py:529).

        Filtering only pitch_sequences is not sufficient -- targets are filtered
        independently inside the dataset, so without split_start the train dataset still
        enumerates pre-2015 games.
        """
        train_ds, _, _, _, _ = _build()
        assert train_ds.split_start is not None, (
            "train dataset built with split_start=None, so game_targets is unbounded below"
        )
        assert pd.Timestamp(train_ds.split_start) == MIN_DATE

    def test_floor_does_not_perturb_val_or_test(self):
        """Adding the floor must change only the train split."""
        _, val_ds, test_ds, train_end, val_end = _build()
        assert pd.Timestamp(val_ds.split_start) == pd.Timestamp(train_end)
        assert pd.Timestamp(val_ds.split_end) == pd.Timestamp(val_end)
        assert pd.Timestamp(test_ds.split_start) == pd.Timestamp(val_end)
        assert test_ds.split_end is None

        val_dates = pd.to_datetime(val_ds.pitch_sequences["game_date"])
        assert val_dates.min() >= pd.Timestamp(train_end)
        assert val_dates.max() < pd.Timestamp(val_end)

    def test_splits_remain_disjoint_and_lose_no_in_era_game(self):
        """No double-counting, and nothing at or after the floor is silently dropped."""
        train_ds, val_ds, test_ds, _, _ = _build()
        pk_sets = [set(ds.pitch_sequences["game_pk"]) for ds in (train_ds, val_ds, test_ds)]
        assert not (pk_sets[0] & pk_sets[1])
        assert not (pk_sets[1] & pk_sets[2])
        assert not (pk_sets[0] & pk_sets[2])

        frames = _make_frames()
        in_era = frames["game_targets"]
        in_era = in_era[in_era["game_date"] >= MIN_DATE]
        expected = set(in_era.loc[in_era["game_type_code"] == "R", "game_pk"])
        assert expected <= set().union(*pk_sets), (
            "regular-season games at or after the floor went missing from every split"
        )


class TestGameTypeFilter:
    def test_non_regular_season_games_excluded_from_every_split(self):
        """Spring training / exhibition / WBC are a different sport and are not traded.

        They were 14.3% of val and 14.0% of test, so they were shaping model selection.
        """
        train_ds, val_ds, test_ds, _, _ = _build()
        spring = set(
            _make_frames()["game_targets"].pipe(
                lambda df: df.loc[df["game_type_code"] != "R", "game_pk"]
            )
        )
        for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
            leaked = spring & set(ds.pitch_sequences["game_pk"])
            assert not leaked, f"{name} contains {len(leaked)} non-regular-season games"

    def test_missing_game_type_column_is_tolerated(self):
        """An older feature store without game_type_code must not crash the build."""
        frames = _make_frames(include_spring=False)
        frames["game_targets"] = frames["game_targets"].drop(columns=["game_type_code"])
        frames["game_meta"] = frames["game_meta"].drop(columns=["game_type_code"])
        train_ds, _, _, _, _ = _build(frames=frames)
        assert len(train_ds.pitch_sequences) > 0


class TestAdversarialEdgeCases:
    """Boundary, dtype and degenerate-population cases."""

    def _frames_with(self, rows):
        """Feature store frames from explicit (pk, date, type) rows."""
        games = pd.DataFrame(rows)
        pitches = games[["game_pk", "game_date"]].copy()
        pitches["release_speed"] = 92.0
        empty = pd.DataFrame({"game_pk": pd.Series(dtype="int64")})
        return {
            "pitch_sequences": pitches,
            "game_targets": games,
            "game_meta": games.copy(),
            "team_games": empty.copy(),
            "player_batting_history": empty.copy(),
        }

    def test_game_exactly_on_the_floor_is_kept(self):
        """The floor is inclusive (`>= min_date`); a game on 2015-01-01 must survive."""
        frames = _make_frames()
        frames["game_targets"] = pd.concat([
            frames["game_targets"],
            pd.DataFrame([{"game_pk": 999999, "game_date": MIN_DATE, "game_type_code": "R"}]),
        ], ignore_index=True)
        frames["pitch_sequences"] = pd.concat([
            frames["pitch_sequences"],
            pd.DataFrame([{"game_pk": 999999, "game_date": MIN_DATE, "release_speed": 92.0}]),
        ], ignore_index=True)
        train_ds, _, _, _, _ = _build(frames=frames)
        assert 999999 in set(train_ds.pitch_sequences["game_pk"])

    def test_game_exactly_on_train_end_belongs_to_val(self):
        """`train_end` is exclusive for train and inclusive for val — no double-count."""
        train_ds, val_ds, _, train_end, _ = _build()
        tr = pd.to_datetime(train_ds.pitch_sequences["game_date"])
        va = pd.to_datetime(val_ds.pitch_sequences["game_date"])
        assert (tr < pd.Timestamp(train_end)).all()
        assert (va >= pd.Timestamp(train_end)).all()

    def test_postseason_is_retained(self):
        """Regression guard: an ("R",)-only list would silently drop 440 games."""
        rows = [{"game_pk": 1, "game_date": pd.Timestamp("2019-06-01"), "game_type_code": "R"}]
        for i, code in enumerate(("F", "D", "L", "W")):
            rows.append({"game_pk": 10 + i,
                         "game_date": pd.Timestamp(f"2019-10-{5 + i:02d}"),
                         "game_type_code": code})
        rows += [{"game_pk": 100 + i,
                  "game_date": pd.Timestamp(f"2019-07-{1 + i:02d}"),
                  "game_type_code": "R"} for i in range(20)]
        rows += [{"game_pk": 200 + i,
                  "game_date": pd.Timestamp(f"2023-07-{1 + i:02d}"),
                  "game_type_code": "R"} for i in range(20)]
        frames = self._frames_with(rows)
        train_ds, val_ds, test_ds, _, _ = _build(frames=frames)
        seen = set().union(*[set(d.pitch_sequences["game_pk"])
                             for d in (train_ds, val_ds, test_ds)])
        assert {10, 11, 12, 13} <= seen, "postseason games were dropped"

    def test_all_star_and_exhibition_are_dropped(self):
        rows = [{"game_pk": 500 + i,
                 "game_date": pd.Timestamp("2019-07-01") + pd.Timedelta(days=i),
                 "game_type_code": "R"} for i in range(30)]
        rows += [
            {"game_pk": 9001, "game_date": pd.Timestamp("2019-07-09"), "game_type_code": "A"},
            {"game_pk": 9002, "game_date": pd.Timestamp("2019-03-15"), "game_type_code": "E"},
        ]
        frames = self._frames_with(rows)
        dss = _build(frames=frames)[:3]
        seen = set().union(*[set(d.pitch_sequences["game_pk"]) for d in dss])
        assert 9001 not in seen and 9002 not in seen

    def test_null_game_type_is_dropped_not_crashed(self):
        """886 rows in the real store carry a null code; isin(NaN) is False, so they go."""
        rows = [{"game_pk": 600 + i,
                 "game_date": pd.Timestamp("2019-07-01") + pd.Timedelta(days=i),
                 "game_type_code": "R"} for i in range(30)]
        rows.append({"game_pk": 9100, "game_date": pd.Timestamp("2019-08-01"),
                     "game_type_code": None})
        frames = self._frames_with(rows)
        dss = _build(frames=frames)[:3]
        seen = set().union(*[set(d.pitch_sequences["game_pk"]) for d in dss])
        assert 9100 not in seen

    def test_nat_dates_land_in_no_split(self):
        """NaT compares False against every bound, so such rows must not silently enter train."""
        rows = [{"game_pk": 700 + i,
                 "game_date": pd.Timestamp("2019-07-01") + pd.Timedelta(days=i),
                 "game_type_code": "R"} for i in range(30)]
        frames = self._frames_with(rows)
        frames["pitch_sequences"] = pd.concat([
            frames["pitch_sequences"],
            pd.DataFrame([{"game_pk": 9200, "game_date": pd.NaT, "release_speed": 92.0}]),
        ], ignore_index=True)
        frames["game_targets"] = pd.concat([
            frames["game_targets"],
            pd.DataFrame([{"game_pk": 9200, "game_date": pd.NaT, "game_type_code": "R"}]),
        ], ignore_index=True)
        dss = _build(frames=frames)[:3]
        seen = set().union(*[set(d.pitch_sequences["game_pk"]) for d in dss])
        assert 9200 not in seen

    def test_categorical_game_date_is_handled(self):
        """pitch_sequences ships game_date as a category dtype to save memory."""
        frames = _make_frames()
        frames["pitch_sequences"]["game_date"] = (
            frames["pitch_sequences"]["game_date"].astype(str).astype("category")
        )
        train_ds, _, _, _, _ = _build(frames=frames)
        dates = pd.to_datetime(train_ds.pitch_sequences["game_date"])
        assert dates.min() >= MIN_DATE

    def test_doubleheader_keeps_both_games(self):
        """Two game_pks on one date must both survive; the split is by date, not by game."""
        rows = [{"game_pk": 800 + i,
                 "game_date": pd.Timestamp("2019-07-01") + pd.Timedelta(days=i),
                 "game_type_code": "R"} for i in range(30)]
        rows += [
            {"game_pk": 9301, "game_date": pd.Timestamp("2019-08-15"), "game_type_code": "R"},
            {"game_pk": 9302, "game_date": pd.Timestamp("2019-08-15"), "game_type_code": "R"},
        ]
        frames = self._frames_with(rows)
        dss = _build(frames=frames)[:3]
        seen = set().union(*[set(d.pitch_sequences["game_pk"]) for d in dss])
        assert {9301, 9302} <= seen

    def test_filters_can_be_disabled_explicitly(self):
        """Opting out must be possible for ablations, and must actually opt out."""
        frames = _make_frames()
        train_ds, _, _, _, _ = _build(frames=frames, min_date=None, game_types=None)
        dates = pd.to_datetime(train_ds.pitch_sequences["game_date"])
        assert dates.min() < MIN_DATE, "min_date=None should restore the unbounded behaviour"

    def test_population_entirely_below_the_floor_yields_empty_train(self):
        """Degenerate case must be empty and obvious, never silently backfilled."""
        rows = [{"game_pk": 900 + i,
                 "game_date": pd.Timestamp("2019-07-01") + pd.Timedelta(days=i),
                 "game_type_code": "R"} for i in range(30)]
        frames = self._frames_with(rows)
        # A floor above every game in the fixture.
        train_ds, _, _, _, _ = _build(frames=frames, min_date=pd.Timestamp("2030-01-01"))
        assert len(train_ds.pitch_sequences) == 0


class TestTemporalSplitDates:
    def test_boundaries_ignore_pre_min_date_games(self):
        """Guard on the half that already worked: cut points come only from in-era dates."""
        frames = _make_frames()
        train_end, val_end = temporal_split_dates(frames["game_targets"], min_date=MIN_DATE)
        assert pd.Timestamp(train_end) >= MIN_DATE
        assert pd.Timestamp(val_end) >= pd.Timestamp(train_end)

        # Adding more ancient history must not move the boundaries at all.
        more = _make_frames(seasons=range(1950, 2027))
        t2, v2 = temporal_split_dates(more["game_targets"], min_date=MIN_DATE)
        assert pd.Timestamp(t2) == pd.Timestamp(train_end)
        assert pd.Timestamp(v2) == pd.Timestamp(val_end)
