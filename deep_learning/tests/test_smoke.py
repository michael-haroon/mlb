from __future__ import annotations

import pandas as pd
import torch

from mlb_dl.datasets import PregameSequenceDataset, SequenceSpec, Standardizer, infer_feature_columns
from mlb_dl.feature_store import build_game_meta_from_pitches, build_team_game_frame
from mlb_dl.models import PregameMultiTaskModel
from mlb_dl.targets import (
    build_game_targets,
    build_player_batting_targets,
    build_player_pitching_targets,
)


def test_smoke() -> None:
    linescore = []
    pitches = []
    batting = []
    pitching = []
    game_pk = 1000
    for game_idx in range(8):
        current_pk = game_pk + game_idx
        date = pd.Timestamp("2024-04-01") + pd.Timedelta(days=game_idx)
        home_runs = [game_idx % 3, 1, 0, 2, 0, 0, 1, 0, 0]
        away_runs = [0, 1, game_idx % 2, 0, 0, 1, 0, 0, 0]
        for inning, (home, away) in enumerate(zip(home_runs, away_runs), start=1):
            linescore.append(
                {
                    "game_pk": current_pk,
                    "season": 2024,
                    "inning": inning,
                    "home_runs": home,
                    "away_runs": away,
                }
            )
        pitches.append(
            {
                "game_pk": current_pk,
                "season": 2024,
                "game_date": str(date.date()),
                "game_datetime_utc": f"{date.date()}T23:00:00Z",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_team_name": "Home",
                "away_team_name": "Away",
                "home_team_abbr": "HOM",
                "away_team_abbr": "AWY",
                "venue_id": 99,
                "venue_name": "Park",
                "day_night": "night",
                "weather_temp": 70.0,
                "weather_condition": "Clear",
                "weather_wind": "5 mph",
                "game_type_code": "R",
                "double_header": "N",
                "game_number": 1,
            }
        )
        for side in ("home", "away"):
            batting.append(
                {
                    "game_pk": current_pk,
                    "season": 2024,
                    "player_id": 10 if side == "home" else 20,
                    "player_name": f"{side} batter",
                    "side": side,
                    "batting_order": 1,
                    "is_substitute": False,
                    "game_ab": 4,
                    "game_runs": 1,
                    "game_hits": 2,
                    "game_doubles": 1,
                    "game_triples": 0,
                    "game_hr": 0,
                    "game_rbi": 1,
                    "game_bb": 0,
                    "game_ibb": 0,
                    "game_so": 1,
                    "game_sb": 0,
                    "game_cs": 0,
                    "game_hbp": 0,
                    "game_sac": 0,
                    "game_sf": 0,
                    "game_gidp": 0,
                    "game_lob": 1,
                }
            )
            pitching.append(
                {
                    "game_pk": current_pk,
                    "season": 2024,
                    "player_id": 30 if side == "home" else 40,
                    "player_name": f"{side} pitcher",
                    "side": side,
                    "is_starter": True,
                    "game_innings_pitched": 5.0,
                    "game_hits": 4,
                    "game_runs": 2,
                    "game_earned_runs": 2,
                    "game_bb": 1,
                    "game_so": 6,
                    "game_hr": 1,
                    "game_hbp": 0,
                    "game_pitches_thrown": 84,
                    "game_strikes_thrown": 55,
                    "game_balls_thrown": 29,
                    "game_strikes_looking": 12,
                    "game_strikes_swinging": 8,
                }
            )

    linescore_df = pd.DataFrame(linescore)
    pitch_meta = build_game_meta_from_pitches(pd.DataFrame(pitches))
    batting_df = pd.DataFrame(batting)
    pitching_df = pd.DataFrame(pitching)

    game_targets = build_game_targets(linescore_df, pitch_meta)
    batting_targets = build_player_batting_targets(batting_df)
    pitching_targets = build_player_pitching_targets(pitching_df)
    team_games = build_team_game_frame(batting_df, pitching_df, pitch_meta)

    assert not game_targets.empty
    assert batting_targets["game_total_bases"].iloc[0] == 3
    assert pitching_targets["target_status"].eq("trainable").all()
    assert not team_games.empty

    feature_columns = infer_feature_columns(team_games)
    standardizer = Standardizer.fit(team_games, feature_columns)
    ds = PregameSequenceDataset(
        team_games,
        game_targets,
        standardizer,
        SequenceSpec(history_length=3, min_history=2),
    )
    assert len(ds) > 0

    sample = ds[0]
    batch = {
        key: value.unsqueeze(0) if isinstance(value, torch.Tensor) and value.ndim > 0 else value
        for key, value in sample.items()
        if key != "game_pk"
    }
    batch["targets"] = {
        key: value.unsqueeze(0) for key, value in sample["targets"].items()
    }
    batch["sample_weight"] = sample["sample_weight"].unsqueeze(0)

    model = PregameMultiTaskModel(feature_dim=len(feature_columns), hidden_dim=16, dropout=0.0)
    out = model(batch)
    assert set(out) == {
        "home_win_logit",
        "yrfi_logit",
        "total_runs_mu",
        "total_runs_sigma",
        "home_run_diff_mu",
        "home_run_diff_sigma",
    }
    print("smoke test passed")


if __name__ == "__main__":
    test_smoke()

