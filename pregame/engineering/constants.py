"""Shared constants for pregame feature engineering.

These are independent copies of MLB schema definitions, decoupled from
the live module to ensure pregame can evolve without cross-dependencies.
"""

# MLB regime-change dates used to build structural-break binary features.
# These mark rule changes that create measurable distributional shifts in the
# historical data. Each flag is set to 1.0 for all games on/after the cutoff.
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
