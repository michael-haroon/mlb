"""Shared constants for pregame feature engineering.

These are independent copies of MLB schema definitions, decoupled from
the live module to ensure pregame can evolve without cross-dependencies.
"""

# The 30 active MLB franchise IDs as assigned by statsapi.mlb.com.
# Used as a hard gate to exclude exhibition opponents (college teams,
# minor-league affiliates, foreign national teams, futures squads).
MLB_FRANCHISE_IDS = frozenset({
    108,  # Los Angeles Angels
    109,  # Arizona Diamondbacks
    110,  # Baltimore Orioles
    111,  # Boston Red Sox
    112,  # Chicago Cubs
    113,  # Cincinnati Reds
    114,  # Cleveland Guardians
    115,  # Colorado Rockies
    116,  # Detroit Tigers
    117,  # Houston Astros
    118,  # Kansas City Royals
    119,  # Los Angeles Dodgers
    120,  # Washington Nationals
    121,  # New York Mets
    133,  # Oakland Athletics
    134,  # Pittsburgh Pirates
    135,  # San Diego Padres
    136,  # Seattle Mariners
    137,  # San Francisco Giants
    138,  # St. Louis Cardinals
    139,  # Tampa Bay Rays
    140,  # Texas Rangers
    141,  # Toronto Blue Jays
    142,  # Minnesota Twins
    143,  # Philadelphia Phillies
    144,  # Atlanta Braves
    145,  # Chicago White Sox
    146,  # Miami Marlins
    147,  # New York Yankees
    158,  # Milwaukee Brewers
})

# game_type_code values that represent competitive MLB play.
# R=Regular Season, S=Spring Training, D=Division Series,
# L=League Championship, F=World Series, W=Wild Card.
# Excluded: E=Exhibition (vs. college/minor-league), A=All-Star (pseudo-teams).
VALID_GAME_TYPE_CODES = frozenset({"R", "S", "D", "L", "F", "W"})

# MLB regime-change dates used to build structural-break binary features.
# These mark rule changes that create measurable distributional shifts in the
# historical data. Each flag is set to 1.0 for all games on/after the cutoff.
MLB_REGIME_CHANGES = {
    # 2020: Relief pitcher 3-batter minimum (ARP rule)
    # Effect: deeper relief appearances → lower BABIP variance per appearance
    "rule_3batter_minimum": "2020-01-01",
    # 2021-01-01: Deadened ball specification (reduced COR, seam height change)
    # Effect: reduced HR/FB rate, lower barrel exit velocity, ~8% HR drop league-wide
    "rule_deadened_ball": "2021-01-01",
    # 2021-06-15: Sticky-substance crackdown enforcement began
    # Effect: league-wide spin rate drop ~100rpm FF, pitcher ERA inflated post-enforcement
    "rule_sticky_substance_ban": "2021-06-15",
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
    # Runners side-assignment join key
    "play_index",
    "half_inning",
    # Pennant race context (available at Scheduled state via GUMBO)
    "home_division_games_back",
    "home_wild_card_games_back",
    "away_division_games_back",
    "away_wild_card_games_back",
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

# Per-table column allowlists for S3 pushdown in load_all().
# Only columns actually consumed by the feature engineering pipeline are listed;
# everything else stays on disk.
LINESCORE_COLUMNS = [
    "game_pk", "season", "inning", "home_runs", "away_runs",
    "home_errors", "away_errors",
    "home_left_on_base", "away_left_on_base",
]

BOXSCORE_BATTING_COLUMNS = ["game_pk", "side"] + BATTING_SUM_COLUMNS

BOXSCORE_PITCHING_COLUMNS = (
    ["game_pk", "side", "is_starter", "player_id"]
    + PITCHING_SUM_COLUMNS
    + ["season_era", "season_whip"]
)

PLAYERS_COLUMNS = ["player_id", "pitch_hand_code"]

# Per-pitch columns needed by pitch_level_features.py.
# One row per pitch event (millions of rows). Only the columns actually consumed
# are listed here to avoid loading the full 170-column pitches table.
PITCH_LEVEL_COLUMNS = [
    "game_pk",
    "season",
    "game_date",
    "game_type_code",
    "home_team_id",
    "away_team_id",
    "pitcher_id",
    "batter_id",
    "is_pitch",
    "release_speed",
    "coord_x0",
    "coord_z0",
    "pitch_type",
    "bat_side_code",
    "pitch_hand_code",
    "event_type",
    "at_bat_index",
    "pitch_number",
    "inning",
    "half_inning",
    "cum_outs",
    "pre_on_first_id",
    "pre_on_second_id",
    "pre_on_third_id",
    # Batted ball quality (Statcast, populated post-game)
    "hit_launch_speed",
    "hit_launch_angle",
    "hit_total_distance",
    "hit_trajectory",
    "hit_hardness",
    "is_in_play",
    "hit_coord_x",
    "hit_coord_y",
    # Spin & movement (Statcast, reliable from ~2017+)
    "spin_rate",
    "pfx_x",
    "pfx_z",
    "end_speed",
    "extension",
    # Command & plate discipline
    "zone_location",
    "pitch_call",
    # Join key for runners side-assignment
    "play_index",
]

RUNNERS_COLUMNS = [
    "game_pk",
    "season",
    "play_index",
    "runner_id",
    "responsible_pitcher_id",
    "movement_start",
    "movement_end",
    "is_out",
    "is_scoring_event",
    "event",
    "event_type",
    "movement_reason",
]
