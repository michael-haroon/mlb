# MLB Raw Data Schema

This document describes the structure of all raw Parquet tables produced by `download_history.py` and stored in S3 at `s3://mlb-265753586044-us-east-1-an/data/`.

## Table of Contents

1. [Overview](#overview)
2. [PITCHES](#pitches) — Complete pitch-level and play-level data
3. [BOXSCORE_BATTING](#boxscore_batting) — Per-game batting statistics
4. [BOXSCORE_PITCHING](#boxscore_pitching) — Per-game pitching statistics
5. [RUNNERS](#runners) — Runner movement on each play
6. [LINESCORE](#linescore) — Inning-by-inning runs, hits, errors
7. [HITS](#hits) — Spray chart coordinates
8. [PLAYERS](#players) — Player biographical data
9. [Data Relationships](#data-relationships)
10. [Important Notes](#important-notes)

---

## Overview

| Table | Rows (2024) | Columns | Grain | Key |
|-------|------------|---------|-------|-----|
| PITCHES | 14,137 | 170 | Pitch/play event | `(game_pk, play_index, pitch_sequence_index)` |
| BOXSCORE_BATTING | 894 | 33 | Batter per game | `(game_pk, player_id, side)` |
| BOXSCORE_PITCHING | 438 | 28 | Pitcher per game | `(game_pk, player_id, side)` |
| RUNNERS | 4,156 | 20 | Runner per play | `(game_pk, play_index, runner_id)` |
| LINESCORE | 363 | 11 | Inning per game | `(game_pk, inning)` |
| HITS | 2,013 | 10 | Hit coordinates | `(game_pk, inning, batter_id, pitcher_id)` |
| PLAYERS | 160 | 25 | Player bio | `(player_id)` |

---

## PITCHES

**Shape:** 14,137 rows × 170 columns  
**Grain:** One row per pitch event or non-pitch play event  
**Primary Key:** `(game_pk, play_index, pitch_sequence_index)`  
**File Location:** `s3://mlb-265753586044-us-east-1-an/data/season={YYYY}/pitches_batch_*.parquet`

### Schema

#### Game & Season
| Column | Type | Description |
|--------|------|-------------|
| `game_pk` | int64 | MLB's unique game identifier |
| `season` | int64 | Season (e.g., 2024) |
| `game_date` | str | Game date in YYYY-MM-DD format |
| `game_datetime_utc` | str | Game start time in ISO 8601 UTC (e.g., 2024-10-05T22:38:00Z) |
| `game_number` | int64 | Game number (1 or 2 for double headers) |
| `game_type_code` | str | One-letter code: S (Spring), R (Regular), F (Postseason), D (Division Series), L (League Championship), W (World Series) |
| `double_header` | str | Y/N flag for double header |
| `tiebreaker` | str | Y/N flag for playoff tiebreaker game |

#### Series Info
| Column | Type | Description |
|--------|------|-------------|
| `series_description` | str | Human-readable series name (e.g., "World Series") |
| `series_game_number` | int64 | Which game in the series (1-7 for best-of-7) |
| `games_in_series` | int64 | Total games in series |

#### Game Status
| Column | Type | Description |
|--------|------|-------------|
| `game_status_detail` | str | Detailed status (e.g., "Final", "In Progress") |
| `game_status_code` | str | Single-letter code (F=Final, I=In Progress, etc.) |
| `start_time_tbd` | bool | True if start time was originally TBD |

#### Venue
| Column | Type | Description |
|--------|------|-------------|
| `venue_id` | int64 | MLB venue identifier |
| `venue_name` | str | Stadium name (e.g., "Yankee Stadium") |
| `venue_city` | str | City |
| `venue_state` | str | State abbreviation (or country for non-US) |
| `venue_latitude` | float64 | Latitude |
| `venue_longitude` | float64 | Longitude |
| `venue_timezone` | str | Timezone ID (often "Unknown" in data) |
| `venue_tz_offset` | float64 | UTC offset in hours |
| `venue_capacity` | int64 | Stadium capacity |
| `venue_surface` | str | Field surface (e.g., "Grass", "Artificial") |
| `venue_roof_type` | str | Roof type (e.g., "Open", "Retractable", "Dome") |

#### Weather
| Column | Type | Description |
|--------|------|-------------|
| `weather_condition` | str | Sky condition (e.g., "Clear", "Cloudy", "Rain") |
| `weather_temp` | float64 | Temperature in Fahrenheit |
| `weather_wind` | str | Wind direction and speed (e.g., "6 mph, In From LF") |

#### Home Team
| Column | Type | Description |
|--------|------|-------------|
| `home_team_id` | int64 | Team ID |
| `home_team_name` | str | Full team name |
| `home_team_abbr` | str | 3-letter abbreviation |
| `home_league_id` | int64 | League ID (103=AL, 104=NL) |
| `home_league_name` | str | "American League" or "National League" |
| `home_division_id` | int64 | Division ID |
| `home_division_name` | str | Division name (e.g., "AL East") |
| `home_wins` | int64 | Wins at time of game |
| `home_losses` | int64 | Losses at time of game |
| `home_win_pct` | float64 | Win percentage (0–1 scale) |
| `home_division_games_back` | float64 | Games behind in division (NaN if leading) |
| `home_wild_card_games_back` | float64 | Games behind wildcard spot (NaN if leading) |
| `home_games_played` | int64 | Games played (W + L) |

#### Away Team
Same structure as home team, prefixed with `away_`.

#### Game Officials & Meta
| Column | Type | Description |
|--------|------|-------------|
| `start_time` | str | Local start time (e.g., "6:38") |
| `day_night` | str | "day" or "night" |
| `attendance` | float64 | Official attendance |
| `game_duration_minutes` | float64 | Total game time in minutes |
| `umpire_hp` | str | Home plate umpire name |
| `umpire_1b` | str | First base umpire name |
| `umpire_2b` | str | Second base umpire name |
| `umpire_3b` | str | Third base umpire name |

#### Pitching Decisions
| Column | Type | Description |
|--------|------|-------------|
| `winner_pitcher_id` | int64 | Pitcher credited with win |
| `winner_pitcher_name` | str | Name of winning pitcher |
| `loser_pitcher_id` | int64 | Pitcher charged with loss |
| `loser_pitcher_name` | str | Name of losing pitcher |
| `save_pitcher_id` | int64 | Pitcher credited with save (-1 if none) |
| `save_pitcher_name` | str | Name of save pitcher or "None" |

#### Manager Challenges (Replay Review)
| Column | Type | Description |
|--------|------|-------------|
| `review_home_challenges_used` | int64 | Number of challenges home team used |
| `review_home_challenges_remaining` | int64 | Challenges remaining for home team |
| `review_away_challenges_used` | int64 | Number of challenges away team used |
| `review_away_challenges_remaining` | int64 | Challenges remaining for away team |

#### Game Flags
| Column | Type | Description |
|--------|------|-------------|
| `flag_no_hitter` | bool | True if game was a combined no-hitter |
| `flag_perfect_game` | bool | True if either team pitched a perfect game |
| `flag_away_team_no_hitter` | bool | True if away team no-hit home team |
| `flag_home_team_no_hitter` | bool | True if home team no-hit away team |

#### Probable Pitchers
| Column | Type | Description |
|--------|------|-------------|
| `probable_pitcher_home_id` | int64 | Pre-game probable starter (home) |
| `probable_pitcher_away_id` | int64 | Pre-game probable starter (away) |

#### Game Leaders (single-game records)
| Column | Type | Description |
|--------|------|-------------|
| `leader_hit_distance` | float64 | Longest hit distance (feet) |
| `leader_hit_distance_player_id` | int64 | Player ID of longest hit |
| `leader_hit_speed` | float64 | Hardest hit (mph) |
| `leader_hit_speed_player_id` | int64 | Player ID of hardest hit |
| `leader_pitch_speed` | float64 | Fastest pitch (mph) |
| `leader_pitch_speed_player_id` | int64 | Player ID of fastest pitch |

#### Alerts
| Column | Type | Description |
|--------|------|-------------|
| `game_alerts_json` | str | JSON array of game alerts (usually "[]") |

---

#### At-Bat Context (play level, repeated for each pitch in an at-bat)
| Column | Type | Description |
|--------|------|-------------|
| `play_index` | int64 | Sequential at-bat index in game (0-indexed) |
| `at_bat_index` | int64 | Alternative at-bat counter |
| `inning` | int64 | Inning number (1–9+) |
| `half_inning` | str | "top" (away batting) or "bottom" (home batting) |
| `is_top_inning` | bool | True if top of inning |
| `captivating_index` | int64 | Internal MLB tracking index |
| `at_bat_start_time` | str | ISO 8601 timestamp when at-bat began |
| `at_bat_end_time` | str | ISO 8601 timestamp when at-bat ended |
| `at_bat_has_review` | bool | True if play was reviewed |
| `at_bat_is_complete` | bool | True if at-bat is complete (always True for final games) |

#### Matchup (batter vs. pitcher)
| Column | Type | Description |
|--------|------|-------------|
| `batter_id` | int64 | Batter's player ID |
| `batter_name` | str | Batter's name |
| `bat_side_code` | str | "L" (left-handed) or "R" (right-handed) |
| `pitcher_id` | int64 | Pitcher's player ID |
| `pitcher_name` | str | Pitcher's name |
| `pitch_hand_code` | str | "L" (left) or "R" (right) |

#### Splits (context for streak analysis)
| Column | Type | Description |
|--------|------|-------------|
| `split_batter` | str | Batter context (e.g., "vs_RHP", "vs_LHP") |
| `split_pitcher` | str | Pitcher context (e.g., "vs_LHB", "vs_RHB") |
| `men_on_base` | str | Runner context (e.g., "Empty", "RISP") |

#### Base State (before at-bat)
| Column | Type | Description |
|--------|------|-------------|
| `pre_on_first_id` | int64 | Runner on 1B before play (-1 if empty) |
| `pre_on_second_id` | int64 | Runner on 2B before play |
| `pre_on_third_id` | int64 | Runner on 3B before play |

#### Base State (after at-bat, if play completes)
| Column | Type | Description |
|--------|------|-------------|
| `post_on_first_id` | int64 | Runner on 1B after play (-1 if empty) |
| `post_on_second_id` | int64 | Runner on 2B after play |
| `post_on_third_id` | int64 | Runner on 3B after play |

#### At-Bat Outcome
| Column | Type | Description |
|--------|------|-------------|
| `at_bat_event` | str | Event result (e.g., "Flyout", "Single", "Strikeout") |
| `event_type` | str | Event category (e.g., "field_out", "single", "strikeout") |
| `is_scoring_play` | bool | True if play resulted in run(s) scoring |
| `rbi_count` | int64 | Number of RBIs credited to batter |
| `score_home` | int64 | Home team score after play |
| `score_away` | int64 | Away team score after play |
| `play_description` | str | English description of play (e.g., "Michael Massey flies out to right fielder Juan Soto") |

#### Count (balls, strikes, outs)
| Column | Type | Description |
|--------|------|-------------|
| `cum_balls` | int64 | Cumulative balls in at-bat count before pitch |
| `cum_strikes` | int64 | Cumulative strikes in at-bat count before pitch |
| `cum_outs` | int64 | Outs in inning before pitch |

---

#### Pitch-Level Detail
| Column | Type | Description |
|--------|------|-------------|
| `pitch_sequence_index` | int64 | 0-indexed pitch number within at-bat |
| `play_id` | str | Unique pitch identifier or "None" |
| `pitch_event_type` | str | Event type (e.g., "pitch", "action", "status_change") |
| `is_pitch` | bool | **True only if this row is an actual pitch; False for non-pitch events** |
| `pitch_number` | int64 | Pitch number in game (cumulative) or -1 if not a pitch |
| `pitch_start_time` | str | ISO 8601 timestamp when pitch/event started |
| `pitch_end_time` | str | ISO 8601 timestamp when pitch/event ended |
| `pitch_count_balls` | int64 | Ball count after pitch |
| `pitch_count_strikes` | int64 | Strike count after pitch |
| `pitch_count_outs` | int64 | Out count after pitch |

#### Pitch Type & Call
| Column | Type | Description |
|--------|------|-------------|
| `pitch_type` | str | Pitch classification (e.g., "FF" fastball, "SL" slider, "CH" changeup, "None" for non-pitches) |
| `pitch_call` | str | Umpire/system call (e.g., "Ball", "Strike", "Called Strike", "In play - no out", "Status Change - Pre-Game") |
| `pitch_event_flags_json` | str | JSON array of flags (usually "[]") |

#### Pitch Classification
| Column | Type | Description |
|--------|------|-------------|
| `is_in_play` | bool | True if ball was put in play |
| `is_strike` | bool | True if strike was called/swung |
| `is_ball` | bool | True if ball was called |
| `has_review` | bool | True if pitch was reviewed |

#### Pitch Velocity & Movement
| Column | Type | Description |
|--------|------|-------------|
| `release_speed` | float64 | Pitch velocity at release (mph) |
| `end_speed` | float64 | Pitch velocity at plate (mph) |
| `strike_zone_top` | float64 | Top of batter's strike zone (feet) |
| `strike_zone_bottom` | float64 | Bottom of batter's strike zone (feet) |
| `type_confidence` | float64 | Confidence in pitch type classification (0–1) |
| `plate_time` | float64 | Time from release to plate (seconds) |
| `extension` | float64 | Release extension (feet) |

#### Pitch Location (Cartesian coordinates)
| Column | Type | Description |
|--------|------|-------------|
| `coord_px` | float64 | Horizontal position at plate (feet, pitcher's perspective: +x = right) |
| `coord_pz` | float64 | Vertical position at plate (feet) |
| `coord_x0` | float64 | Initial release position (horizontal, feet) |
| `coord_y0` | float64 | Initial release position (distance, feet) |
| `coord_z0` | float64 | Initial release position (vertical, feet) |
| `coord_vx0` | float64 | Initial velocity (horizontal, ft/s) |
| `coord_vy0` | float64 | Initial velocity (distance, ft/s) |
| `coord_vz0` | float64 | Initial velocity (vertical, ft/s) |
| `coord_ax` | float64 | Acceleration (horizontal, ft/s²) |
| `coord_ay` | float64 | Acceleration (distance, ft/s²) |
| `coord_az` | float64 | Acceleration (vertical, ft/s²) |

#### Pitch Break & Spin
| Column | Type | Description |
|--------|------|-------------|
| `pfx_x` | float64 | Horizontal movement (inches) |
| `pfx_z` | float64 | Vertical movement (inches) |
| `break_angle` | float64 | Direction of break (degrees) |
| `break_length` | float64 | Total break (inches) |
| `break_y` | float64 | Break distance point (feet from plate) |
| `spin_rate` | float64 | Spin rate (RPM) |
| `spin_direction` | float64 | Spin direction (degrees, 0–360) |
| `zone_location` | int64 | Strike zone region (1–9, or 0 for out-of-zone) |

#### Hit Data (populated only if `is_in_play` = True)
| Column | Type | Description |
|--------|------|-------------|
| `hit_launch_speed` | float64 | Exit velocity (mph) |
| `hit_launch_angle` | float64 | Launch angle (degrees) |
| `hit_total_distance` | float64 | Projected distance (feet) |
| `hit_trajectory` | str | Trajectory type (e.g., "fly_ball", "ground_ball", "line_drive") |
| `hit_hardness` | str | Hit hardness (e.g., "hard", "medium", "soft") |
| `hit_coord_x` | float64 | Hit location (x-coordinate on field) |
| `hit_coord_y` | float64 | Hit location (y-coordinate on field) |

---

## BOXSCORE_BATTING

**Shape:** 894 rows × 33 columns  
**Grain:** One row per batter appearance per game  
**Primary Key:** `(game_pk, player_id, side)`  
**File Location:** `s3://mlb-265753586044-us-east-1-an/data/season={YYYY}/boxscore_batting_batch_*.parquet`

### Schema

#### Identity
| Column | Type | Description |
|--------|------|-------------|
| `game_pk` | int64 | Game ID |
| `season` | int64 | Season |
| `player_id` | int64 | Player ID |
| `player_name` | str | Player name |
| `side` | str | "home" or "away" |
| `batting_order` | int64 | Position in batting order (1–9) |
| `all_positions_json` | str | JSON array of positions played (e.g., `["1B"]`, `["OF", "DH"]`) |
| `is_substitute` | bool | True if player entered game as substitute |

#### Game Batting Stats
| Column | Type | Description |
|--------|------|-------------|
| `game_ab` | int64 | At-bats |
| `game_runs` | int64 | Runs scored |
| `game_hits` | int64 | Hits |
| `game_doubles` | int64 | Doubles |
| `game_triples` | int64 | Triples |
| `game_hr` | int64 | Home runs |
| `game_rbi` | int64 | Runs batted in |
| `game_bb` | int64 | Walks (bases on balls) |
| `game_ibb` | int64 | Intentional walks |
| `game_so` | int64 | Strikeouts |
| `game_sb` | int64 | Stolen bases |
| `game_cs` | int64 | Caught stealing |
| `game_hbp` | int64 | Hit by pitch |
| `game_sac` | int64 | Sacrifice bunts |
| `game_sf` | int64 | Sacrifice flies |
| `game_gidp` | int64 | Grounded into double plays |
| `game_lob` | int64 | Left on base |

#### Season Batting Stats (at time of game)
| Column | Type | Description |
|--------|------|-------------|
| `season_avg` | float64 | Batting average |
| `season_obp` | float64 | On-base percentage |
| `season_slg` | float64 | Slugging percentage |
| `season_ops` | float64 | OPS (on-base plus slugging) |
| `season_hr` | int64 | Season home runs |
| `season_rbi` | int64 | Season RBIs |
| `season_sb` | int64 | Season stolen bases |
| `season_games_played` | int64 | Games played in season |

---

## BOXSCORE_PITCHING

**Shape:** 438 rows × 28 columns  
**Grain:** One row per pitcher appearance per game  
**Primary Key:** `(game_pk, player_id, side)`  
**File Location:** `s3://mlb-265753586044-us-east-1-an/data/season={YYYY}/boxscore_pitching_batch_*.parquet`

### Schema

#### Identity
| Column | Type | Description |
|--------|------|-------------|
| `game_pk` | int64 | Game ID |
| `season` | int64 | Season |
| `player_id` | int64 | Player ID |
| `player_name` | str | Player name |
| `side` | str | "home" or "away" |
| `is_starter` | bool | True if pitcher was game starter |

#### Game Pitching Stats
| Column | Type | Description |
|--------|------|-------------|
| `game_innings_pitched` | float64 | Innings pitched (e.g., 1.1 = 1⅓) |
| `game_hits` | int64 | Hits allowed |
| `game_runs` | int64 | Runs allowed |
| `game_earned_runs` | int64 | Earned runs allowed |
| `game_bb` | int64 | Walks allowed |
| `game_so` | int64 | Strikeouts |
| `game_hr` | int64 | Home runs allowed |
| `game_hbp` | int64 | Hit batsmen |
| `game_pitches_thrown` | int64 | Total pitches thrown |
| `game_strikes_thrown` | int64 | Strikes thrown |
| `game_balls_thrown` | int64 | Balls thrown |
| `game_strikes_looking` | int64 | Called strikes (looking) |
| `game_strikes_swinging` | int64 | Swinging strikes |

#### Season Pitching Stats (at time of game)
| Column | Type | Description |
|--------|------|-------------|
| `season_era` | float64 | Earned run average |
| `season_whip` | float64 | Walks + hits per inning pitched |
| `season_wins` | int64 | Season wins |
| `season_losses` | int64 | Season losses |
| `season_saves` | int64 | Season saves |
| `season_innings_pitched` | float64 | Season innings pitched |
| `season_so` | int64 | Season strikeouts |
| `season_bb` | int64 | Season walks allowed |
| `season_games_played` | int64 | Season games appeared in |

---

## RUNNERS

**Shape:** 4,156 rows × 20 columns  
**Grain:** One row per runner per play  
**Primary Key:** `(game_pk, play_index, runner_id)`  
**File Location:** `s3://mlb-265753586044-us-east-1-an/data/season={YYYY}/runners_batch_*.parquet`

### Schema

#### Identity & Location
| Column | Type | Description |
|--------|------|-------------|
| `game_pk` | int64 | Game ID |
| `season` | int64 | Season |
| `play_index` | int64 | At-bat index in game |
| `play_event_index` | int64 | Event index within play |
| `runner_id` | int64 | Runner's player ID |
| `runner_name` | str | Runner's name |
| `responsible_pitcher_id` | int64 | Pitcher responsible for runner (for scoring purposes) |

#### Base Movement
| Column | Type | Description |
|--------|------|-------------|
| `movement_start` | str | Starting base ("1B", "2B", "3B") or null if batter |
| `movement_end` | str | Ending base or outcome ("1B", "2B", "3B", "score", or null if still on base) |

#### Out Detail
| Column | Type | Description |
|--------|------|-------------|
| `is_out` | bool | True if runner was put out |
| `out_base` | str | Base where runner was put out (e.g., "2B", "1B") |
| `out_number` | int64 | Out number in inning (1, 2, or 3) |

#### Scoring
| Column | Type | Description |
|--------|------|-------------|
| `is_scoring_event` | bool | True if runner scored |
| `rbi` | bool | True if this runner caused an RBI |
| `earned` | bool | True if run was earned |
| `team_unearned` | bool | True if run was unearned (due to error) |

#### Play Context
| Column | Type | Description |
|--------|------|-------------|
| `event` | str | Event name (e.g., "Single", "Double Play", "Flyout") |
| `event_type` | str | Event category (e.g., "single", "double_play", "field_out") |
| `movement_reason` | str | Reason for movement or null |
| `credits_json` | str | JSON array of fielding credits (putout, assist, error) |

**Example credits_json:**
```json
[
  {"player_id": 665742, "credit": "f_putout", "position_code": "9"},
  {"player_id": 667517, "credit": "f_assist", "position_code": "6"}
]
```

---

## LINESCORE

**Shape:** 363 rows × 11 columns  
**Grain:** One row per inning per game  
**Primary Key:** `(game_pk, inning)`  
**File Location:** `s3://mlb-265753586044-us-east-1-an/data/season={YYYY}/linescore_batch_*.parquet`

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `game_pk` | int64 | Game ID |
| `season` | int64 | Season |
| `inning` | int64 | Inning number (1–9+) |
| `home_runs` | int64 | Runs scored by home team in this inning |
| `away_runs` | int64 | Runs scored by away team in this inning |
| `home_hits` | int64 | Hits by home team in this inning |
| `away_hits` | int64 | Hits by away team in this inning |
| `home_errors` | int64 | Errors by home team in this inning |
| `away_errors` | int64 | Errors by away team in this inning |
| `home_left_on_base` | int64 | Runners left on base by home team in this inning |
| `away_left_on_base` | int64 | Runners left on base by away team in this inning |

---

## HITS

**Shape:** 2,013 rows × 10 columns  
**Grain:** One row per batted ball (hit)  
**Primary Key:** `(game_pk, inning, batter_id, pitcher_id)` (not unique; multiple hits per inning possible)  
**File Location:** `s3://mlb-265753586044-us-east-1-an/data/season={YYYY}/hits_batch_*.parquet`

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `game_pk` | int64 | Game ID |
| `season` | int64 | Season |
| `inning` | int64 | Inning number |
| `side` | str | "home" or "away" (team batting) |
| `batter_id` | int64 | Batter's player ID |
| `pitcher_id` | int64 | Pitcher's player ID |
| `hit_x` | float64 | Hit location x-coordinate on field (Cartesian) |
| `hit_y` | float64 | Hit location y-coordinate on field |
| `hit_type` | str | Hit type (e.g., "H" for hit, "O" for out) |
| `team_id` | int64 | Batting team's ID |

**Note:** These coordinates represent positions on a standard baseball field plot, with the pitcher's mound at approximately (0, 0) and home plate at (0, ~45).

---

## PLAYERS

**Shape:** 160 rows × 25 columns (2024 sample; grows as new players appear)  
**Grain:** One row per unique player  
**Primary Key:** `player_id`  
**File Location:** `s3://mlb-265753586044-us-east-1-an/data/players/players_batch_*.parquet`

### Schema

#### Name Variants
| Column | Type | Description |
|--------|------|-------------|
| `player_id` | int64 | MLB unique player ID |
| `full_name` | str | Full legal name |
| `use_name` | str | Name commonly used (e.g., "Mike" instead of "Michael") |
| `boxscore_name` | str | Name printed in boxscores (e.g., "Valdez, F") |
| `first_name` | str | First name |
| `last_name` | str | Last name |

#### Physical & Biographical
| Column | Type | Description |
|--------|------|-------------|
| `primary_number` | str | Jersey number |
| `birth_date` | str | Birth date in YYYY-MM-DD format |
| `birth_city` | str | Birth city |
| `birth_state` | str | Birth state/province or "Unknown" |
| `birth_country` | str | Birth country |
| `height` | str | Height (e.g., "5' 11\"") |
| `weight` | float64 | Weight in pounds |
| `current_age` | int64 | Age at time of ingestion |

#### Baseball Position
| Column | Type | Description |
|--------|------|-------------|
| `position_code` | str | Single-character code (P=pitcher, C=catcher, 1=1B, etc.) |
| `position_name` | str | Full position name (e.g., "Pitcher", "First Base") |
| `position_type` | str | Position category (e.g., "Pitcher", "Infielder", "Outfielder") |
| `position_abbreviation` | str | Abbreviated position (e.g., "P", "1B", "OF") |

#### Handedness & Strike Zone
| Column | Type | Description |
|--------|------|-------------|
| `bat_side` | str | Batting side ("L"=left, "R"=right, "S"=switch) |
| `pitch_hand` | str | Pitching hand ("L"=left, "R"=right) |
| `strike_zone_top` | float64 | Top of strike zone (feet, from batter's perspective) |
| `strike_zone_bottom` | float64 | Bottom of strike zone (feet) |

#### Career
| Column | Type | Description |
|--------|------|-------------|
| `mlb_debut_date` | str | Date of MLB debut in YYYY-MM-DD format |
| `draft_year` | int64 | MLB draft year or -1 if not drafted/unknown |
| `is_active` | bool | True if player is on an MLB roster |

---

## Data Relationships

### Foreign Keys

```
BOXSCORE_BATTING.game_pk → PITCHES.game_pk
BOXSCORE_BATTING.player_id → PLAYERS.player_id
BOXSCORE_BATTING.game_pk + BOXSCORE_BATTING.side ← PITCHES.game_pk + side (home/away teams)

BOXSCORE_PITCHING.game_pk → PITCHES.game_pk
BOXSCORE_PITCHING.player_id → PLAYERS.player_id

RUNNERS.game_pk → PITCHES.game_pk
RUNNERS.runner_id → PLAYERS.player_id
RUNNERS.play_index ← PITCHES.play_index

LINESCORE.game_pk → PITCHES.game_pk
LINESCORE.inning ← PITCHES.inning

HITS.game_pk → PITCHES.game_pk
HITS.inning ← PITCHES.inning
HITS.batter_id → PLAYERS.player_id
HITS.pitcher_id → PLAYERS.player_id
```

### Join Pattern: Complete Game with Pitches

```python
import pandas as pd

# Get all pitches for a specific game with player names
game_pitches = pitches_df[pitches_df['game_pk'] == 775334]
game_pitches = game_pitches.merge(
    players_df[['player_id', 'full_name', 'position_name']],
    left_on='batter_id',
    right_on='player_id',
    how='left',
    suffixes=('', '_batter')
)
game_pitches = game_pitches.merge(
    players_df[['player_id', 'full_name', 'position_name']],
    left_on='pitcher_id',
    right_on='player_id',
    how='left',
    suffixes=('', '_pitcher')
)
```

---

## Important Notes

### 1. **Denormalization in PITCHES**
Game context (venue, teams, weather, standings) is repeated in **every row** of the PITCHES table. This is intentional:
- Enables efficient querying without joins
- Allows filtering/analysis by any game attribute
- Trade-off: larger file size (~14k rows × 170 columns vs. normalized schema)

### 2. **Timestamps**
- **UTC format**: `game_datetime_utc`, `pitch_start_time`, `at_bat_start_time` are in ISO 8601 UTC
- **Local time**: `start_time` is local time (text, e.g., "6:38")
- **Timezone info**: `venue_timezone` is often "Unknown"; infer from `venue_state` if needed

### 3. **NaN vs. "None"**
- Float columns use `NaN` (numpy) for missing values
- String columns use `"None"` (literal string) for missing values
- Integer columns use -1 or 0 as sentinel values for "missing"
- Always check with `pd.isna()` or `df['col'].isna()` for floats; `df['col'] == "None"` for strings

### 4. **Filtering for Actual Pitches**
The PITCHES table contains non-pitch events (status changes, replays, etc.). Filter:
```python
actual_pitches = pitches_df[pitches_df['is_pitch'] == True]
```

### 5. **Play Granularity**
- **PITCHES**: One row per pitch event; multiple rows per at-bat
- **RUNNERS**: One row per runner per play; captures all runner movements
- **BOXSCORE_***: Aggregates per player per game
- **LINESCORE**: Inning-by-inning aggregates

### 6. **Run Scoring Attribution**
- `RUNNERS.is_scoring_event` = True when a run scores
- `RUNNERS.earned` indicates if run counts as earned
- `RUNNERS.rbi` indicates if run is credited as RBI to batter
- `RUNNERS.responsible_pitcher_id` tracks pitcher credited/charged with run

### 7. **Files Organization**
```
s3://mlb-265753586044-us-east-1-an/data/
├── season=2024/
│   ├── pitches_batch_*.parquet          (~200 files)
│   ├── boxscore_batting_batch_*.parquet (~80 files)
│   ├── boxscore_pitching_batch_*.parquet (~50 files)
│   ├── runners_batch_*.parquet          (~80 files)
│   ├── linescore_batch_*.parquet        (~30 files)
│   └── hits_batch_*.parquet             (~60 files)
├── season=2023/
│   └── ...
└── players/
    └── players_batch_*.parquet          (~50 files total)
```

### 8. **Schema Enforcement**
All tables are written with explicit Parquet schema using `SCHEMA_TYPE_MAP` in `download_history.py`. This ensures:
- Consistent dtypes across all batches
- Early detection of API schema changes
- Type safety for downstream pipelines

### 9. **Checkpoint & Retry**
- `checkpoint.json`: Set of completed `game_pk`s to avoid re-ingestion
- `retry_queue.json`: Games that failed during ingestion, with error details
- Running ingestion is idempotent: already-completed games are skipped

