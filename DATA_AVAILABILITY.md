# Data Availability Reference

Complete reference for when each MLB data source becomes available relative to game time.
Confirmed against GUMBO live API checks (2026-08-13/14/16/18) and Open-Meteo/ERA5 pipeline code.

## Game State Progression (GUMBO)

```
S (Scheduled) → P (Pre-Game, ~60-90 min) → PW (Warmup, ~30 min) → I (In Progress) → F (Final)
                                                                         ↕
                                                              II (Delayed: mid-game)
                                                              DR (Delayed: pre-game start)
```

### Delay States

| Code | detailedState | abstractGameState | Notes |
|------|---------------|-------------------|-------|
| `II` | Delayed: Inclement Weather | **Live** | Game started, then paused. Has full boxscore, allPlays up to delay point, batting orders, umpires. Same data availability as `I`. |
| `DR` | Delayed Start: Rain | **Preview** | Game not yet started. Has same data as `S`/`PW` — no boxscore, no allPlays. |

**Confirmed 2026-08-18:** CHW @ CHC (gamePk=824639), bottom 8th, 3-3, `statusCode=II`.
- `weather`: populated (`{'condition':'Partly Cloudy','temp':'77','wind':'9 mph, Out To CF'}`)
- `probablePitchers`: populated (Bryan Hudson / Kevin Gausman)
- `boxscore.pitchers`: populated (home 3, away 4)
- `battingOrder`: populated (9 each side)
- `allPlays`: 66 plays through bottom 8th
- `boxscore.officials`: all 4 umpires (HP: Manny Gonzalez, 1B: Tom Hanahan, 2B: Ron Kulpa, 3B: Scott Barry)
- `liveData.linescore`: inning=8, half=Bottom, outs=0, balls=0, strikes=0

**Key implication:** For the live repricing module, `II` games should be treated as `I` — all in-game data is present. Do NOT skip them as "not started."

---

## Availability Timeline

```
Scheduled ─────────────────────────────────────────────────── First Pitch
│                                                                    │
│  ALWAYS AVAILABLE (T0):                                            │
│  • Team rolling stats (all windows)                                │
│  • Rating systems (Elo, SRS, Pythag, Wolfe, Log5, consensus)       │
│  • Venue metadata (park_factor, is_dome, air_density_index)        │
│  • Schedule context (league, division, day/night)                   │
│  • Rule flags                                                       │
│  • Rest/density                                                     │
│  • Momentum/streaks                                                 │
│  • BSR offense/defense                                              │
│                                                                     │
│         12-36h before (T1 — PITCHER ANNOUNCED):                     │
│         • SP season ERA, WHIP, handedness                           │
│         • SP platoon splits (K-BB%, FIP by hand)                    │
│         • SP TTO decay, release consistency                         │
│         • Pitch-mix matchup score                                   │
│         • Team platoon wOBA vs SP hand                              │
│         • Weather forecast (Open-Meteo day-of):                     │
│           - temperature_f, wet_bulb_f ✓ reliable                   │
│           - air_density, air_density_ratio ✓ reliable              │
│           - surface_pressure ✓ excellent                           │
│           - humidity, vpd ~ moderate                                │
│           - wind_speed ~ moderate                                   │
│           - wind_toward_cf, wind_crossfield ~ marginal             │
│           - wind_gusts, precip ~ poor                              │
│                                                                     │
│                   60-90 min before (T2 — PRE-GAME STATE):           │
│                   • Confirmed lineups (batting order)                │
│                   • Umpire HP/2B assignments                        │
│                   • GUMBO weather snapshot                          │
│                   • (Weather forecast now very accurate)             │
│                                                                     │
│                              ─── FIRST PITCH ───                    │
│                                                                     │
│                              T3 (IN-GAME):                          │
│                              • Pitch sequences                      │
│                              • Score, count, outs                    │
│                              • Baserunners                           │
│                              • Live reliever stats                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Sources

### 1. Team Historical Stats (Rolling, EWMA, Ratings, Momentum)

| Attribute | Value |
|-----------|-------|
| **Available** | Always — as soon as prior game reaches Final (~30 min post-game) |
| **Source** | `boxscore_batting`, `boxscore_pitching` from GUMBO, deposited to S3 by `live_daemon.py` |
| **Lag** | ~30 min after last out (F state triggers deposit) |
| **Refresh** | Per-game (one new row per team per game) |

**Features derived:**
- Rolling batting: `{side}_roll{5,10,20}_{avg,obp,slg,iso,hr_rate,k_rate,bb_rate,babip,ops}`
- Rolling pitching: `{side}_roll{5,10,20}_{era,whip,k9,bb9,hr9,fip}`
- EWMA (halflife=15): `{side}_ewma_{avg,obp,slg,ops,era,whip,k9,fip}`
- Unified (both sides): `{side}_all_roll{10,20}_{stat}`, `{side}_all_ewma_{stat}`
- Diffs/Sums: `diff_roll{w}_{stat}`, `sum_roll{w}_{stat}`, `diff_ewma_{stat}`, `sum_ewma_{stat}`
- Momentum: `{side}_roll{10,20}_winpct`, `{side}_win_streak`, `{side}_roll{5,10,20}_rd_{mean,std}`
- BaseRuns: `home_bsr_offense`, `home_bsr_defense`, etc.
- Rating systems: `home_elo`, `elo_diff`, `elo_prob`, `home_srs`, `srs_diff`, `home_pythag_{1st,2nd}`, `home_wolfe`, `wolfe_prob`, `log5_prob`, `consensus_home_win_prob`, `consensus_home_win_std`
- Rest/density: `{side}_days_rest`, `{side}_games_last_7d`, `is_doubleheader`

---

### 2. Probable Pitchers

| Attribute | Value |
|-----------|-------|
| **Available** | Typically 12-36h before game (evening before for next-day games) |
| **Source** | `gameData.probablePitchers` from GUMBO — exposed at S (Scheduled) state once announced |
| **Official posting** | MLB updates "Probable Pitchers" page ~5-7pm ET the day before |
| **Confirmed** | Morning of game day (after any overnight scratches) |
| **Can change** | Yes — scratches happen day-of (injury, illness, postponement) |
| **Caveat** | Not all teams announce at the same time; some wait until morning |

**GUMBO API behavior:** `probablePitchers` field is populated at Scheduled state as soon as MLB receives the announcement. It does NOT wait for Pre-Game state.

**Features derived (require knowing pitcher identity):**
- SP season: `sp_{side}_season_era`, `sp_{side}_season_whip`, `sp_{side}_is_lefty`
- SP diffs: `sp_era_diff`, `sp_era_sum`, `sp_whip_diff`, `sp_whip_sum`
- SP platoon splits: `{side}_sp_kpct_vs_{lhh,rhh}_roll{5,10}`, `{side}_sp_bbpct_vs_{lhh,rhh}_roll{5,10}`, `{side}_sp_kbb_diff_vs_{lhh,rhh}_roll{5,10}`, `{side}_sp_fip_vs_{lhh,rhh}_roll{5,10}`
- SP fatigue/consistency: `{side}_sp_tto_velo_decay_roll{5,10}`, `{side}_sp_tto_release_{x,z}_std_roll{5,10}`
- Matchup: `{side}_team_pitchmix_matchup_score_roll10`
- Team platoon wOBA: `{side}_team_woba_vs_{lhp,rhp}_roll{100,200}pa`

---

### 3. Lineups (Batting Order)

| Attribute | Value |
|-----------|-------|
| **Available** | ~60-90 min before first pitch (Pre-Game state) |
| **Source** | `battingOrder` from GUMBO live feed |
| **API behavior** | `[]` at Scheduled, full 9-man at Pre-Game (confirmed 2026-08-14) |
| **Caveat** | Late scratches can change lineup after initial posting |

**Features derived:** Currently NONE directly. The platoon wOBA features (`team_woba_vs_{lhp,rhp}`) use recent game batters as a proxy, not the actual posted lineup.

---

### 4. Umpire Assignments

| Attribute | Value |
|-----------|-------|
| **Available (GUMBO)** | ~60-90 min before first pitch (Pre-Game state) |
| **Source** | `boxscore.officials` from GUMBO live feed |
| **API behavior** | Not populated until Pre-Game (P) state (confirmed 2026-08-16) |
| **Predictable earlier?** | YES — crew assignments to series are public 2-3 days before. MLB uses 4-man crews that rotate HP→1B→2B→3B each game. If you track the rotation, HP umpire is deterministic once crew is known. |
| **Third-party sources** | UmpireScorecard.com, Close Call Sports publish crew assignments 2-3 days ahead |

**Confirmed 2026-08-16:**
| State | Umpires Available |
|-------|-------------------|
| S (Scheduled) | NO |
| P (Pre-Game) | YES |
| PW (Warmup) | YES |
| I (In Progress) | YES |

**Features derived:**
- HP umpire: `ump_hp_rpg_factor`, `ump_hp_bb_per_game`, `ump_hp_k_per_game`, `ump_hp_called_strike_pct`
- 2B umpire: `ump_2b_sb_per_game`, `ump_2b_cs_per_game`

**Note:** Currently dead code at inference — `synthetic.py` never fetches umpire data, and umpire features don't appear in importance reports.

---

### 5. Weather (GUMBO — game-time snapshot)

| Attribute | Value |
|-----------|-------|
| **Available** | ~60-90 min before first pitch (Pre-Game state) |
| **Source** | `weather` field in GUMBO game data |
| **API behavior** | `{}` at Scheduled, populated at Pre-Game (confirmed 2026-08-14) |
| **Content** | Single snapshot: `{'condition':'Partly Cloudy', 'temp':'80', 'wind':'5 mph, R To L'}` |
| **Limitation** | Only condition + temp + wind string. No humidity, pressure, dew point, precipitation. |

Not used directly as model features — the full weather feature set comes from ERA5/Open-Meteo.

---

### 6. Weather (Open-Meteo Forecast — inference source)

| Attribute | Value |
|-----------|-------|
| **Available** | 7 days ahead (deterministic) + 50-member ensemble spread |
| **Source** | `api.open-meteo.com/v1/forecast` (best_match model) |
| **Refresh** | Daily via `fetch_weather.py --mode daily` |
| **S3 path** | `data/weather/source=forecast/venue_id={id}/date={date}.parquet` |
| **Units** | mph (wind), Fahrenheit (temperature) — matches feature engineering |

#### Forecast Accuracy by Variable and Lead Time

| Variable | 24h | 48h | 72h | 7-day | Notes |
|----------|-----|-----|-----|-------|-------|
| **Temperature** | MAE ~1-2°F | MAE ~2-3°F | MAE ~3-4°F | MAE ~5-6°F | Very reliable 24h; gradual degradation |
| **Surface pressure** | MAE ~1 hPa | MAE ~1.5 hPa | MAE ~2 hPa | MAE ~2.5 hPa | Excellent at all lead times (dynamically driven) |
| **Dew point** | Tracks temp | ~2-3°F | ~3-5°F | ~5-8°F | Moisture harder than temperature |
| **Relative humidity** | ±5-8% RH | ±8-12% RH | ±10-15% RH | ±15-20% RH | Compounds temp + dew point errors |
| **VPD** | Good | Moderate | Moderate | Poor | Sensitive to both temp and humidity |
| **Wind speed** | MAE ~3-5 mph | MAE ~4-6 mph | MAE ~5-8 mph | MAE ~7-10 mph | Moderate; misses gusts |
| **Wind direction** | MAE ~25-35° | MAE ~40-55° | MAE ~50-70° | Useless | Bimodal errors; regime-dependent |
| **Wind gusts** | Very noisy | Poor | Poor | Useless | Convective gusts unpredictable |
| **Precipitation (binary)** | POD ~0.8 | POD ~0.7 | POD ~0.6 | POD ~0.4 | Better for rain/no-rain than amount |
| **Precipitation (amount)** | RMSE ~2mm | RMSE ~3mm | Unreliable | Unreliable | Convective precip dominates summer |
| **Cloud cover** | Moderate | Poor | Poor | Useless | Shallow convection poorly resolved |

#### Feature Reliability at Inference

| Feature | Reliable at 24h? | Reliable at 72h? | Notes |
|---------|-----------------|-----------------|-------|
| `air_density` | YES | Marginal | f(temp, dew_point, pressure) — pressure excellent, temp good |
| `air_density_ratio` | YES | Marginal | Same as above |
| `temperature_f` | YES | Marginal | Directly from forecast |
| `humidity` | Moderate | Poor | Compounds temp + moisture errors |
| `vpd` | Moderate | Poor | Sensitive to humidity |
| `wet_bulb_f` | YES | Marginal | Tracks temperature closely |
| `wind_toward_cf` | Marginal | NO | Requires accurate wind direction |
| `wind_crossfield` | Marginal | NO | Same |
| `wind_speed` | Moderate | Poor | Speed OK-ish but noisy |
| `wind_gusts` | Poor | NO | Convective gusts unpredictable |
| `precip_6h` | Moderate | Poor | Binary signal OK; amount unreliable |
| `precip_24h` | Moderate | Poor | Same |
| `surface_pressure` | YES | YES | Excellent even at 7 days |
| `*_anomaly` z-scores | Inherits parent | Inherits parent | Anomaly amplifies absolute error |
| `wind_toward_cf_open` | Marginal | NO | Wind direction unreliable |

#### Ensemble Spread (Uncertainty)

50-member ECMWF ensemble provides `{var}_ens_std` — quantifies forecast confidence.
Larger spread = less reliable forecast. Model can learn to down-weight anomaly when uncertainty is high.

Variables: temperature, precipitation, wind_speed, wind_gusts, cloud_cover, surface_pressure.

---

### 7. Weather

| Attribute | Value |
|-----------|-------|
| **DL tensor training source** | Open-Meteo Historical Forecast `models=ecmwf_ifs` (dims 0-10, 12-15), `gfs_hrrr` (dims 11, 20-21), CAMS (17-19), ERA5 (16 soil moisture only) — see `deep_learning/mlb_dl/feature_store.py` `build_multihour_weather_frame` |
| **DL inference source** | Parity-matched per dim via `deep_learning/mlb_dl/weather_context.py` `fetch_live_weather` (ECMWF live product, HRRR visibility/pressure overwrite, ERA5 soil persistence) |
| **ERA5 reanalysis** | `archive-api.open-meteo.com/v1/archive`, `ARCHIVE_LAG_DAYS = 7` in `fetch_weather.py` — feeds soil-moisture persistence and classical features; ~7 days lagged |
| **S3 path** | `data/weather/source={source}/venue_id={id}/year={year}.parquet` |

**Known residual gap (2026-08-29):** the Historical Forecast training source is a stitched 0-2h-lead composite, so training weather carries shorter leads than live inference sees. The as-of obs+forecast rebuild (ASOS observations + HRRR/GFS forecasts-as-issued) replaces this design; see `deep_learning/mlb_dl/weather_asof.py` once landed.

---

### 8. Venue Metadata

| Attribute | Value |
|-----------|-------|
| **Available** | Always (static per venue) |
| **Source** | Pitches parquet metadata |
| **Features** | `park_factor` (expanding mean), `air_density_index` (elevation via ISA), `is_dome` |

---

### 9. Schedule Context

| Attribute | Value |
|-----------|-------|
| **Available** | Weeks/months ahead (schedule published pre-season) |
| **Source** | GUMBO schedule API |
| **Features** | `is_same_league`, `is_same_division`, `is_night_game`, `is_doubleheader` |

---

### 10. Rule Change Flags

| Attribute | Value |
|-----------|-------|
| **Available** | Always (temporal — keyed to `game_date`) |
| **Features** | `rule_3batter_minimum` (2020+), `rule_deadened_ball` (2021), `rule_sticky_substance_ban` (2021-06+), `rule_universal_dh` (2022+), `rule_shift_ban_pitch_clock` (2023+) |

---

### 11. In-Game Data (Live Module Only)

| Attribute | Value |
|-----------|-------|
| **Available** | Only during In Progress state (after first pitch) |
| **Source** | `allPlays` from GUMBO live feed |
| **Content** | Pitch-by-pitch: velocity, movement, spin, location, result, count, baserunners, score |

---

### 12. Statcast Rolling Features (NEW — from pitch-level data)

Source: `pitches` parquet (Statcast) and `runners` parquet from prior COMPLETED games (Final state).
All use shift(1) — only prior-game data enters the rolling window.

| Feature Family | Tier | Available When | SP-Dependent? | Features |
|---|---|---|---|---|
| **Batted Ball Quality (team)** | T0 | Always (prior games are Final) | No | `{side}_roll{10,20}_barrel_rate`, `hard_hit_pct`, `avg_ev`, `sweet_spot_pct`, `gb_rate`, `fb_rate`, `ld_rate` |
| **Batted Ball Quality (SP)** | T1* | Pitcher confirmed | Yes | `{side}_sp_barrel_allowed_roll{5,10}`, `hard_hit_allowed`, `avg_ev_allowed`, `gb_rate`, `fb_rate` |
| **Spin & Movement (SP)** | T1* | Pitcher confirmed | Yes | `{side}_sp_spin_rate_roll{5,10}`, `spin_FF/SL/CH`, `pfx_x/z`, `extension`, `velo_retention`, `spin_trend` |
| **Command (SP)** | T1* | Pitcher confirmed | Yes | `{side}_sp_zone_pct_roll{5,10}`, `first_pitch_strike`, `chase_induced`, `whiff_rate`, `csw_pct` |
| **Command (team batting)** | T0 | Always | No | `{side}_roll{10,20}_team_chase_rate`, `contact_rate`, `whiff_rate` |
| **Spray Direction** | T0 | Always | No | `{side}_roll{10,20}_pull_pct`, `center_pct`, `oppo_pct` |
| **Platoon Composition** | T1 | Pitcher confirmed | Yes (needs SP hand) | `platoon_advantage_index` |
| **Baserunning** | T0 | Always | No | `{side}_roll{10,20}_sb_success_rate`, `extra_base_taken_rate`, `first_to_third_rate`, `score_from_second_rate` |
| **Defense & Stranding** | T0 | Always | No | `{side}_roll{10,20}_errors_per_game`, `lob_per_game`, `stranding_rate` |
| **Pennant Race** | T0 | Always (standings snapshots from prior day) | No | `{side}_div_games_back`, `wc_games_back`, `diff_div_games_back`, `in_contention`, `season_progress` |

**\*T1 caveat — current inference architecture:** `_lookup_game_row()` returns the prior matchup row. SP-specific features on that row reflect the PRIOR game's SP, not today's. So effectively T0 in terms of data availability (the row exists), but the SP features may be STALE (wrong pitcher). Once synthetic-row inference is built, these become true T1 features keyed to today's confirmed SP.

#### Availability verified empirically (2026-08-15/16/18):
- **Standings source**: GUMBO per-game feed always returns `"-"` for `divisionGamesBack` (useless). Actual source: dedicated standings API (`/api/v1/standings?date=YYYY-MM-DD`) stored as daily snapshots (`standings_YYYY-MM-DD.parquet`). Prior-day lookup (game_date - 1 day) ensures no leakage. Leaders have `games_back=0.0`.
- **GUMBO Scheduled state** confirmed: `gamesPlayed` populated (used for `season_progress`).
- **GUMBO Final state** confirmed: all Statcast fields (hit_launch_speed, spin_rate, pfx_x, etc.), runners table, linescore errors/LOB fully populated.
- **Inference lag**: Features are fresh as of the most recent `build_features_incremental()` run (~30 min after prior game Final). For next-day games, features are always current. For same-day doubleheaders, game 2 features include game 1 data only if rebuild runs between games.

#### Data availability constraints (pre-2017):
- `spin_rate`, `pfx_x`, `pfx_z`, `extension`: NULL before Statcast era (~2017). Features will be NaN — correct behavior. Models handle via tree splits or imputation.
- `hit_launch_speed`, `hit_launch_angle`: Available from 2015+ (TrackMan). Sparse before that.
- Runners table: Available from 2015+.
- `hit_coord_x`, `hit_coord_y`: Available from 2015+ but coordinate system may shift between seasons.

---

### 13. Feature-to-Trading-Time Matrix

This is the key reference: after importance identifies signal-carrying features, WHEN can we place trades?

| Trading Window | Description | Feature Tiers Available | Strategy |
|---|---|---|---|
| **T0: Night before** | After prior day's games reach Final + rebuild | T0 only | Bet early on rolling/EWMA/ratings signals. Best odds, thinnest markets. |
| **T1: Morning of** | After SP confirmed (~12-36h before) | T0 + T1 | Full SP-specific model. Most liquid window for pre-game bets. |
| **T2: 60-90 min before** | Pre-Game state (lineups, umpires, weather) | T0 + T1 + T2 | Incremental edge from umpire/lineup. Markets tightest. |
| **T3: Live** | After first pitch (live module) | All + in-game state | Separate model (out of scope for pregame). |

**Practical implication:** If importance reveals that T0 features dominate, we can trade the night before and capture better odds. If SP-specific features dominate, we wait for pitcher confirmation. If umpire features matter, we must wait for Pre-Game state (60-90 min). This determines the optimal trading time.

---

## Train vs Inference Sources

| Feature group | Training source | Inference source | Mismatch risk |
|---------------|----------------|-----------------|---------------|
| Rolling/EWMA/Ratings | Historical box scores | Same (from S3 parquet) | None |
| SP features | Post-game SP ID (fallback to `probable_pitcher_*_id`) | GUMBO `probablePitchers` at Scheduled state | Low |
| Umpire | Post-game `officials` from boxscore | GUMBO Pre-Game state | None |
| Weather | ERA5 reanalysis (ground truth, 7-day lag) | Open-Meteo forecast (predicted) | **HIGH for wind direction/gusts; LOW for temp/pressure** |
| Venue/schedule | Static metadata | Same | None |

---

## Tier Summary for Feature Gating

| Tier | Gate Condition | Feature Count | Key Groups |
|------|---------------|---------------|------------|
| T0 | Always | ~480 (+100 new) | Rolling, EWMA, ratings, venue, schedule, rules, rest, momentum, **batted ball (team), spray, baserunning, defense, pennant race** |
| T1 | Pitcher confirmed | ~185 (+75 new) | SP stats, SP platoon, SP fatigue, matchup, weather forecast, **SP batted ball, spin/movement, command, platoon composition** |
| T2 | Pre-Game state | ~6-10 | Umpire HP/2B, lineups (future), precise weather |
| T3 | In Progress | N/A | Pitch sequences, score, baserunners (deep_learning/ module only) |
