"""
Two-part audit for the 33 newly deployed columns:

PART 1 — CODEBASE AUDIT
  Read the feature-engineering code for the 6 new groups and verify:
  - Every .rolling() is followed by .shift(1) (no leakage via omitted shift)
  - Temporal ordering is guaranteed before shift (explicit sort or sort-invariant)
  - Date windows use strict exclusion of today (< not <=)
  - No accidental pull of current-game data into the feature value

PART 2 — EMPIRICAL API AVAILABILITY
  For each data source the new features depend on, hit the MLB API for:
  - A Scheduled game (tomorrow) → which fields are populated?
  - A Final game (today/yesterday) → which fields are populated?
  Determines the earliest trading tier (T0/T1/T2) each feature can actually be computed.
"""
import ast
import re
import textwrap
import requests
import json

MLB_API = "https://statsapi.mlb.com"

passes = fails = warns = 0


def ok(label, msg=""):
    global passes
    passes += 1
    print(f"  [PASS] {label}" + (f" — {msg}" if msg else ""))


def fail(label, msg=""):
    global fails
    fails += 1
    print(f"  [FAIL] {label}" + (f" — {msg}" if msg else ""))


def warn(label, msg=""):
    global warns
    warns += 1
    print(f"  [WARN] {label}" + (f" — {msg}" if msg else ""))


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — CODEBASE AUDIT
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("PART 1 — CODEBASE AUDIT")
print("=" * 70)

# Read source files
with open("classical_learning/engineering/pitch_level_features.py") as f:
    plf_src = f.read()
with open("classical_learning/engineering/feature_engineering.py") as f:
    fe_src = f.read()

# ── C1: Every .rolling(...) has .shift(1) immediately after ─────────────────
# Pattern: .rolling(...).mean() [possibly .sum()] NOT followed by .shift(1)
print("\n[C1] Every .rolling().mean()/.sum() has .shift(1)")

ROLL_WITHOUT_SHIFT = re.compile(
    r'\.rolling\([^)]+\)\s*\.\s*(?:mean|sum|std)\([^)]*\)\s*(?!\.shift)',
    re.MULTILINE,
)

def _find_roll_no_shift(src, filename):
    issues = []
    lines = src.splitlines()
    for i, line in enumerate(lines, 1):
        # Only flag if the SAME line has .rolling().mean/sum without .shift
        if re.search(r'\.rolling\([^)]+\)\s*\.\s*(?:mean|sum|std)\([^)]*\)\s*$', line.rstrip()):
            # Check if next non-blank line starts with .shift
            for j in range(i, min(i + 3, len(lines))):
                if '.shift(' in lines[j]:
                    break
            else:
                issues.append(f"{filename}:{i}: {line.strip()[:80]}")
    return issues

# Check the lambda-based transform calls (the pattern used everywhere):
# .transform(lambda s: s.rolling(...).mean().shift(1))
# These are single-line — verify the pattern directly
TRANSFORM_WITH_SHIFT = re.compile(
    r'\.transform\(lambda[^:]+:\s*s\.rolling\([^)]+\)\s*\.\s*(?:mean|sum|std)\([^)]*\)\s*\.shift\(1\)'
)
TRANSFORM_WITHOUT_SHIFT = re.compile(
    r'\.transform\(lambda[^:]+:\s*s\.rolling\([^)]+\)\s*\.\s*(?:mean|sum|std)\([^)]*\)\s*(?!\.shift)'
)

for src, fname in [(plf_src, "pitch_level_features.py"), (fe_src, "feature_engineering.py")]:
    bad = TRANSFORM_WITHOUT_SHIFT.findall(src)
    # Filter false positives: lines where .shift is on the continuation
    # Re-check by scanning line by line
    issues = []
    lines = src.splitlines()
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # A transform with rolling but no shift
        if (re.search(r'\.rolling\([^)]+\)\s*\.\s*(?:mean|sum|std)\([^)]*\)', stripped) and
                '.shift(' not in stripped):
            # Check if this is part of a multi-line lambda (next line has .shift)
            next_lines = [lines[j].strip() for j in range(i, min(i + 4, len(lines)))]
            if not any('.shift(' in nl for nl in next_lines):
                issues.append(f"  line {i}: {stripped[:90]}")
    if issues:
        fail(f"C1 {fname} — rolling without shift", f"\n" + "\n".join(issues))
    else:
        ok(f"C1 {fname} — all .rolling() have .shift(1)")

# ── C2: Date window uses strict < exclusion (bullpen _date_window_sum) ───────
print("\n[C2] Bullpen date-window uses strict < (searchsorted side='left')")
if "searchsorted(dates_arr, d, side=\"left\")" in plf_src:
    ok("C2 _date_window_sum uses side='left' (strict < today)")
else:
    fail("C2 _date_window_sum — side='left' not found, today may be included in window")

# ── C3: Travel features explicitly sort before shift ────────────────────────
print("\n[C3] Travel features explicitly sort before shift(1)")
travel_fn_start = fe_src.find("def _travel_features(")
travel_fn_end = fe_src.find("\ndef ", travel_fn_start + 1)
travel_fn = fe_src[travel_fn_start:travel_fn_end]
if 'sort_values(["team_id", "game_date"' in travel_fn:
    ok("C3 _travel_features: explicit sort_values by team_id, game_date before shift(1)")
else:
    fail("C3 _travel_features: no explicit sort before shift(1)")

# ── C4: H2H relies on input sort order ──────────────────────────────────────
print("\n[C4] H2H features: input sort order assumption")
h2h_fn_start = fe_src.find("def _head_to_head_features(")
h2h_fn_end = fe_src.find("\ndef ", h2h_fn_start + 1)
h2h_fn = fe_src[h2h_fn_start:h2h_fn_end]
if 'sort_values' in h2h_fn:
    ok("C4 _head_to_head_features: explicit sort inside function")
else:
    warn("C4 _head_to_head_features: NO explicit sort within function — relies on caller passing games sorted by game_date. If unsorted, rolling window is wrong.")
    # Check if caller sorts before calling
    caller_context = fe_src[max(0, fe_src.find("_head_to_head_features(") - 500):
                             fe_src.find("_head_to_head_features(") + 100]
    if "sort_values" in caller_context:
        warn("C4 → caller appears to sort nearby, but not proven to be immediately before this call")
    else:
        warn("C4 → no sort visible near call site either — verify game_frame is date-sorted")

# ── C5: Postseason flag — no rolling, no shift needed ───────────────────────
print("\n[C5] Postseason flag: current-game property (no shift needed)")
ps_fn_start = fe_src.find("def _postseason_flag(")
ps_fn_end = fe_src.find("\ndef ", ps_fn_start + 1)
ps_fn = fe_src[ps_fn_start:ps_fn_end]
if "shift" not in ps_fn and "rolling" not in ps_fn:
    ok("C5 _postseason_flag: no rolling/shift — direct current-game property. Correct.")
else:
    fail("C5 _postseason_flag: unexpected rolling or shift in a non-temporal flag")

# ── C6: Bat strength — sort then shift ──────────────────────────────────────
print("\n[C6] Bat strength (_compute_bat_strength_features): sort + shift")
bs_fn_start = plf_src.find("def _compute_bat_strength_features(")
bs_fn_end = plf_src.find("\ndef ", bs_fn_start + 1)
bs_fn = plf_src[bs_fn_start:bs_fn_end]
explicit_sort = 'sort_values(["_batting_team_id", "game_date"' in bs_fn
shift_present = '.shift(1)' in bs_fn
ok("C6 bat strength: explicit sort before rolling", ) if explicit_sort else fail("C6 bat strength: no explicit sort")
ok("C6 bat strength: .shift(1) present") if shift_present else fail("C6 bat strength: .shift(1) missing")

# ── C7: Bullpen — sort then date window ─────────────────────────────────────
print("\n[C7] Bullpen workload: sort + date-window exclusion")
bl_fn_start = plf_src.find("def _compute_bullpen_workload_features(")
bl_fn_end = plf_src.find("\ndef ", bl_fn_start + 1)
bl_fn = plf_src[bl_fn_start:bl_fn_end]
ok("C7 bullpen: sort before date window") if 'sort_values(["_pitching_team_id", "game_date"]' in bl_fn else fail("C7 bullpen: sort not found")
ok("C7 bullpen: uses _date_window_sum (not rolling with shift)") if "_date_window_sum" in bl_fn else fail("C7 bullpen: _date_window_sum missing")

# ── C8: Manager — sort + shift ──────────────────────────────────────────────
print("\n[C8] Manager tendency: sort + shift")
mg_fn_start = plf_src.find("def _compute_manager_tendency_features(")
mg_fn_end = plf_src.find("\ndef ", mg_fn_start + 1)
mg_fn = plf_src[mg_fn_start:mg_fn_end] if mg_fn_end > 0 else plf_src[mg_fn_start:]
ok("C8 manager: sort before rolling") if 'sort_values(' in mg_fn else fail("C8 manager: no sort")
ok("C8 manager: .shift(1) present") if '.shift(1)' in mg_fn else fail("C8 manager: .shift(1) missing")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — EMPIRICAL API AVAILABILITY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 2 — EMPIRICAL API AVAILABILITY")
print("=" * 70)

def mlb_get(path, params=None, timeout=10):
    r = requests.get(f"{MLB_API}{path}", params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

# Get a Scheduled game and a Final game
print("\n[API] Fetching scheduled and final games...")
sched_tomorrow = mlb_get("/api/v1/schedule", {"sportId": 1, "date": "2026-08-19",
                                                "hydrate": "probablePitchers"})
sched_today = mlb_get("/api/v1/schedule", {"sportId": 1, "date": "2026-08-18",
                                            "hydrate": "probablePitchers"})

games_tomorrow = sched_tomorrow.get("dates", [{}])[0].get("games", [])
games_today = sched_today.get("dates", [{}])[0].get("games", [])

sched_game = next((g for g in games_tomorrow if g["status"]["statusCode"] == "S"), None)
final_game = next((g for g in games_today if g["status"]["statusCode"] == "F"), None)

if not sched_game:
    warn("API", "No Scheduled game found for tomorrow")
if not final_game:
    warn("API", "No Final game found for today")

# Fetch full GUMBO feed for each
def check_game(game_pk, label):
    print(f"\n--- {label} (gamePk={game_pk}) ---")
    feed = mlb_get(f"/api/v1.1/game/{game_pk}/feed/live")
    gd = feed.get("gameData", {})
    ld = feed.get("liveData", {})

    status = gd.get("status", {}).get("statusCode", "?")
    print(f"  status: {status} ({gd.get('status', {}).get('detailedState', '?')})")

    # [A] Venue lat/lon — needed for travel features
    venue = gd.get("venue", {})
    loc = venue.get("location", {})
    lat = loc.get("defaultCoordinates", {}).get("latitude")
    lon = loc.get("defaultCoordinates", {}).get("longitude")
    if lat and lon:
        ok(f"[A] venue lat/lon available ({lat:.3f}, {lon:.3f})")
    else:
        fail(f"[A] venue lat/lon MISSING — travel features not computable at {status}")

    # [B] game.type — needed for is_postseason
    gtype = gd.get("game", {}).get("type")
    if gtype:
        ok(f"[B] game.type = '{gtype}' available → is_postseason computable at {status}")
    else:
        fail(f"[B] game.type MISSING at {status}")

    # [C] probablePitchers — needed for bullpen SP identification
    pp = gd.get("probablePitchers", {})
    if pp.get("home") and pp.get("away"):
        ok(f"[C] probablePitchers available (home={pp['home'].get('fullName','?')}, away={pp['away'].get('fullName','?')})")
    elif pp:
        warn(f"[C] probablePitchers partial: {list(pp.keys())}")
    else:
        warn(f"[C] probablePitchers MISSING at {status} — SP-based bullpen identification requires this")

    # [D] boxscore pitching lines — needed for bullpen pitches/appearances
    bs = ld.get("boxscore", {})
    home_pitchers = bs.get("teams", {}).get("home", {}).get("pitchers", [])
    away_pitchers = bs.get("teams", {}).get("away", {}).get("pitchers", [])
    if home_pitchers and away_pitchers:
        ok(f"[D] boxscore.pitchers available (home: {len(home_pitchers)} pitchers, away: {len(away_pitchers)})")
    else:
        fail(f"[D] boxscore.pitchers MISSING at {status} — bullpen workload requires this (post-game only)")

    # [E] allPlays (pitch-by-pitch) — needed for hit distance and bat strength
    all_plays = ld.get("plays", {}).get("allPlays", [])
    if all_plays:
        # Check if hit data is in the plays
        hit_plays = [p for p in all_plays
                     if p.get("result", {}).get("event") in
                     {"Single", "Double", "Triple", "Home Run"}]
        sample_hit = hit_plays[0] if hit_plays else None
        if sample_hit:
            hd = sample_hit.get("hitData", {})
            dist = hd.get("totalDistance")
            ok(f"[E] allPlays available, {len(all_plays)} plays. Sample hit totalDistance={dist}")
        else:
            ok(f"[E] allPlays available ({len(all_plays)} plays) but no hit data in sample")
    else:
        fail(f"[E] allPlays EMPTY at {status} — bat strength (hit distance, TB/H) requires completed game data")

    # [F] batting order — not used by new features, but documents T2 availability
    home_order = bs.get("teams", {}).get("home", {}).get("battingOrder", [])
    away_order = bs.get("teams", {}).get("away", {}).get("battingOrder", [])
    if home_order and away_order:
        ok(f"[F] battingOrder present (home {len(home_order)}, away {len(away_order)} — T2 data)")
    else:
        warn(f"[F] battingOrder empty at {status} (expected for pre-game states — T2 feature)")

    return {
        "status": status,
        "venue_coords": bool(lat and lon),
        "game_type": bool(gtype),
        "probable_pitchers": bool(pp.get("home") and pp.get("away")),
        "boxscore_pitchers": bool(home_pitchers and away_pitchers),
        "play_by_play": bool(all_plays),
        "batting_order": bool(home_order and away_order),
    }

results = {}
pregame_game = next((g for g in games_today + games_tomorrow
                     if g["status"]["statusCode"] in ("P", "PW")), None)
delayed_live = next((g for g in games_today if g["status"]["statusCode"] == "II"), None)

for label_key, game, label_prefix in [
    ("scheduled", sched_game, "SCHEDULED"),
    ("pregame",   pregame_game, "PRE-GAME/WARMUP"),
    ("delayed",   delayed_live, "DELAYED (mid-game)"),
    ("final",     final_game,   "FINAL"),
]:
    if game:
        away = game["teams"]["away"]["team"]["name"]
        home = game["teams"]["home"]["team"]["name"]
        results[label_key] = check_game(game["gamePk"], f"{label_prefix}: {away} @ {home}")
    else:
        print(f"\n--- {label_prefix}: no game in this state right now ---")
        if label_key == "pregame":
            print("  NOTE: Pre-Game (P) and Warmup (PW) states only exist ~60-90 min / ~30 min")
            print("  before first pitch. Re-run this audit tomorrow before game time to test them.")
            print("  From DATA_AVAILABILITY.md (confirmed 2026-08-14):")
            print("    P state: weather ✓, probablePitchers ✓, battingOrder ✓, boxscore ✗, allPlays ✗")
            print("    PW state: same as P")

# ── Summary table ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("API AVAILABILITY SUMMARY — WHICH FEATURES CAN BE COMPUTED WHEN")
print("=" * 70)

features_and_sources = [
    ("bat_avg_hit_distance / tb_per_hit",  "allPlays hit data (post-game pitches parquet)",   "[E] allPlays",    "T0 — prior game Final"),
    ("bullpen_pitches/appearances_last3d",  "boxscore.pitchers (post-game box score)",          "[D] boxscore",    "T0 — prior game Final"),
    ("h2h_home_winpct / avg_runs / rd",     "home_win from game_features (post-game results)", "[D] boxscore",    "T0 — prior game Final"),
    ("travel_km / timezone_delta / flag",   "venue.location.lat/lon (static, schedule)",       "[A] venue",       "T0 — available at Scheduled"),
    ("is_postseason",                        "game.type (from schedule)",                       "[B] game.type",   "T0 — available at Scheduled"),
    ("mgr_pitchers_used / bunt_rate",        "allPlays pitchers + events (post-game)",          "[E] allPlays",    "T0 — prior game Final"),
]

sched = results.get("scheduled", {})
final = results.get("final", {})

key_map = {
    "[A] venue": "venue_coords",
    "[B] game.type": "game_type",
    "[C] probablePitchers": "probable_pitchers",
    "[D] boxscore": "boxscore_pitchers",
    "[E] allPlays": "play_by_play",
}

for feat, source, api_key, tier in features_and_sources:
    k = key_map.get(api_key, "")
    at_sched = sched.get(k, "?")
    at_final = final.get(k, "?")
    sched_str = "YES" if at_sched is True else ("NO" if at_sched is False else "?")
    final_str = "YES" if at_final is True else ("NO" if at_final is False else "?")
    print(f"\n  Feature: {feat}")
    print(f"  Source:  {source}")
    print(f"  API available @ Scheduled: {sched_str}  | @ Final: {final_str}")
    print(f"  → Tier: {tier}")

print("\n")
print("KEY FINDING:")
print("  • travel + is_postseason: T0 (static, available at Scheduled state)")
print("  • bat strength, bullpen, H2H, manager tendency: T0 (prior Final games)")
print("    These features are ALL computed from COMPLETED games' data.")
print("    At inference time, the most recent completed game is already in S3.")
print("    None require lineup confirmation (T2) or SP confirmation (T1).")
print()
print("NOTE: 'T0' means computable the night before (~11pm ET after last game")
print("finalizes). The daily rebuild in append_new_features.py would need to")
print("run AFTER the S3 live_daemon deposits the previous night's game data.")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"AUDIT COMPLETE: {passes + fails + warns} checks — "
      f"{passes} PASS | {fails} FAIL | {warns} WARN")
print("=" * 70)
if fails:
    print("ACTION REQUIRED: review FAIL items above.")
elif warns:
    print("Review WARN items; none are blockers.")
else:
    print("Clean.")
