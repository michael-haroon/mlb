"""Measure how often select_asof_obs picks the BACKUP station over the primary.

select_asof_obs ends in `rows.loc[rows["valid_utc"].idxmax()]`, and load_obs_for_venues
concatenates the primary and backup frames into one, dropping which station a row came
from. So the selector resolves purely on recency, and the station map's own declaration of
which station is primary is never consulted. A backup that happens to report a few minutes
later therefore outranks a healthy primary at every hour, all season.

That is not hypothetical. Both remaining weather_asof audit failures are backup stations:
  venue 12   Tropicana Field   primary SPG (:53)  backup MCF (:56)  -- MCF's p01i gauge is
             faulted; it reported 19.63 in/h with no weather group while SPG covered the
             same hours at 1.56 in/h max with +TSRA and 0.25 SM. MCF wins by 3 minutes.
  venue 2680 Petco Park        primary SAN        backup NZY        -- NZY's thermometer
             was faulted Mar-Apr 2024, reporting -53 to -80 F with dwpf and relh NaN.

This script quantifies the blast radius before anything is changed: per venue, over the
real population game hours, what share of as-of selections land on the backup. A tiny share
would make this a curiosity; a large one makes it a systematic source-quality inversion
across the whole artifact.

FIXED 2026-08-30: select_asof_obs now prefers the primary when the frame carries a
`station_role` column, and both the training builder and the live path tag it. This script
kept its own `_role` column for reporting, which the selector ignores, so by default it
measures the OLD recency-only regime -- useful as the baseline, misleading if mistaken for
current behaviour. --tag-roles replays the regime that ships, where a backup win means the
primary was genuinely unavailable at that decision (the fallback the concat exists for).
Run both to see the fix's blast radius as a difference.

Read-only, measures only. Usage:
    python3.11 data_curation/scripts/measure_asof_backup_station_wins.py [--season 2018]
    python3.11 data_curation/scripts/measure_asof_backup_station_wins.py --tag-roles
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "deep_learning"))

from mlb_dl.build_weather_asof import (  # noqa: E402
    STATION_MAP_KEY, _read_json, _read_parquet, load_population,
)
from mlb_dl.weather_asof import (  # noqa: E402
    N_DECISIONS, TARGET_HOURS, metar_to_era5, select_asof_obs,
)


def tagged_obs(vid: int, year: int, vmap: dict, tag_roles: bool = False) -> pd.DataFrame:
    """Same frames load_obs_for_venues builds, but keeping the station role.

    `_role` is always present because the report needs it. `station_role` is what the
    SELECTOR consults, so it is added only under --tag-roles -- that is the switch between
    the two regimes, and keeping them separate columns is what lets one script measure both.
    """
    m = vmap.get(str(vid))
    if m is None:
        return pd.DataFrame()
    frames = []
    for role, st, elev in (("primary", m["primary_station"], m.get("primary_elev_m")),
                           ("backup", m["backup_station"], m.get("backup_elev_m"))):
        try:
            raw = _read_parquet(
                f"data/weather/source=asos_obs/station={st}/year={year}.parquet")
        except Exception:
            continue
        df = metar_to_era5(raw, float(elev or 0.0))
        if not len(df):
            continue
        df = df.copy()
        df["_role"] = role
        df["_station"] = st
        if tag_roles:
            df["station_role"] = role
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2018)
    ap.add_argument("--tag-roles", action="store_true",
                    help="replay the shipped regime (primary preferred) instead of the "
                         "pre-fix recency-only baseline")
    a = ap.parse_args()

    vmap = _read_json(STATION_MAP_KEY)
    pop = load_population(a.season)
    print(f"season {a.season}: {len(pop)} population games, "
          f"{pop['venue_id'].nunique()} venues  "
          f"[{'role-preferring (shipped)' if a.tag_roles else 'recency-only (pre-fix)'}]")

    per_venue: dict[int, dict] = defaultdict(lambda: {"primary": 0, "backup": 0, "none": 0})
    for vid, grp in pop.groupby("venue_id"):
        obs = tagged_obs(int(vid), a.season, vmap, tag_roles=a.tag_roles)
        if not len(obs):
            continue
        m = vmap.get(str(int(vid)), {})
        for gh in grp["game_hour_utc"]:
            gh = pd.Timestamp(gh)
            # Replay the real selection grid: every (decision, elapsed target hour) pair
            # the assembler would fill. h < d is the elapsed gate.
            for d in range(N_DECISIONS):
                dt = gh + pd.Timedelta(hours=d)
                for h in TARGET_HOURS:
                    if h >= d:
                        continue
                    row = select_asof_obs(obs, dt, gh + pd.Timedelta(hours=h))
                    if row is None:
                        per_venue[int(vid)]["none"] += 1
                    else:
                        per_venue[int(vid)][row["_role"]] += 1
        c = per_venue[int(vid)]
        tot = c["primary"] + c["backup"]
        print(f"  venue {vid:>4} {m.get('primary_station',''):<5}/"
              f"{m.get('backup_station',''):<5} "
              f"backup wins {c['backup']:>7} of {tot:>7} "
              f"({100.0*c['backup']/max(tot,1):5.1f}%)  unfilled {c['none']}")

    gp = sum(c["primary"] for c in per_venue.values())
    gb = sum(c["backup"] for c in per_venue.values())
    print(f"\nTOTAL: backup selected {gb:,} of {gp+gb:,} filled selections "
          f"({100.0*gb/max(gp+gb,1):.1f}%)")
    n_bad = sum(1 for c in per_venue.values()
                if c["backup"] > (c["primary"] + c["backup"]) * 0.5)
    print(f"venues where the BACKUP wins the majority of selections: "
          f"{n_bad} of {len(per_venue)}")


if __name__ == "__main__":
    main()
