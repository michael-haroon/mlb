"""
Build station_venue_map.json: every population venue -> primary + backup ASOS station.

The map is the ground-truth join key for the OBSERVED channel of the as-of weather
tensor (see fetch_asos_obs.py). Regenerate when the venue population changes
(new ballpark, relocation) — output is committed at data_curation/station_venue_map.json.

Hard-won rules baked in:
  1. The IEM networks CSV is NOT a reliable eligibility oracle — it files each stid
     under ONE network, so OAK/MSP/LGA live under *_DCP and Chicago Midway under
     TWDR (with degraded coords) despite all having full hourly METAR in the
     archive. So there is NO network filter: eligibility comes from probing the
     asos.py archive for actual 2015 AND 2024 hourly coverage.
  2. A failing probe must be retried: v2 of this map rejected SAN and BJC on
     transient IEM errors (re-probe returned 86/949 and 117/121 rows), silently
     degrading Petco and Coors to stations 2-4x farther away.
  3. The raw pool needs shape filters or dense junk clusters starve real airports
     out of the nearest-N scan (v3 gave Wrigley ZERO stations because 20 digit-id
     COOP/hydro sensors sit closer than Midway): METAR ids are 3-4 pure-alpha
     chars. K-prefixed ICAO aliases (KSAN vs SAN) are the same physical station
     and must be deduped or primary == backup.

Usage:
  conda run -n pred python data_curation/scripts/build_station_venue_map.py
"""
import io
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parent.parent.parent
FEATURE_STORE = REPO / "deep_learning" / "feature_store"
OUT_PATH = REPO / "data_curation" / "station_venue_map.json"

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
# All IEM networks in one CSV (stid, station_name, lat, lon, begints, endts, iem_network)
IEM_NETWORKS_URL = "https://mesonet.agron.iastate.edu/sites/networks.php?network=_ALL_&format=csv&nohtml=on"
STATIONS_CACHE = Path("/tmp/iem_stations.csv")

# Population definition must match the training population (2015+ Statcast floor,
# game types R/F/D/L/W) or the map can miss venues that only host early-era games.
POP_MIN_DATE = "2015-01-01"
POP_GAME_TYPES = ["R", "F", "D", "L", "W"]

# Venues whose game_meta rows carry no coordinates in any season.
# Turner Field is 1,671 population games (Braves through 2016) — not optional.
MANUAL_VENUE_COORDS = {
    16:   dict(name="Turner Field", lat=33.73472, lon=-84.38917, tz="America/New_York"),
    2701: dict(name="Estadio de Beisbol Monterrey", lat=25.72070, lon=-100.31170, tz="America/Monterrey"),
    5010: dict(name="Fort Bragg Field", lat=35.14010, lon=-78.99460, tz="America/New_York"),
    5340: dict(name="Estadio Alfredo Harp Helu", lat=19.40420, lon=-99.09070, tz="America/Mexico_City"),
}

# Stations with archive data that the networks CSV mislabels/degrades.
# MDW: filed under TWDR at (41.65, -87.73); true field location from the KMDW row.
MANUAL_EXTRA_STATIONS = [
    dict(stid="MDW", station_name="Chicago Midway", lat=41.78597, lon=-87.75242,
         elev=189.0),
]

PROBE_CACHE_PATH = Path("/tmp/iem_probe_cache.json")

logger = logging.getLogger("STATION_MAP")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    os.makedirs(REPO / "data" / "logs", exist_ok=True)
    fh = logging.FileHandler(REPO / "data" / "logs" / "station_venue_map.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[MAP] %(levelname)s %(message)s"))
    logger.addHandler(ch)


def load_population_venues() -> tuple[dict, pd.Series]:
    gm = pd.read_parquet(FEATURE_STORE / "game_meta.parquet", columns=[
        "game_pk", "game_date", "game_type_code", "venue_id", "venue_name",
        "venue_latitude", "venue_longitude", "venue_timezone"])
    gm["game_date"] = pd.to_datetime(gm["game_date"])
    pop = gm[(gm["game_date"] >= POP_MIN_DATE) & gm["game_type_code"].isin(POP_GAME_TYPES)]
    counts = pop["venue_id"].value_counts()

    v = (pop.dropna(subset=["venue_latitude"])
            .sort_values("game_date").drop_duplicates("venue_id", keep="last"))
    venues = {int(r.venue_id): dict(name=r.venue_name, lat=float(r.venue_latitude),
                                    lon=float(r.venue_longitude), tz=r.venue_timezone)
              for _, r in v.iterrows()}
    for vid, m in MANUAL_VENUE_COORDS.items():
        if vid in counts.index and vid not in venues:
            venues[vid] = m
    unresolved = set(counts.index) - set(venues)
    if unresolved:
        logger.warning(f"venues with no coords and no manual entry: {sorted(unresolved)} "
                       f"({counts.loc[list(unresolved)].sum()} games) — add to MANUAL_VENUE_COORDS")
    logger.info(f"venues resolved: {len(venues)}/{len(counts)} "
                f"({counts.loc[[i for i in counts.index if i in venues]].sum()} of {counts.sum()} games)")
    return venues, counts


def load_candidate_stations() -> pd.DataFrame:
    if not STATIONS_CACHE.exists():
        logger.info("downloading IEM networks CSV ...")
        r = requests.get(IEM_NETWORKS_URL, timeout=120)
        r.raise_for_status()
        STATIONS_CACHE.write_text(r.text)
    st = pd.read_csv(STATIONS_CACHE)
    # No network/begints filter (rule 1) — the coverage probe is the only oracle.
    st = st.dropna(subset=["lat", "lon"])
    # Rule 3a: METAR ids are 3-4 pure-alpha chars; digit-bearing ids are
    # COOP/hydro sensors that would starve the nearest-N scan.
    st = st[st["stid"].astype(str).str.fullmatch(r"[A-Za-z]{3,4}")]
    # Rule 3b: drop K-prefixed ICAO aliases when the bare id is also present
    # (KSAN and SAN are one physical station).
    ids = set(st["stid"])
    st = st[~(st["stid"].str.len().eq(4) & st["stid"].str.startswith("K")
              & st["stid"].str[1:].isin(ids))]
    # Manual entries override the CSV row for the same stid (its coords may be degraded).
    manual_ids = {m["stid"] for m in MANUAL_EXTRA_STATIONS}
    st = st[~st["stid"].isin(manual_ids)]
    st = pd.concat([st[["stid", "station_name", "lat", "lon", "elev"]],
                    pd.DataFrame(MANUAL_EXTRA_STATIONS)], ignore_index=True)
    logger.info(f"candidate stations: {len(st)}")
    return st


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# Disk-backed so iterating on pool logic doesn't re-pay ~1s x hundreds of probes.
# A positive probe can't be a transient artifact (the rows existed), so caching
# across runs is safe; failures are re-tried within a run anyway.
def _load_probe_cache() -> dict:
    if PROBE_CACHE_PATH.exists():
        with open(PROBE_CACHE_PATH) as f:
            return {tuple(k.split("|")): v for k, v in json.load(f).items()}
    return {}


def _save_probe_cache() -> None:
    with open(PROBE_CACHE_PATH, "w") as f:
        json.dump({f"{k[0]}|{k[1]}": v for k, v in _probe_cache.items()}, f)


_probe_cache: dict[tuple, int] = _load_probe_cache()


def _probe_once(stid: str, year: int) -> int:
    r = requests.get(IEM_ASOS_URL, params=dict(
        station=stid, data="tmpf", year1=year, month1=6, day1=1,
        year2=year, month2=6, day2=4, tz="Etc/UTC", format="onlycomma",
        missing="M", trace="T"), timeout=60)
    try:
        return max(len(pd.read_csv(io.StringIO(r.text))), 0)
    except Exception:
        return 0


def probe(stid: str, year: int) -> int:
    """Hourly rows in a 3-day archive sample — <60 means unusable that era (rule 2:
    retry before rejecting, transient IEM errors produce false zeros)."""
    k = (stid, str(year))
    if k not in _probe_cache:
        n = _probe_once(stid, year)
        if n < 60:
            time.sleep(3.0)
            n = max(n, _probe_once(stid, year))
        logger.debug(f"probe {stid} {year}: {n} rows")
        _probe_cache[k] = n
        _save_probe_cache()
        time.sleep(0.5)
    return _probe_cache[k]


def station_ok(stid: str) -> bool:
    # >=60 of ~72 possible hourly reports in both eras = a live, dense station.
    return probe(stid, 2015) >= 60 and probe(stid, 2024) >= 60


def main() -> None:
    venues, counts = load_population_venues()
    st = load_candidate_stations()

    out = {}
    for vid, m in venues.items():
        d = haversine_km(m["lat"], m["lon"], st["lat"].values, st["lon"].values)
        order = np.argsort(d)
        chosen = []
        for idx in order[:20]:
            s = st.iloc[idx]
            # Backup must be a physically distinct station — a second id at the
            # same field adds zero redundancy against an outage.
            if chosen and haversine_km(chosen[0][0]["lat"], chosen[0][0]["lon"],
                                       s["lat"], s["lon"]) < 1.0:
                continue
            if station_ok(s["stid"]):
                chosen.append((s, float(d[idx])))
            if len(chosen) == 2:
                break
        if len(chosen) < 2:
            logger.warning(f"{m['name']}: only {len(chosen)} validated stations in nearest 20")
            while 0 < len(chosen) < 2:
                chosen.append(chosen[0])           # backup = primary (degraded, but valid)
        if not chosen:
            logger.error(f"{m['name']}: NO validated station — venue left out of map")
            continue
        (prim, dp), (back, db) = chosen[0], chosen[1]
        out[str(vid)] = {
            "venue_name": m["name"], "venue_lat": round(m["lat"], 5),
            "venue_lon": round(m["lon"], 5), "venue_tz": m["tz"],
            "n_population_games": int(counts.get(vid, 0)),
            "primary_station": prim["stid"], "primary_name": prim["station_name"],
            "primary_km": round(dp, 1),
            # Station elevation feeds the altimeter -> station-pressure
            # conversion in weather_asof (air density and dim 13 depend on it).
            "primary_elev_m": None if pd.isna(prim.get("elev")) else round(float(prim["elev"]), 1),
            "backup_station": back["stid"], "backup_name": back["station_name"],
            "backup_km": round(db, 1),
            "backup_elev_m": None if pd.isna(back.get("elev")) else round(float(back["elev"]), 1),
        }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    logger.info(f"wrote {OUT_PATH} ({len(out)} venues)")

    for m in sorted(out.values(), key=lambda x: -x["n_population_games"]):
        logger.info(f"{m['venue_name']:42s} {m['n_population_games']:6d}  "
                    f"{m['primary_station']:5s} {m['primary_km']:5.1f}km  "
                    f"{m['backup_station']:5s} {m['backup_km']:5.1f}km")


if __name__ == "__main__":
    main()
