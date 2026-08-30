"""
Surgically append the as-of weather arrays to EXISTING prepared-tensor splits.

Only the weather channel changed, so rebuilding the full 27 GB prepared set
(hours of context/player tensor work) is waste — this writes exactly the two
arrays precollate would have produced, in the same order the split already
uses (game_pks.npy / sample_to_game.npy / prefix_length.npy), and patches the
manifest. PreparedDataset then serves the [7,99] decision row per sample.

prefix_length.npy stores the FULL cut count per sample (not the truncated
window length — verified precollate.py:225), which is exactly the semantic
_get_wx_decision_hour expects (offsets[prefix_len-1]).

Run on the GPU box:
  python -m mlb_dl.append_weather_asof_to_prepared \
      --feature-store /mnt/fast/feature_store \
      --prepared-dir /mnt/fast/prepared_tensors
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from .train_unified import _load_weather_asof_artifacts
from .weather_asof import ASOF_CHANNELS, N_DECISIONS, N_TARGET_HOURS

log = logging.getLogger("APPEND_WX_ASOF")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

# prefix_length and wx_hour_offset both derive from the same pitches parquet, so a prefix
# reaching past the end of a game's offsets array should be impossible. When it happens
# the two artifacts came from different snapshots -- refreshing the feature store without
# rebuilding the offsets does exactly that -- and the clamp below quietly hands every
# affected sample the game's FINAL decision hour, the leakiest row available. The true
# rate is 0, so 1% is pure headroom for isolated edge cases.
MAX_OFFSET_TRUNCATION_RATE = 0.01

# A rate over a handful of samples is noise (one clamped sample out of four is 25%), so
# the guard needs volume before it judges. Real splits carry 41k-315k samples.
MIN_SAMPLES_FOR_TRUNCATION_RATE = 1000


def assert_norm_sidecar(fs_path: Path) -> None:
    """Refuse to bake weather into the prepared tensors before norm stats exist.

    _load_weather_asof_artifacts z-scores only when weather_asof_norm.json is present;
    without it, it logs a warning and returns raw physical units. Appending those writes
    raw units permanently into weather_asof.npy, and training loads the .npy without
    re-checking -- so the treatment arm trains on unnormalized weather and the A/B reads
    as "weather did not help" rather than failing.

    norm-stats can only run once every season is built, so this ordering is easy to get
    wrong and expensive to detect afterwards.
    """
    norm_file = Path(fs_path) / "weather_asof_norm.json"
    if not norm_file.exists():
        raise SystemExit(
            f"REFUSING to append: {norm_file} does not exist, so the artifact would be "
            f"written in RAW physical units (temperature ~300, pressure ~101325) and the "
            f"treatment arm would train unnormalized. Build every season, then run "
            f"`python -m mlb_dl.build_weather_asof norm-stats`, then re-run this."
        )


def append_split(split_dir: Path, asof: dict, offsets: dict) -> None:
    with open(split_dir / "manifest.json") as f:
        manifest = json.load(f)
    game_pks = np.load(split_dir / "game_pks.npy")
    sample_to_game = np.load(split_dir / "sample_to_game.npy")
    prefix_length = np.load(split_dir / "prefix_length.npy")

    n_games = len(game_pks)
    T = np.zeros((n_games, N_DECISIONS, N_TARGET_HOURS, ASOF_CHANNELS), dtype=np.float32)
    covered = 0
    for i, pk in enumerate(game_pks):
        t = asof.get(int(pk))
        if t is not None:
            T[i] = t
            covered += 1
    cov = covered / max(n_games, 1)
    log.info("%s: weather_asof %d/%d games (%.1f%%)", split_dir.name, covered, n_games, 100 * cov)
    if cov < 0.95:
        raise SystemExit(f"{split_dir.name}: as-of coverage {cov:.1%} < 95% — "
                         f"refusing to write a mostly-empty weather channel")
    np.save(split_dir / "weather_asof.npy", T)
    del T

    wx_d = np.zeros(len(sample_to_game), dtype=np.int8)
    n_missing_off = 0
    n_truncated = 0
    for i in range(len(sample_to_game)):
        plen = int(prefix_length[i])
        if plen <= 0:
            continue
        offs = offsets.get(int(game_pks[sample_to_game[i]]))
        if offs is None or len(offs) == 0:
            n_missing_off += 1
            continue
        if plen - 1 > len(offs) - 1:
            n_truncated += 1
        wx_d[i] = min(max(int(offs[min(plen - 1, len(offs) - 1)]), 0), 6)

    trunc_rate = n_truncated / max(len(wx_d), 1)
    if (len(wx_d) >= MIN_SAMPLES_FOR_TRUNCATION_RATE
            and trunc_rate > MAX_OFFSET_TRUNCATION_RATE):
        raise SystemExit(
            f"{split_dir.name}: decision hour TRUNCATED for {n_truncated}/{len(wx_d)} "
            f"samples ({trunc_rate:.1%}) whose prefix runs past the end of their game's "
            f"offsets array, over the "
            f"{MAX_OFFSET_TRUNCATION_RATE:.0%} tolerance. prefix_length and "
            f"wx_hour_offset must come from the same pitches snapshot; rebuild "
            f"`build_weather_asof pitch-offsets` for the affected seasons, or re-run "
            f"precollate, so the clamp does not hand these samples the game's final "
            f"(leakiest) decision hour."
        )
    if n_truncated:
        log.warning("%s: %d sample(s) truncated/clamped to the last offset (%.3f%%) — under "
                    "tolerance but nonzero; both artifacts should come from the same "
                    "pitches snapshot", split_dir.name, n_truncated, 100 * trunc_rate)
    np.save(split_dir / "wx_decision_hour.npy", wx_d)
    log.info("%s: wx_decision_hour for %d samples (%d without offsets -> d=0); "
             "d distribution: %s", split_dir.name, len(wx_d), n_missing_off,
             np.bincount(wx_d, minlength=7).tolist())

    manifest["has_weather_asof"] = True
    manifest["asof_channels"] = int(ASOF_CHANNELS)
    manifest["asof_decisions"] = int(N_DECISIONS)
    manifest["asof_target_hours"] = int(N_TARGET_HOURS)
    with open(split_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("%s: manifest patched", split_dir.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-store", required=True)
    ap.add_argument("--prepared-dir", required=True)
    args = ap.parse_args()
    # Before the loader runs: it silently falls back to raw units when the sidecar is
    # absent, and this script would bake that into the tensors permanently.
    assert_norm_sidecar(Path(args.feature_store))
    asof, offsets = _load_weather_asof_artifacts(Path(args.feature_store))
    if not asof:
        raise SystemExit("no weather_asof artifact under the feature store")
    log.info("artifact: %d games, offsets for %d games", len(asof), len(offsets))
    for split in ("train", "val", "test"):
        append_split(Path(args.prepared_dir) / split, asof, offsets)
    log.info("done — fit-unified with this --prepared-dir now trains as-of weather")


if __name__ == "__main__":
    main()
