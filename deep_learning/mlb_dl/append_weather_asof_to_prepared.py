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
    for i in range(len(sample_to_game)):
        plen = int(prefix_length[i])
        if plen <= 0:
            continue
        offs = offsets.get(int(game_pks[sample_to_game[i]]))
        if offs is None or len(offs) == 0:
            n_missing_off += 1
            continue
        wx_d[i] = min(max(int(offs[min(plen - 1, len(offs) - 1)]), 0), 6)
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
    asof, offsets = _load_weather_asof_artifacts(Path(args.feature_store))
    if not asof:
        raise SystemExit("no weather_asof artifact under the feature store")
    log.info("artifact: %d games, offsets for %d games", len(asof), len(offsets))
    for split in ("train", "val", "test"):
        append_split(Path(args.prepared_dir) / split, asof, offsets)
    log.info("done — fit-unified with this --prepared-dir now trains as-of weather")


if __name__ == "__main__":
    main()
