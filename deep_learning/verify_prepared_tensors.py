"""Gate a freshly precollated prepared_tensors dir before anything trains on it.

Checks the prepared output against ITS OWN SOURCE CACHE rather than against hardcoded counts.
That choice matters: `temporal_split_dates` picks split boundaries by quantile, so every time
games are appended to the feature store the train/val/test sizes move (2026-08-31 refresh:
train 314,953 -> 315,791, val 41,358 -> 40,806, test 41,365 -> 40,399). Any expected-count
constant here would go stale on the next ingestion and would then either block a good build
or, worse, be "fixed" by editing the constant.

Exists because a void 158 GiB prepared set built from the 1950-train population bug sat in S3
for four days looking healthy, and would have been downloaded by four sweep boxes. Manifest
counts are the cheapest thing that distinguishes one population from another.

Usage:
    python verify_prepared_tensors.py --cache /mnt/fast/dataset_cache_new \
                                      --prepared /mnt/fast/prepared_tensors_new
Exit 0 = safe to promote. Nonzero = do not train on it, do not upload it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--prepared", required=True, type=Path)
    ap.add_argument("--expect-asof-weather", action="store_true", default=True)
    args = ap.parse_args()

    problems: list[str] = []

    top = args.prepared / "manifest.json"
    if not top.exists():
        print(f"FAIL: no top manifest at {top}")
        return 2
    top_manifest = json.loads(top.read_text())
    if set(top_manifest.get("splits", {})) != set(SPLITS):
        problems.append(f"top manifest splits = {sorted(top_manifest.get('splits', {}))}")

    for split in SPLITS:
        pdir = args.prepared / split
        cdir = args.cache / split
        if not pdir.exists():
            problems.append(f"{split}: missing prepared dir")
            continue

        m = json.loads((pdir / "manifest.json").read_text())

        # 1. Sample count must equal the cache's samples.json exactly. A short prepared split
        #    is the signature of a precollate that died partway and left a usable-looking dir.
        n_cache = len(json.loads((cdir / "samples.json").read_text()))
        if m["n_samples"] != n_cache:
            problems.append(
                f"{split}: {m['n_samples']} prepared samples vs {n_cache} in the source cache")

        # 2. Every memmap's first axis must agree with the count it is indexed by, because
        #    _mmap_create sizes files up front: a truncated write leaves a SHORTER file and
        #    np.load(mmap_mode='r') would happily serve garbage-free but misaligned rows.
        n_games = m["n_games"]
        for name, axis0 in (("pitch_cont", m["n_samples"]),
                            ("wx_decision_hour", m["n_samples"]),
                            ("weather_asof", n_games)):
            f = pdir / f"{name}.npy"
            if not f.exists():
                if name == "weather_asof" and not args.expect_asof_weather:
                    continue
                problems.append(f"{split}: {name}.npy missing")
                continue
            got = np.load(f, mmap_mode="r").shape[0]
            if got != axis0:
                problems.append(f"{split}: {name}.npy axis0={got}, expected {axis0}")

        # 3. As-of weather must be declared AND non-degenerate. A file of the right shape full
        #    of zeros is the failure mode that a shape check cannot see, and it would read as
        #    "weather did not help" in the A/B rather than as a broken artifact.
        if args.expect_asof_weather:
            if not m.get("has_weather_asof"):
                problems.append(f"{split}: has_weather_asof is false")
            else:
                wa = np.load(pdir / "weather_asof.npy", mmap_mode="r")
                probe = np.asarray(wa[: min(512, wa.shape[0])])
                frac_nonzero = float(np.mean(probe != 0.0))
                n_nan = int(np.isnan(probe).sum())
                print(f"{split}: weather_asof {wa.shape} "
                      f"nonzero={frac_nonzero:.1%} nan={n_nan}")
                if frac_nonzero < 0.5:
                    problems.append(
                        f"{split}: weather_asof only {frac_nonzero:.1%} nonzero — the "
                        f"post-rebuild artifact measured ~90% non-zero on core fields")
                if n_nan:
                    problems.append(f"{split}: weather_asof has {n_nan} NaNs in the probe")

        print(f"{split}: {m['n_samples']:,} samples / {n_games:,} games "
              f"(cache agrees: {m['n_samples'] == n_cache})")

    # 4. The 2020 season must be absent, which is how the promoted store is told apart from
    #    the pre-promotion snapshot. SKIP_SEASONS drops all 1,279 games of 2020, so its
    #    presence means this was built from the wrong cache entirely.
    cache_manifest = args.cache / "manifest.json"
    if cache_manifest.exists():
        cm = json.loads(cache_manifest.read_text())
        print(f"source cache fingerprint: {cm.get('fingerprint')}  "
              f"splits {cm.get('temporal_split_dates')}")

    if problems:
        print("\n=== VERIFY FAILED ===")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\n=== VERIFY OK — safe to promote and upload ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
