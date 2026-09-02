"""Pipeline test: gate → routing → strategy module.

Exercises the exact code path train_target uses to resolve features per family,
without actually training. Prints every (target, family) → feature count and
the top-10 features, flags skips and zero-feature starvations.
"""
import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from classical_learning.analysis.feature_routing import (
    get_feature_set,
    get_feature_set_uncapped,
)
from classical_learning.strategy.config import IMPORTANCE_DIR, TARGETS_CLASSIFICATION
from classical_learning.strategy.models import list_families

TARGETS = [
    "home_win", "yrfi", "extra_innings", "first_5_home_win",
    "home_runs", "away_runs", "total_runs", "home_run_diff",
    "first_5_total_runs", "first_5_home_run_diff",
]

# Load sizing caps from latest sizing dir (pick highest version or default)
def load_sizing(target: str) -> dict:
    for version in ("sizing", "sizing_v9", "sizing_v8", "sizing_v7"):
        p = BASE / "pregame/artifacts" / version / f"sizing_curve_{target}.json"
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            caps = {}
            if "per_family" in data:
                for fam, fd in data["per_family"].items():
                    if "optimal_S" in fd:
                        caps[fam] = fd["optimal_S"]
            elif "optimal_S" in data and "error" not in data:
                caps["hist_gradient_boosting"] = data["optimal_S"]
            return caps
    return {}


FAMILIES = list_families()

# ── Header ──────────────────────────────────────────────────────────────────
print("=" * 80)
print("PIPELINE TEST: gate → routing → strategy")
print("=" * 80)

all_results = {}

for target in TARGETS:
    report_path = IMPORTANCE_DIR / target / "filtered" / "feature_report.csv"
    if not report_path.exists():
        print(f"\n{target}: MISSING feature_report.csv — skipping")
        continue

    filter_report = pd.read_csv(report_path, index_col="feature")
    sizing = load_sizing(target)

    tier_counts = filter_report["tier"].value_counts().to_dict()
    task = "clf" if target in TARGETS_CLASSIFICATION else "reg"

    print(f"\n{'─'*80}")
    print(f"TARGET: {target}  ({task})  |  ACCEPTED={tier_counts.get('ACCEPTED',0)}  "
          f"NEEDS_SPEC={tier_counts.get('NEEDS SPECIFICATION',0)}  "
          f"REJECTED={tier_counts.get('REJECTED',0)}")
    print(f"{'─'*80}")

    target_results = {}
    skipped = []
    for family in FAMILIES:
        if family in sizing:
            ordered = get_feature_set_uncapped(family, filter_report)
            feats = ordered[: sizing[family]]
            method = f"uncapped S*={sizing[family]}"
        else:
            feats = get_feature_set(family, filter_report)
            method = "routed"

        n = len(feats)
        target_results[family] = {"n": n, "method": method, "feats": feats}

        if n == 0:
            skipped.append(family)

    # Print compact table
    print(f"  {'Family':<28}  {'N':>4}  {'Method':<20}  Top-5 features")
    print(f"  {'─'*28}  {'─'*4}  {'─'*20}  {'─'*30}")
    for family in FAMILIES:
        r = target_results[family]
        top5 = ", ".join(r["feats"][:5]) if r["feats"] else "—"
        skip_marker = " ← SKIP" if r["n"] == 0 else ""
        print(f"  {family:<28}  {r['n']:>4}  {r['method']:<20}  {top5}{skip_marker}")

    if skipped:
        print(f"\n  ⚠ ZERO-FEATURE SKIPS ({len(skipped)}): {', '.join(skipped)}")
    else:
        print(f"\n  ✓ All {len(FAMILIES)} families receive features")

    all_results[target] = target_results

# ── Cross-target summary matrix ──────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY: feature counts per (family × target)")
print("=" * 80)

# Header row
col_w = 7
targets_short = [t.replace("first_5_", "f5_").replace("home_run_diff", "hrd")
                   .replace("home_run", "hr").replace("total_runs", "tr")
                   .replace("away_runs", "ar").replace("home_runs", "hrs")
                   .replace("home_win", "hw").replace("yrfi", "yrfi")
                   .replace("extra_innings", "ei").replace("f5_home_win", "f5hw")
                   .replace("f5_total_runs", "f5tr").replace("f5_hrd", "f5hrd") for t in TARGETS]
header = f"  {'Family':<28}" + "".join(f"  {s:>{col_w}}" for s in targets_short)
print(header)
print("  " + "─" * (28 + (col_w + 2) * len(TARGETS)))

for family in FAMILIES:
    row = f"  {family:<28}"
    for target in TARGETS:
        n = all_results.get(target, {}).get(family, {}).get("n", "?")
        marker = "!" if n == 0 else ""
        row += f"  {str(n)+marker:>{col_w}}"
    print(row)

print("\n  ! = zero features (family will be skipped by train_target)")
