"""Route features to model families based on importance analysis results.

Features are classified into groups based on which importance methods they
pass, then routed to appropriate model families.
"""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def route_features(
    mdi: pd.DataFrame,
    mda: pd.DataFrame,
    sfi: pd.DataFrame,
    top_pct: float = 0.5,
) -> dict[str, list[str]]:
    """Classify features and route to model families.

    Categories:
    - accepted: passes MDI + MDA + SFI (all methods agree)
    - complementary: MDI + MDA pass, SFI fails (interaction features)
    - linear_only: SFI passes, MDI/MDA fail (global linear signal)
    - absorbed: MDI passes only (redundant with accepted features)
    - rejected: fails all methods

    Routing:
    - trees: accepted + complementary
    - linear: accepted + linear_only
    - diversity: accepted + absorbed (for ensemble diversity)
    - full: all non-rejected
    """
    # Determine thresholds based on top_pct
    n_features = len(mdi)
    top_n = max(int(n_features * top_pct), 10)

    mdi_pass = set(mdi.head(top_n)["feature"].tolist())
    mda_pass = set(mda.head(top_n)["feature"].tolist())
    sfi_pass = set(sfi.head(top_n)["feature"].tolist())

    all_features = set(mdi["feature"].tolist())

    # Classification
    accepted = mdi_pass & mda_pass & sfi_pass
    complementary = (mdi_pass & mda_pass) - sfi_pass
    linear_only = sfi_pass - mdi_pass - mda_pass
    absorbed = mdi_pass - mda_pass - sfi_pass
    rejected = all_features - mdi_pass - mda_pass - sfi_pass

    log.info(f"Feature routing: accepted={len(accepted)}, complementary={len(complementary)}, "
             f"linear_only={len(linear_only)}, absorbed={len(absorbed)}, rejected={len(rejected)}")

    # Build feature subsets for routing
    routing = {
        "trees": sorted(accepted | complementary),
        "linear": sorted(accepted | linear_only),
        "diversity": sorted(accepted | absorbed),
        "full": sorted(all_features - rejected),
        "accepted": sorted(accepted),
        "complementary": sorted(complementary),
        "linear_only": sorted(linear_only),
        "absorbed": sorted(absorbed),
        "rejected": sorted(rejected),
    }

    return routing


def build_feature_subsets(
    feature_columns: list[str],
    routing: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Build named feature subsets for the model candidate pool.

    Uses column name patterns to identify feature groups, then intersects
    with the importance-based routing.
    """
    subsets = {}

    # Pattern-based groups
    subsets["ratings_only"] = [c for c in feature_columns
                               if any(x in c for x in ("srs", "elo", "wolfe", "log5", "pythag", "bsr"))]
    subsets["rolling_short"] = [c for c in feature_columns if "roll5_" in c]
    subsets["rolling_medium"] = [c for c in feature_columns if "roll10_" in c]
    subsets["rolling_long"] = [c for c in feature_columns if "roll20_" in c]
    subsets["pitching"] = [c for c in feature_columns
                           if any(x in c for x in ("sp_", "era", "whip", "fip", "k9", "bb9"))]
    subsets["context"] = [c for c in feature_columns
                          if any(x in c for x in ("temp", "dome", "night", "rest", "games_last",
                                                   "park_factor", "doubleheader"))]
    subsets["momentum"] = [c for c in feature_columns
                           if any(x in c for x in ("streak", "winpct", "rd_mean", "rd_std"))]
    subsets["efficiency"] = [c for c in feature_columns
                             if any(x in c for x in ("ops", "fip", "whip", "iso", "babip"))]
    subsets["matchup"] = [c for c in feature_columns if "h2h" in c]

    # Composite groups
    subsets["ratings_plus_momentum"] = subsets["ratings_only"] + subsets["momentum"]
    subsets["short_window"] = subsets["rolling_short"] + subsets["momentum"]
    subsets["long_window"] = subsets["rolling_long"] + subsets["ratings_only"]

    # Importance-based
    if routing:
        subsets["all_survivors"] = routing.get("full", feature_columns)
        subsets["trees_routed"] = routing.get("trees", feature_columns)
        subsets["linear_routed"] = routing.get("linear", feature_columns)

    # Filter empty subsets
    subsets = {k: v for k, v in subsets.items() if v}

    log.info(f"Built {len(subsets)} feature subsets: {[(k, len(v)) for k, v in subsets.items()]}")
    return subsets
