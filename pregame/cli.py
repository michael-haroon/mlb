#!/usr/bin/env python3
"""Standalone CLI for raw EDA analysis."""

import argparse
import json
from pathlib import Path

from .raw_eda import run_raw_eda


def season_range(start, end):
    """Return list of seasons from start to end (inclusive)."""
    if start is None:
        start = 2015
    if end is None:
        end = 2026
    return list(range(start, end + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="MLB raw data EDA (classical ML analysis)")
    parser.add_argument("--source", required=True, help="S3 or local path to raw data")
    parser.add_argument("--output", required=True, help="Output directory for EDA artifacts")
    parser.add_argument("--season-start", type=int, help="First season to ingest (default: 2015)")
    parser.add_argument("--season-end", type=int, help="Last season to ingest (default: 2026)")

    args = parser.parse_args()

    seasons = season_range(args.season_start, args.season_end)
    outputs = run_raw_eda(
        source_uri=args.source,
        output_dir=args.output,
        seasons=seasons,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
