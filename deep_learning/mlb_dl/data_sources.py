from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import glob
from pathlib import Path
from typing import Iterable


TABLE_PATTERNS = {
    "pitches": "season={season}/pitches_batch_*.parquet",
    "linescore": "season={season}/linescore_batch_*.parquet",
    "runners": "season={season}/runners_batch_*.parquet",
    "boxscore_batting": "season={season}/boxscore_batting_batch_*.parquet",
    "boxscore_pitching": "season={season}/boxscore_pitching_batch_*.parquet",
    "hits": "season={season}/hits_batch_*.parquet",
    "players": "players/players_batch_*.parquet",
    "standings": "season={season}/standings_*.parquet",
    "pitcher_stats": "season={season}/pitcher_stats_*.parquet",
    "hitter_stats": "season={season}/hitter_stats_*.parquet",
    "pitcher_splits": "season={season}/pitcher_splits_*.parquet",
    "hitter_splits": "season={season}/hitter_splits_*.parquet",
    "venue_info": "venue_info.parquet",
}

# Weather tables use venue_id/year partitioning, not season.
WEATHER_TABLE_PATTERNS = {
    "hrrr_forecast": "weather/source=hrrr_forecast/venue_id={venue_id}/year={year}.parquet",
    "hrrr_forecast_pressure": "weather/source=hrrr_forecast_pressure/venue_id={venue_id}/year={year}.parquet",
    "ecmwf_ifs_hres_forecast": "weather/source=ecmwf_ifs_hres_forecast/venue_id={venue_id}/year={year}.parquet",
    "ecmwf_ifs_hres_forecast_pressure": "weather/source=ecmwf_ifs_hres_forecast_pressure/venue_id={venue_id}/year={year}.parquet",
    "era5": "weather/source=era5/venue_id={venue_id}/year={year}.parquet",
    "era5_pressure": "weather/source=era5_pressure/venue_id={venue_id}/year={year}.parquet",
    "air_quality": "weather/source=air_quality/venue_id={venue_id}/year={year}.parquet",
    "forecast": "weather/source=forecast/venue_id={venue_id}/date={date}.parquet",
    "ensemble": "weather/source=ensemble/venue_id={venue_id}/date={date}.parquet",
}


@dataclass(frozen=True)
class ParquetCatalog:
    """Discover and read the Parquet layout produced by download_history.py."""

    base_uri: str

    @property
    def is_s3(self) -> bool:
        return self.base_uri.startswith("s3://")

    def resolve_table_paths(
        self,
        table: str,
        seasons: Iterable[int] | None = None,
    ) -> list[str]:
        if table not in TABLE_PATTERNS:
            raise KeyError(f"Unknown table {table!r}. Known tables: {sorted(TABLE_PATTERNS)}")

        pattern = TABLE_PATTERNS[table]
        if "{season}" in pattern:
            if seasons is None:
                season_patterns = [pattern.format(season="*")]
            else:
                season_patterns = [pattern.format(season=int(season)) for season in seasons]
        else:
            season_patterns = [pattern]

        paths: list[str] = []
        for rel_pattern in season_patterns:
            full_pattern = f"{self.base_uri.rstrip('/')}/{rel_pattern}"
            paths.extend(_glob_uri(full_pattern, is_s3=self.is_s3))

        return sorted(set(paths))

    def read_table(
        self,
        table: str,
        columns: list[str] | None = None,
        seasons: Iterable[int] | None = None,
    ):
        """Read a table into a pandas DataFrame.

        Each worker returns a PyArrow Table.  pa.concat_tables joins them with
        zero-copy column-chunk concatenation; a single .to_pandas() call at the
        end is the only pandas allocation, eliminating the pd.concat peak spike.

        Workers are capped at 24 — safe on instances with >=32GB RAM since all
        files end up in the tables dict before concat regardless of concurrency.
        """
        import logging
        import pandas as pd
        from concurrent.futures import ThreadPoolExecutor, as_completed

        import pyarrow as pa

        log = logging.getLogger("mlb_dl.data_sources")

        paths = self.resolve_table_paths(table, seasons=seasons)
        if not paths:
            return pd.DataFrame(columns=columns or [])

        total = len(paths)
        tables: dict[int, pa.Table] = {}
        completed = [0]
        error_count = [0]

        # PyArrow's C++-backed S3FileSystem is genuinely thread-safe unlike s3fs
        # (which wraps aiobotocore's async event loop and deadlocks across threads).
        shared_fs = None
        if self.is_s3:
            from pyarrow.fs import S3FileSystem
            shared_fs = S3FileSystem()

        def _load(args):
            idx, path = args
            return idx, _read_single_parquet_to_arrow(
                path, columns=columns, is_s3=self.is_s3, fs=shared_fs
            )

        pool = ThreadPoolExecutor(max_workers=24)
        try:
            future_to_path = {pool.submit(_load, (i, p)): p for i, p in enumerate(paths)}
            for fut in as_completed(future_to_path):
                completed[0] += 1
                try:
                    idx, tbl = fut.result()
                    tables[idx] = tbl
                except Exception as exc:
                    error_count[0] += 1
                    log.warning(f"Skipping unreadable file {future_to_path[fut]}: {exc}")
                if completed[0] % 500 == 0 or completed[0] == total:
                    rows_so_far = sum(t.num_rows for t in tables.values())
                    print(
                        f"\r  [{table}] {completed[0]}/{total} files read"
                        f" ({rows_so_far:,} rows, {error_count[0]} errors)",
                        end="", flush=True,
                    )
        except KeyboardInterrupt:
            for f in future_to_path:
                f.cancel()
            pool.shutdown(wait=False)
            raise
        else:
            pool.shutdown(wait=True)

        if total > 0:
            print()

        ordered = [tables[i] for i in sorted(tables)]
        if not ordered:
            return pd.DataFrame(columns=columns or [])
        if len(ordered) == 1:
            return ordered[0].to_pandas()

        # promote_options="default" fills null-typed columns to match the target
        # type when same-season files have schema divergence (e.g. Statcast gaps).
        combined = pa.concat_tables(ordered, promote_options="default")
        del ordered
        return combined.to_pandas()


    def read_weather(
        self,
        source: str,
        venue_ids: Iterable[int],
        years: Iterable[int],
        columns: list[str] | None = None,
    ):
        """Read weather parquets partitioned by venue_id/year.

        Weather tables use a different layout than game tables (venue × year
        rather than season-partitioned batches).
        """
        import logging
        import pandas as pd
        from concurrent.futures import ThreadPoolExecutor, as_completed

        import pyarrow as pa

        log = logging.getLogger("mlb_dl.data_sources")

        if source not in WEATHER_TABLE_PATTERNS:
            raise KeyError(f"Unknown weather source {source!r}. Known: {sorted(WEATHER_TABLE_PATTERNS)}")

        pattern_template = WEATHER_TABLE_PATTERNS[source]
        paths: list[str] = []

        for vid in venue_ids:
            for year in years:
                rel = pattern_template.format(venue_id=int(vid), year=int(year))
                full = f"{self.base_uri.rstrip('/')}/{rel}"
                resolved = _glob_uri(full, is_s3=self.is_s3)
                paths.extend(resolved)

        if not paths:
            return pd.DataFrame(columns=columns or [])

        paths = sorted(set(paths))
        tables: dict[int, pa.Table] = {}

        shared_fs = None
        if self.is_s3:
            from pyarrow.fs import S3FileSystem
            shared_fs = S3FileSystem()

        def _load(args):
            idx, path = args
            return idx, _read_single_parquet_to_arrow(
                path, columns=columns, is_s3=self.is_s3, fs=shared_fs
            )

        pool = ThreadPoolExecutor(max_workers=24)
        try:
            future_to_path = {pool.submit(_load, (i, p)): p for i, p in enumerate(paths)}
            for fut in as_completed(future_to_path):
                try:
                    idx, tbl = fut.result()
                    tables[idx] = tbl
                except Exception as exc:
                    log.warning(f"Skipping weather file {future_to_path[fut]}: {exc}")
        finally:
            pool.shutdown(wait=True)

        ordered = [tables[i] for i in sorted(tables)]
        if not ordered:
            return pd.DataFrame(columns=columns or [])
        if len(ordered) == 1:
            return ordered[0].to_pandas()

        combined = pa.concat_tables(ordered, promote_options="default")
        del ordered
        return combined.to_pandas()


def _read_single_parquet_to_arrow(
    path: str,
    columns: list[str] | None,
    is_s3: bool,
    fs,
):
    """Read one Parquet file as a PyArrow Table.

    Dictionary columns are decoded to their value types and float64 columns are
    cast to float32 entirely in the Arrow layer before any pandas object is
    created.  Unchanged fields incur zero copy cost under PyArrow's cast().
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if is_s3:
        # S3FileSystem expects "bucket/key", not "s3://bucket/key".
        # ParquetFile bypasses the dataset API, which would infer a "season"
        # partition column from the Hive path and then fail when its int64
        # conflicts with the dict<int32> stored inside the file.
        arrow_path = path.removeprefix("s3://")
        with fs.open_input_file(arrow_path) as f:
            pf = pq.ParquetFile(f)
            table = pf.read(columns=columns)
    else:
        pf = pq.ParquetFile(path)
        table = pf.read(columns=columns)

    new_fields = []
    changed = False
    for field in table.schema:
        if pa.types.is_dictionary(field.type):
            new_fields.append(field.with_type(field.type.value_type))
            changed = True
        elif field.type == pa.float64():
            new_fields.append(field.with_type(pa.float32()))
            changed = True
        else:
            new_fields.append(field)

    if changed:
        table = table.cast(pa.schema(new_fields))

    return table


def _glob_uri(pattern: str, is_s3: bool) -> list[str]:
    if is_s3:
        # Use a fresh boto3 client each call rather than the fsspec module-level
        # cached instance, which enters a bad state after the pyarrow
        # ThreadPoolExecutor runs hundreds of parallel reads via its own S3
        # client.  boto3.client() is cheap and this path is called O(tables)
        # times per run, not O(files).
        import boto3
        import fnmatch

        stripped = pattern.removeprefix("s3://")
        bucket, _, key_pattern = stripped.partition("/")
        # All patterns end with "*.parquet"; derive the longest literal prefix
        # to minimise the paginator scan.
        prefix = key_pattern.split("*")[0]

        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        results = []
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if fnmatch.fnmatch(key, key_pattern):
                    results.append(f"s3://{bucket}/{key}")
        return results

    return [str(Path(path)) for path in glob.glob(pattern) if Path(path).is_file()]


def season_range(season_start: int | None, season_end: int | None) -> list[int] | None:
    if season_start is None and season_end is None:
        return None
    if season_start is None:
        raise ValueError("season_start is required when season_end is set")
    if season_end is None:
        season_end = datetime.now().year
    if season_end < season_start:
        raise ValueError("season_end must be >= season_start")
    return list(range(int(season_start), int(season_end) + 1))
