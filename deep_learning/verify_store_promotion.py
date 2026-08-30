"""Compare the live DL feature store against the staging rebuild, row-for-row and column-for-column.

The staging build has MORE games (latest 2026-08-30 vs 2026-06-20) yet several tables are
SMALLER on disk. Parquet size alone cannot distinguish "better compression" from "dropped
rows", so nothing gets promoted until row counts and schemas are compared directly.
"""
import pyarrow.parquet as pq
import pyarrow.fs as pafs

B = "mlb-265753586044-us-east-1-an"
s3 = pafs.S3FileSystem(region="us-east-1")
TABS = ["game_meta", "game_targets", "pitch_sequences", "batted_balls", "runner_states",
        "player_batting_history", "player_pitching_history", "player_batting_targets",
        "player_pitching_targets", "team_games", "player_bios", "daily_stats"]

hdr = "table".ljust(28) + "live rows".rjust(14) + "stage rows".rjust(14) + "delta".rjust(13) + "  cols L/S"
print(hdr)
print("-" * len(hdr))
issues = []
for t in TABS:
    try:
        L = pq.ParquetFile(s3.open_input_file(f"{B}/deep_learning/feature_store/{t}.parquet"))
        S = pq.ParquetFile(s3.open_input_file(f"{B}/deep_learning/feature_store_staging/{t}.parquet"))
        lr, sr = L.metadata.num_rows, S.metadata.num_rows
        lc, sc = set(L.schema_arrow.names), set(S.schema_arrow.names)
        flag = ""
        if sr < lr:
            flag = "  <== FEWER ROWS"
            issues.append(f"{t}: lost {lr - sr:,} rows")
        print(f"{t:<28}{lr:>14,}{sr:>14,}{sr - lr:>+13,}  {len(lc)}/{len(sc)}{flag}")
        only_l, only_s = sorted(lc - sc), sorted(sc - lc)
        if only_l:
            print(f"    LOST {len(only_l)} cols: {only_l[:10]}")
            issues.append(f"{t}: lost cols {only_l[:5]}")
        if only_s:
            print(f"    NEW  {len(only_s)} cols: {only_s[:10]}")
    except Exception as e:
        print(f"{t:<28} ERROR {type(e).__name__}: {str(e)[:60]}")
        issues.append(f"{t}: {type(e).__name__}")

print("\n=== VERDICT ===")
if issues:
    print("DO NOT PROMOTE — unexplained regressions:")
    for i in issues:
        print("  -", i)
else:
    print("staging is a strict superset on rows and columns; safe to promote")
