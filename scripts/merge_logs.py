# merge_logs.py (Final Clean Version - No Fake Zeros)
import pandas as pd
from pathlib import Path

# ---------------- CONFIGURATION ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
OUT_DIR = BASE_DIR / "data" / "agg"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"🚦 Merging logs from → {LOG_DIR}\n")


def merge_logs():
    all_dfs = []
    vehicle_columns = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
    total_col = "total_vehicles"

    # -------- Load CSV logs --------
    for csv_file in LOG_DIR.glob("*.csv"):
        if any(x in csv_file.name.lower() for x in ["merged", "per_min", "per_hr"]):
            print(f"⏭️ Skipping processed file: {csv_file.name}")
            continue

        df = pd.read_csv(csv_file)

        if "timestamp" not in df.columns:
            print(f"⚠️ Skipping (no timestamp): {csv_file.name}")
            continue

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.dropna(subset=["timestamp"], inplace=True)

        # Convert numeric values safely
        for col in vehicle_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Compute total vehicles accurately
        df[total_col] = df[vehicle_columns].sum(axis=1)

        all_dfs.append(df)
        print(f"📥 Loaded {csv_file.name}: {len(df)} rows")

    if not all_dfs:
        print("❌ No valid logs found.")
        return

    # -------- Combine --------
    merged = pd.concat(all_dfs, ignore_index=True)
    merged.sort_values("timestamp", inplace=True)
    merged.drop_duplicates(subset="timestamp", keep="last", inplace=True)

    # Recalculate totals cleanly
    merged[total_col] = merged[vehicle_columns].sum(axis=1)

    # Save merged raw file
    merged_out = OUT_DIR / "traffic_merged.csv"
    merged.to_csv(merged_out, index=False)
    print(f"💾 Saved → {merged_out} ({len(merged)} rows)")

    # -------- Per-minute aggregation --------
    merged.set_index("timestamp", inplace=True)
    per_min = merged.resample("1min").sum()

    # 🔥 Remove rows that are fake (all 0 — recording gap)
    per_min = per_min[(per_min[vehicle_columns].sum(axis=1) > 0)]

    # Ensure integers not floats
    for col in vehicle_columns + [total_col]:
        per_min[col] = per_min[col].round().astype(int)

    per_min.reset_index(inplace=True)
    per_min_out = OUT_DIR / "traffic_per_min.csv"
    per_min.to_csv(per_min_out, index=False)
    print(f"⏱️ Per-minute saved → {per_min_out} ({len(per_min)} rows)")

    # -------- Per-hour aggregation --------
    per_hr = merged.resample("1H").sum()

    # Remove empty recording gaps
    per_hr = per_hr[(per_hr[vehicle_columns].sum(axis=1) > 0)]

    # Convert to integer
    for col in vehicle_columns + [total_col]:
        per_hr[col] = per_hr[col].round().astype(int)

    per_hr.reset_index(inplace=True)
    per_hr_out = OUT_DIR / "traffic_per_hr.csv"
    per_hr.to_csv(per_hr_out, index=False)

    print(f"🕒 Per-hour saved → {per_hr_out} ({len(per_hr)} rows)")

    print("\n🎉 Merge & Cleaning Completed Successfully!")


if __name__ == "__main__":
    merge_logs()
