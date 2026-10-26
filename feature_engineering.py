from collections import deque
import numpy as np
import pandas as pd

FEATURES = [
    "value",
    "diff_1",
    "abs_diff",
    "pct_change",
    "rolling_mean_10",
    "rolling_std_10",
    "rolling_mean_30",
    "rolling_std_30",
    "z_score",
    "trend_deviation",
    "hour_of_day",
    "day_of_week"
]


# =========================================================
# BATCH FEATURE GENERATION (No data dropping)
# =========================================================

def create_features(df):
    """
    Generate feature matrix without dropping rows to preserve time sequence.
    Guarantees strict alignment with StreamingFeatureExtractor outputs.
    """
    df = df.copy()

    # Temporal / Calendar features for context
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
    else:
        ts_col = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour_of_day"] = ts_col.dt.hour.fillna(0).astype(int)
        df["day_of_week"] = ts_col.dt.dayofweek.fillna(0).astype(int)

    # Differences
    df["diff_1"] = df["value"].diff().fillna(0.0)
    df["abs_diff"] = df["diff_1"].abs()

    # Percentage change
    prev_val = df["value"].shift(1)
    df["pct_change"] = np.where(
        prev_val.isna(),
        0.0,
        ((df["value"] - prev_val) / (prev_val.abs() + 1e-8))
    )
    df["pct_change"] = df["pct_change"].fillna(0.0)

    # Rolling window 10 (ddof=1 for standard sample std)
    df["rolling_mean_10"] = df["value"].rolling(window=10, min_periods=1).mean()
    df["rolling_std_10"] = df["value"].rolling(window=10, min_periods=1).std(ddof=1).fillna(0.0)

    # Rolling window 30 (ddof=1 for standard sample std)
    df["rolling_mean_30"] = df["value"].rolling(window=30, min_periods=1).mean()
    df["rolling_std_30"] = df["value"].rolling(window=30, min_periods=1).std(ddof=1).fillna(0.0)

    # Rolling z-score
    df["z_score"] = ((df["value"] - df["rolling_mean_30"]) / (df["rolling_std_30"] + 1e-8)).fillna(0.0)

    # Trend deviation
    df["trend_deviation"] = df["value"] - df["rolling_mean_30"]

    # Handle numeric bounds without dropping indices
    df = df.replace([np.inf, -np.inf], 0.0)
    df = df.fillna(0.0)

    return df


# =========================================================
# REAL-TIME INCREMENTAL FEATURE GENERATOR (O(1) Memory)
# =========================================================

class StreamingFeatureExtractor:
    """
    Calculates features for a single incoming data point using an online buffer.
    Matches pandas sample statistics (ddof=1) exactly.
    """
    def __init__(self, max_window=30):
        self.buffer = deque(maxlen=max_window)
        self.prev_value = None

    def extract_point_features(self, timestamp, value):
        val = float(value)
        self.buffer.append(val)
        buf_arr = np.array(self.buffer, dtype=float)

        # 1. Temporal
        ts = pd.to_datetime(timestamp)
        hour = int(ts.hour) if pd.notnull(ts) else 0
        day = int(ts.dayofweek) if pd.notnull(ts) else 0

        # 2. Differences
        if self.prev_value is None:
            diff = 0.0
            abs_diff = 0.0
            pct = 0.0
        else:
            diff = val - self.prev_value
            abs_diff = abs(diff)
            pct = diff / (abs(self.prev_value) + 1e-8)

        # 3. Rolling metrics via buffer (using ddof=1 to match pandas sample std)
        w10 = buf_arr[-10:]
        rm10 = float(np.mean(w10))
        rstd10 = float(np.std(w10, ddof=1)) if len(w10) > 1 else 0.0

        rm30 = float(np.mean(buf_arr))
        rstd30 = float(np.std(buf_arr, ddof=1)) if len(buf_arr) > 1 else 0.0

        z_score = (val - rm30) / (rstd30 + 1e-8)
        trend_dev = val - rm30

        self.prev_value = val

        return {
            "value": val,
            "diff_1": diff,
            "abs_diff": abs_diff,
            "pct_change": pct,
            "rolling_mean_10": rm10,
            "rolling_std_10": rstd10,
            "rolling_mean_30": rm30,
            "rolling_std_30": rstd30,
            "z_score": z_score,
            "trend_deviation": trend_dev,
            "hour_of_day": hour,
            "day_of_week": day
        }


def engineer_all_datasets():
    """
    Load all NAB datasets, add ground-truth labels,
    generate engineered features, and return all datasets.
    """
    from data_loader import (
        get_dataset_files,
        load_dataset,
        load_labels,
        add_ground_truth
    )

    files = get_dataset_files()
    labels = load_labels()

    engineered_datasets = {}

    for file_path in files:
        df = load_dataset(file_path)
        df = add_ground_truth(df, file_path, labels)
        df = create_features(df)

        # Use relative dataset path as the dataset name
        dataset_name = str(
            file_path.relative_to(
                file_path.parents[1]
            )
        ).replace("\\", "/")

        engineered_datasets[dataset_name] = df

    print(f"TOTAL: {len(engineered_datasets)}")

    return engineered_datasets