from pathlib import Path
import json
import pandas as pd


# Location of the existing NAB dataset
NAB_PATH = Path(__file__).resolve().parent.parent / "NAB-master"
LABELS_PATH = NAB_PATH / "labels" / "combined_labels.json"


def get_dataset_files():
    """Find all CSV files inside the NAB dataset."""

    csv_files = list((NAB_PATH / "data").rglob("*.csv"))

    return csv_files


def load_dataset(file_path):
    """Load a NAB CSV file into a pandas DataFrame."""

    df = pd.read_csv(file_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df

def load_labels():
    """Load NAB ground-truth anomaly labels."""

    with open(LABELS_PATH, "r") as file:
        labels = json.load(file)

    return labels

def add_ground_truth(df, file_path, labels):
    """Add NAB ground-truth anomaly labels to a dataset."""

    relative_path = file_path.relative_to(NAB_PATH / "data")

    relative_path = str(relative_path).replace("\\", "/")

    anomaly_timestamps = labels.get(relative_path, [])

    # Convert NAB label timestamps to datetime
    anomaly_timestamps = pd.to_datetime(anomaly_timestamps)

    # Match timestamps
    df["ground_truth"] = (
        df["timestamp"].isin(anomaly_timestamps).astype(int)
    )

    return df

if __name__ == "__main__":

    files = get_dataset_files()

    labels = load_labels()

    # Select a dataset that has known anomalies
    file_path = (
            NAB_PATH
            / "data"
            / "artificialWithAnomaly"
            / "art_daily_jumpsup.csv"
    )

    print(f"Dataset: {file_path}")

    df = load_dataset(file_path)

    df = add_ground_truth(df, file_path, labels)

    print("\nData types:")
    print(df.dtypes)

    print("\nTime range:")
    print(df["timestamp"].min(), "→", df["timestamp"].max())

    print("\nMissing values:")
    print(df.isnull().sum())

