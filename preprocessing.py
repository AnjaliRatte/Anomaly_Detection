from data_loader import (
    get_dataset_files,
    load_dataset,
    load_labels,
    add_ground_truth
)

import pandas as pd


def preprocess_dataset(file_path, labels):
    """
    Load and preprocess one NAB dataset.
    """

    # Load raw data
    df = load_dataset(file_path)

    # Add NAB ground-truth labels
    df = add_ground_truth(df, file_path, labels)

    # Remove duplicate timestamps
    df = df.drop_duplicates(subset="timestamp")

    # Remove rows with missing values
    df = df.dropna(subset=["timestamp", "value"])

    # Ensure chronological order
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def preprocess_all_datasets():
    """
    Preprocess every CSV file in the NAB dataset.
    """

    files = get_dataset_files()
    labels = load_labels()

    processed_data = {}

    for file_path in files:

        df = preprocess_dataset(file_path, labels)

        processed_data[str(file_path)] = df

    return processed_data

def create_dataset_report(processed_data):
    """Create a summary report for all processed datasets."""

    report = []

    for file_path, df in processed_data.items():

        report.append({
            "dataset": file_path,
            "rows": len(df),
            "columns": len(df.columns),
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_timestamps": int(
                df["timestamp"].duplicated().sum()
            ),
            "anomaly_count": int(
                df["ground_truth"].sum()
            ),
            "start_time": df["timestamp"].min(),
            "end_time": df["timestamp"].max()
        })

    return pd.DataFrame(report)


if __name__ == "__main__":

    processed_data = preprocess_all_datasets()

    print(f"Total datasets processed: {len(processed_data)}")

    report = create_dataset_report(processed_data)

    print("\nDataset Report:")
    print(report.to_string(index=False))

    print("\nTotal missing values:")
    print(report["missing_values"].sum())

    print("\nTotal duplicate timestamps:")
    print(report["duplicate_timestamps"].sum())

    print("\nTotal labeled anomalies:")
    print(report["anomaly_count"].sum())