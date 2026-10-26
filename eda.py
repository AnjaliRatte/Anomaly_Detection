from preprocessing import (
    preprocess_all_datasets,
    create_dataset_report
)

import pandas as pd


def create_eda_report(processed_data):
    """
    Create a detailed EDA report for all NAB datasets.
    """

    report = []

    for file_path, df in processed_data.items():

        dataset_name = file_path.split("\\")[-1]
        category = file_path.split("\\")[-2]

        values = df["value"]

        anomaly_count = int(df["ground_truth"].sum())
        total_rows = len(df)

        anomaly_percentage = (
            (anomaly_count / total_rows) * 100
            if total_rows > 0 else 0
        )

        report.append({
            "dataset": dataset_name,
            "category": category,
            "rows": total_rows,
            "mean": values.mean(),
            "median": values.median(),
            "std": values.std(),
            "min": values.min(),
            "max": values.max(),
            "anomaly_count": anomaly_count,
            "anomaly_percentage": anomaly_percentage,
            "start_time": df["timestamp"].min(),
            "end_time": df["timestamp"].max()
        })

    return pd.DataFrame(report)


def create_category_summary(eda_report):
    """
    Create category-level EDA summary.
    """

    category_summary = (
        eda_report
        .groupby("category")
        .agg(
            datasets=("dataset", "count"),
            total_rows=("rows", "sum"),
            total_anomalies=("anomaly_count", "sum"),
            average_mean=("mean", "mean"),
            average_std=("std", "mean")
        )
        .reset_index()
    )

    category_summary["anomaly_percentage"] = (
        category_summary["total_anomalies"]
        / category_summary["total_rows"]
        * 100
    )

    return category_summary


if __name__ == "__main__":

    print("Loading and preprocessing NAB datasets...")

    processed_data = preprocess_all_datasets()

    print(f"Total datasets: {len(processed_data)}")

    # Basic preprocessing report
    dataset_report = create_dataset_report(processed_data)

    print("\n===== BASIC DATASET SUMMARY =====")

    print(f"Total datasets: {len(dataset_report)}")
    print(f"Total observations: {dataset_report['rows'].sum():,}")
    print(
        f"Total labeled anomalies: "
        f"{dataset_report['anomaly_count'].sum():,}"
    )

    total_rows = dataset_report["rows"].sum()
    total_anomalies = dataset_report["anomaly_count"].sum()

    overall_anomaly_percentage = (
        total_anomalies / total_rows * 100
        if total_rows > 0 else 0
    )

    print(
        f"Overall anomaly percentage: "
        f"{overall_anomaly_percentage:.4f}%"
    )

    # Detailed EDA
    eda_report = create_eda_report(processed_data)

    print("\n===== DATASET-WISE EDA =====")

    print(
        eda_report[
            [
                "dataset",
                "category",
                "rows",
                "mean",
                "median",
                "std",
                "min",
                "max",
                "anomaly_count",
                "anomaly_percentage"
            ]
        ].to_string(index=False)
    )

    # Category summary
    category_summary = create_category_summary(eda_report)

    print("\n===== CATEGORY-WISE SUMMARY =====")

    print(
        category_summary.to_string(index=False)
    )