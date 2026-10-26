import time
from pathlib import Path

import pandas as pd

from preprocessing import preprocess_all_datasets
from anomaly_detection import detect_anomalies_with_scores
from evaluation import evaluate_predictions


ALGORITHMS = [
    "Isolation Forest",
    "LOF",
    "One-Class SVM",
    "DBSCAN"
]


def run_experiments():
    """
    Run all anomaly detection algorithms
    across all NAB datasets.

    Each algorithm returns:
    1. Binary anomaly predictions
    2. Anomaly scores for PR-AUC
    """

    print("Loading and preprocessing datasets...")

    processed_data = preprocess_all_datasets()

    print(f"Datasets loaded: {len(processed_data)}")

    results = []

    for dataset_number, (file_path, df) in enumerate(
        processed_data.items(),
        start=1
    ):

        dataset_name = Path(file_path).name
        category = Path(file_path).parent.name

        print(
            f"\n[{dataset_number}/{len(processed_data)}] "
            f"{dataset_name}"
        )

        y_true = df["ground_truth"].values

        for algorithm in ALGORITHMS:

            print(
                f"   Running {algorithm}...",
                end=" "
            )

            start_time = time.perf_counter()

            try:

                # Get predictions AND real anomaly scores
                y_pred, scores = detect_anomalies_with_scores(
                    df,
                    algorithm
                )

                runtime = (
                    time.perf_counter()
                    - start_time
                )

                # Evaluate using real anomaly scores
                metrics = evaluate_predictions(
                    y_true,
                    y_pred,
                    scores
                )

                results.append({
                    "dataset": dataset_name,
                    "category": category,
                    "algorithm": algorithm,
                    "rows": len(df),

                    "actual_anomalies":
                        metrics["actual_anomalies"],

                    "predicted_anomalies":
                        metrics["predicted_anomalies"],

                    "precision":
                        metrics["precision"],

                    "recall":
                        metrics["recall"],

                    "f1":
                        metrics["f1"],

                    "pr_auc":
                        metrics["pr_auc"],

                    "true_positive":
                        metrics["true_positive"],

                    "false_positive":
                        metrics["false_positive"],

                    "false_negative":
                        metrics["false_negative"],

                    "true_negative":
                        metrics["true_negative"],

                    "runtime_seconds":
                        runtime
                })

                print(
                    f"F1={metrics['f1']:.4f} "
                    f"| PR-AUC={metrics['pr_auc']:.4f}"
                    if pd.notna(metrics["pr_auc"])
                    else
                    f"F1={metrics['f1']:.4f} "
                    f"| PR-AUC=NaN"
                )

            except Exception as e:

                print(f"FAILED: {e}")

                results.append({
                    "dataset": dataset_name,
                    "category": category,
                    "algorithm": algorithm,
                    "rows": len(df),

                    "actual_anomalies":
                        y_true.sum(),

                    "predicted_anomalies":
                        None,

                    "precision":
                        None,

                    "recall":
                        None,

                    "f1":
                        None,

                    "pr_auc":
                        None,

                    "true_positive":
                        None,

                    "false_positive":
                        None,

                    "false_negative":
                        None,

                    "true_negative":
                        None,

                    "runtime_seconds":
                        None
                })

    return pd.DataFrame(results)


def create_summary(results):
    """
    Create algorithm-level summary.
    """

    summary = (
        results
        .groupby("algorithm")
        .agg(
            datasets=("dataset", "count"),

            mean_precision=(
                "precision",
                "mean"
            ),

            mean_recall=(
                "recall",
                "mean"
            ),

            mean_f1=(
                "f1",
                "mean"
            ),

            mean_pr_auc=(
                "pr_auc",
                "mean"
            ),

            mean_runtime=(
                "runtime_seconds",
                "mean"
            )
        )
        .reset_index()
    )

    return summary


if __name__ == "__main__":

    print("=" * 60)
    print("NAB ANOMALY DETECTION EXPERIMENT")
    print("=" * 60)

    results = run_experiments()

    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Save dataset-level results
    results_file = (
        results_dir /
        "dataset_results_with_scores.csv"
    )

    results.to_csv(
        results_file,
        index=False
    )

    print("\nDataset-level results saved to:")
    print(results_file)

    # Create algorithm summary
    summary = create_summary(results)

    summary_file = (
        results_dir /
        "model_summary_with_scores.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print("\nModel summary:")
    print(
        summary.to_string(index=False)
    )

    print("\nModel summary saved to:")
    print(summary_file)

    print("\nExperiment completed.")