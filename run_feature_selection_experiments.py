import time
from pathlib import Path

import pandas as pd

from feature_engineering import engineer_all_datasets
from anomaly_detection import detect_anomalies_with_scores
from evaluation import evaluate_predictions


ALGORITHMS = [
    "Isolation Forest",
    "LOF",
    "One-Class SVM",
    "DBSCAN"
]


# Top 5 features obtained from the feature-importance analysis
SELECTED_FEATURES = [
    "trend_deviation",
    "diff_1",
    "abs_diff",
    "value",
    "pct_change"
]


def run_feature_selection_experiments():

    print("Loading feature-engineered datasets...")

    feature_data = engineer_all_datasets()

    print(f"Datasets loaded: {len(feature_data)}")

    print("\nSelected features:")
    for feature in SELECTED_FEATURES:
        print(f"  - {feature}")

    results = []

    for dataset_number, (file_path, df) in enumerate(
        feature_data.items(),
        start=1
    ):

        dataset_name = Path(file_path).name
        category = Path(file_path).parent.name

        print(
            f"\n[{dataset_number}/{len(feature_data)}] "
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

                y_pred, scores = detect_anomalies_with_scores(
                    df,
                    algorithm,
                    features=SELECTED_FEATURES
                )

                runtime = (
                    time.perf_counter()
                    - start_time
                )

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

                    "features_used": len(
                        SELECTED_FEATURES
                    ),

                    "feature_set": "top_5",

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
                    f"F1={metrics['f1']:.4f}"
                )

            except Exception as e:

                print(f"FAILED: {e}")

                results.append({

                    "dataset": dataset_name,

                    "category": category,

                    "algorithm": algorithm,

                    "rows": len(df),

                    "features_used":
                        len(SELECTED_FEATURES),

                    "feature_set": "top_5",

                    "actual_anomalies":
                        int(y_true.sum()),

                    "predicted_anomalies": None,

                    "precision": None,

                    "recall": None,

                    "f1": None,

                    "pr_auc": None,

                    "true_positive": None,

                    "false_positive": None,

                    "false_negative": None,

                    "true_negative": None,

                    "runtime_seconds": None
                })

    return pd.DataFrame(results)


def create_summary(results):

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

    print("=" * 70)
    print("NAB TOP-5 FEATURE SELECTION EXPERIMENT")
    print("=" * 70)

    results = run_feature_selection_experiments()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Dataset-level results
    results_file = (
        results_dir
        / "top5_feature_selection_dataset_results.csv"
    )

    results.to_csv(
        results_file,
        index=False
    )

    print(
        "\nDataset-level results saved to:"
    )

    print(results_file)

    # Model summary
    summary = create_summary(results)

    summary_file = (
        results_dir
        / "top5_feature_selection_model_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("TOP-5 FEATURE SELECTION MODEL SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nModel summary saved to:"
    )

    print(summary_file)

    print(
        "\nFeature-selection experiment completed."
    )