import time
from pathlib import Path

import pandas as pd

from feature_engineering import (
    engineer_all_datasets,
    FEATURES
)

from anomaly_detection import (
    detect_optimized_anomalies,
    OPTIMIZED_PARAMETERS
)

from evaluation import evaluate_predictions


# ============================================================
# Algorithms
# ============================================================

ALGORITHMS = [
    "Isolation Forest",
    "LOF",
    "One-Class SVM",
    "DBSCAN"
]


# ============================================================
# Run final optimized experiment
# ============================================================

def run_final_optimized_experiment():

    print("=" * 70)
    print("NAB FINAL OPTIMIZED ANOMALY DETECTION EXPERIMENT")
    print("=" * 70)

    print("\nLoading feature-engineered datasets...")

    feature_data = engineer_all_datasets()

    print(
        f"Datasets loaded: {len(feature_data)}"
    )

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

        # Ground truth is used ONLY for evaluation.
        y_true = df["ground_truth"].values

        for algorithm in ALGORITHMS:

            print(
                f"   Running optimized {algorithm}...",
                end=" "
            )

            start_time = time.perf_counter()

            try:

                # --------------------------------------------
                # Run algorithm using tuned parameters
                # --------------------------------------------

                y_pred, scores = detect_optimized_anomalies(
                    df,
                    algorithm,
                    features=FEATURES
                )

                runtime = (
                    time.perf_counter()
                    - start_time
                )

                # --------------------------------------------
                # Evaluate predictions
                # --------------------------------------------

                metrics = evaluate_predictions(
                    y_true,
                    y_pred,
                    scores
                )

                # --------------------------------------------
                # Store results
                # --------------------------------------------

                params = OPTIMIZED_PARAMETERS[
                    algorithm
                ]

                results.append({

                    "dataset":
                        dataset_name,

                    "category":
                        category,

                    "algorithm":
                        algorithm,

                    "parameters":
                        str(params),

                    "rows":
                        len(df),

                    "features_used":
                        len(FEATURES),

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

                print(
                    f"FAILED: {e}"
                )

                params = OPTIMIZED_PARAMETERS[
                    algorithm
                ]

                results.append({

                    "dataset":
                        dataset_name,

                    "category":
                        category,

                    "algorithm":
                        algorithm,

                    "parameters":
                        str(params),

                    "rows":
                        len(df),

                    "features_used":
                        len(FEATURES),

                    "actual_anomalies":
                        int(y_true.sum()),

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


# ============================================================
# Create model-level summary
# ============================================================

def create_summary(results):

    summary = (
        results
        .groupby("algorithm")
        .agg(

            datasets=(
                "dataset",
                "count"
            ),

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


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    results = run_final_optimized_experiment()

    # --------------------------------------------------------
    # Results directory
    # --------------------------------------------------------

    results_dir = Path("results")

    results_dir.mkdir(
        exist_ok=True
    )

    # --------------------------------------------------------
    # Dataset-level results
    # --------------------------------------------------------

    dataset_results_file = (
        results_dir
        / "final_optimized_dataset_results.csv"
    )

    results.to_csv(
        dataset_results_file,
        index=False
    )

    print(
        "\nDataset-level results saved to:"
    )

    print(
        dataset_results_file
    )

    # --------------------------------------------------------
    # Model summary
    # --------------------------------------------------------

    summary = create_summary(
        results
    )

    summary_file = (
        results_dir
        / "final_optimized_model_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    # --------------------------------------------------------
    # Print final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL OPTIMIZED MODEL SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nFinal model summary saved to:"
    )

    print(
        summary_file
    )

    print(
        "\nFinal optimized experiment completed."
    )