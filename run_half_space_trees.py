import time
from pathlib import Path

import numpy as np
import pandas as pd

from river import anomaly

from feature_engineering import (
    engineer_all_datasets,
    FEATURES
)

from evaluation import evaluate_predictions


# ============================================================
# HALF-SPACE TREES CONFIGURATION
# ============================================================

N_TREES = 10
HEIGHT = 8
WINDOW_SIZE = 250
SEED = 42

ALGORITHM_NAME = "Half-Space Trees"


# ============================================================
# RUN HALF-SPACE TREES ON ONE DATASET
# ============================================================

def run_half_space_trees(df):

    X = (
        df[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    model = anomaly.HalfSpaceTrees(
        n_trees=N_TREES,
        height=HEIGHT,
        window_size=WINDOW_SIZE,
        seed=SEED
    )

    predictions = []
    scores = []

    # --------------------------------------------------------
    # Half-Space Trees is an online algorithm.
    # Each observation is processed sequentially.
    # --------------------------------------------------------

    for _, row in X.iterrows():

        x = row.to_dict()

        # Get anomaly score before learning
        score = model.score_one(x)

        scores.append(score)

        # Learn from current observation
        model.learn_one(x)

    scores = np.asarray(scores)

    # --------------------------------------------------------
    # Convert anomaly scores into binary predictions
    #
    # We use the contamination level from our previous
    # experiments to create approximately 1% anomalies.
    # --------------------------------------------------------

    threshold = np.percentile(
        scores,
        99
    )

    y_pred = (
        scores >= threshold
    ).astype(int)

    return y_pred, scores


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiment():

    print("=" * 70)
    print("NAB HALF-SPACE TREES EXPERIMENT")
    print("=" * 70)

    print("\nLoading feature-engineered datasets...")

    datasets = engineer_all_datasets()

    print(
        f"Datasets loaded: {len(datasets)}"
    )

    results = []

    for dataset_number, (file_path, df) in enumerate(
        datasets.items(),
        start=1
    ):

        dataset_name = Path(file_path).name
        category = Path(file_path).parent.name

        print(
            f"\n[{dataset_number}/{len(datasets)}] "
            f"{dataset_name}"
        )

        print(
            "   Running Half-Space Trees...",
            end=" "
        )

        # Ground truth ONLY used for evaluation
        y_true = df["ground_truth"].values

        start_time = time.perf_counter()

        try:

            y_pred, scores = run_half_space_trees(df)

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

                "dataset":
                    dataset_name,

                "category":
                    category,

                "algorithm":
                    ALGORITHM_NAME,

                "rows":
                    len(df),

                "features_used":
                    len(FEATURES),

                "n_trees":
                    N_TREES,

                "height":
                    HEIGHT,

                "window_size":
                    WINDOW_SIZE,

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

            results.append({

                "dataset":
                    dataset_name,

                "category":
                    category,

                "algorithm":
                    ALGORITHM_NAME,

                "rows":
                    len(df),

                "features_used":
                    len(FEATURES),

                "n_trees":
                    N_TREES,

                "height":
                    HEIGHT,

                "window_size":
                    WINDOW_SIZE,

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
# CREATE MODEL SUMMARY
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
# MAIN
# ============================================================

if __name__ == "__main__":

    results = run_experiment()

    results_dir = Path("results")
    results_dir.mkdir(
        exist_ok=True
    )

    # --------------------------------------------------------
    # Dataset-level results
    # --------------------------------------------------------

    results_file = (
        results_dir
        / "half_space_trees_dataset_results.csv"
    )

    results.to_csv(
        results_file,
        index=False
    )

    print(
        "\nDataset-level results saved to:"
    )

    print(results_file)

    # --------------------------------------------------------
    # Model summary
    # --------------------------------------------------------

    summary = create_summary(
        results
    )

    summary_file = (
        results_dir
        / "half_space_trees_model_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("HALF-SPACE TREES MODEL SUMMARY")
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
        "\nHalf-Space Trees experiment completed."
    )