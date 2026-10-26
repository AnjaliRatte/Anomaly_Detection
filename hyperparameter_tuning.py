import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from feature_engineering import engineer_all_datasets, FEATURES
from evaluation import evaluate_predictions


# ============================================================
# FIXED SEARCH SPACE
# ============================================================

ISOLATION_FOREST_PARAMS = {
    "n_estimators": [100, 200],
    "contamination": [0.005, 0.01, 0.02],
    "max_samples": ["auto", 0.8]
}


def generate_parameter_combinations():
    """
    Generate all Isolation Forest parameter combinations.
    """

    combinations = []

    for n_estimators in ISOLATION_FOREST_PARAMS["n_estimators"]:

        for contamination in ISOLATION_FOREST_PARAMS["contamination"]:

            for max_samples in ISOLATION_FOREST_PARAMS["max_samples"]:

                combinations.append({
                    "n_estimators": n_estimators,
                    "contamination": contamination,
                    "max_samples": max_samples
                })

    return combinations


def run_isolation_forest(df, params):
    """
    Run Isolation Forest using a specific parameter configuration.
    """

    X = df[FEATURES].replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    model = IsolationForest(
        n_estimators=params["n_estimators"],
        contamination=params["contamination"],
        max_samples=params["max_samples"],
        random_state=42,
        n_jobs=-1
    )

    predictions = model.fit_predict(X)

    # Isolation Forest:
    # -1 = anomaly
    #  1 = normal
    y_pred = (predictions == -1).astype(int)

    # Higher score = more anomalous
    scores = -model.decision_function(X)

    return y_pred, scores


def run_tuning():

    print("=" * 70)
    print("ISOLATION FOREST AUTOMATED HYPERPARAMETER TUNING")
    print("=" * 70)

    print("\nLoading feature-engineered datasets...")

    datasets = engineer_all_datasets()

    print(f"Datasets loaded: {len(datasets)}")

    parameter_combinations = (
        generate_parameter_combinations()
    )

    print(
        f"Parameter combinations: "
        f"{len(parameter_combinations)}"
    )

    results = []

    total_experiments = (
        len(datasets)
        * len(parameter_combinations)
    )

    experiment_number = 0

    for params in parameter_combinations:

        print("\n" + "-" * 70)

        print(
            "Testing parameters:"
        )

        print(params)

        print("-" * 70)

        for dataset_number, (file_path, df) in enumerate(
            datasets.items(),
            start=1
        ):

            experiment_number += 1

            dataset_name = Path(file_path).name
            category = Path(file_path).parent.name

            y_true = df["ground_truth"].values

            start_time = time.perf_counter()

            try:

                y_pred, scores = run_isolation_forest(
                    df,
                    params
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

                    "algorithm":
                        "Isolation Forest",

                    "dataset":
                        dataset_name,

                    "category":
                        category,

                    "n_estimators":
                        params["n_estimators"],

                    "contamination":
                        params["contamination"],

                    "max_samples":
                        params["max_samples"],

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

            except Exception as e:

                print(
                    f"FAILED: {dataset_name} | {e}"
                )

    results = pd.DataFrame(results)

    return results


def create_tuning_summary(results):

    summary = (
        results
        .groupby(
            [
                "n_estimators",
                "contamination",
                "max_samples"
            ]
        )
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


if __name__ == "__main__":

    results = run_tuning()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------------
    # Save every individual experiment
    # ---------------------------------------------------------

    detailed_file = (
        results_dir
        / "isolation_forest_hyperparameter_trials.csv"
    )

    results.to_csv(
        detailed_file,
        index=False
    )

    print(
        "\nDetailed tuning results saved to:"
    )

    print(detailed_file)

    # ---------------------------------------------------------
    # Create parameter-level summary
    # ---------------------------------------------------------

    summary = create_tuning_summary(
        results
    )

    summary_file = (
        results_dir
        / "isolation_forest_hyperparameter_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print(
        "\nParameter-level summary:"
    )

    print(
        summary
        .sort_values(
            "mean_f1",
            ascending=False
        )
        .to_string(index=False)
    )

    print(
        "\nParameter-level summary saved to:"
    )

    print(summary_file)

    # ---------------------------------------------------------
    # Best configuration
    # ---------------------------------------------------------

    best = (
        summary
        .sort_values(
            "mean_f1",
            ascending=False
        )
        .iloc[0]
    )

    print("\n" + "=" * 70)
    print("BEST ISOLATION FOREST CONFIGURATION")
    print("=" * 70)

    print(
        best.to_string()
    )