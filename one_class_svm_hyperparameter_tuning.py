import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.svm import OneClassSVM

from feature_engineering import engineer_all_datasets, FEATURES
from evaluation import evaluate_predictions


# ============================================================
# FIXED SEARCH SPACE
# ============================================================

SVM_PARAMS = {
    "nu": [0.005, 0.01, 0.02, 0.05],
    "gamma": ["scale", 0.01, 0.1, 1.0]
}


def generate_parameter_combinations():

    combinations = []

    for nu in SVM_PARAMS["nu"]:

        for gamma in SVM_PARAMS["gamma"]:

            combinations.append({
                "nu": nu,
                "gamma": gamma
            })

    return combinations


def run_one_class_svm(df, params):

    X = (
        df[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    model = OneClassSVM(
        kernel="rbf",
        nu=params["nu"],
        gamma=params["gamma"]
    )

    predictions = model.fit_predict(X)

    # -1 = anomaly
    y_pred = (
        predictions == -1
    ).astype(int)

    # Higher value = more anomalous
    scores = -model.decision_function(X)

    return y_pred, scores


def run_tuning():

    print("=" * 70)
    print("ONE-CLASS SVM AUTOMATED HYPERPARAMETER TUNING")
    print("=" * 70)

    print("\nLoading feature-engineered datasets...")

    datasets = engineer_all_datasets()

    print(
        f"Datasets loaded: {len(datasets)}"
    )

    parameter_combinations = (
        generate_parameter_combinations()
    )

    print(
        f"Parameter combinations: "
        f"{len(parameter_combinations)}"
    )

    results = []

    for combination_number, params in enumerate(
        parameter_combinations,
        start=1
    ):

        print("\n" + "-" * 70)

        print(
            f"Configuration "
            f"{combination_number}/"
            f"{len(parameter_combinations)}"
        )

        print(params)

        print("-" * 70)

        for dataset_number, (file_path, df) in enumerate(
            datasets.items(),
            start=1
        ):

            dataset_name = Path(file_path).name
            category = Path(file_path).parent.name

            y_true = df["ground_truth"].values

            start_time = time.perf_counter()

            try:

                y_pred, scores = run_one_class_svm(
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
                        "One-Class SVM",

                    "dataset":
                        dataset_name,

                    "category":
                        category,

                    "nu":
                        params["nu"],

                    "gamma":
                        params["gamma"],

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

    return pd.DataFrame(results)


def create_tuning_summary(results):

    summary = (
        results
        .groupby(
            [
                "nu",
                "gamma"
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

    # ========================================================
    # Save every individual trial
    # ========================================================

    detailed_file = (
        results_dir
        / "one_class_svm_hyperparameter_trials.csv"
    )

    results.to_csv(
        detailed_file,
        index=False
    )

    print(
        "\nDetailed tuning results saved to:"
    )

    print(detailed_file)

    # ========================================================
    # Parameter-level summary
    # ========================================================

    summary = create_tuning_summary(
        results
    )

    summary_file = (
        results_dir
        / "one_class_svm_hyperparameter_summary.csv"
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

    # ========================================================
    # Best configuration
    # ========================================================

    best = (
        summary
        .sort_values(
            "mean_f1",
            ascending=False
        )
        .iloc[0]
    )

    print("\n" + "=" * 70)
    print("BEST ONE-CLASS SVM CONFIGURATION")
    print("=" * 70)

    print(
        best.to_string()
    )