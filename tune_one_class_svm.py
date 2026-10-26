import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

from feature_engineering import engineer_all_datasets, FEATURES
from evaluation import evaluate_predictions


# ============================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================

SVM_PARAMS = {
    "nu": [0.005, 0.01, 0.05, 0.1],
    "gamma": ["scale", 0.01, 0.1]
}


# ============================================================
# REPRESENTATIVE DATASETS FOR TUNING
# ============================================================

TUNING_DATASETS = [
    "art_daily_flatmiddle.csv",
    "art_daily_jumpsup.csv",
    "art_increase_spike_density.csv",
    "exchange-2_cpm_results.csv",
    "exchange-4_cpm_results.csv",
    "ec2_cpu_utilization_24ae8d.csv",
    "ec2_cpu_utilization_5f5533.csv",
    "ambient_temperature_system_failure.csv",
    "machine_temperature_system_failure.csv",
    "nyc_taxi.csv",
    "occupancy_t4013.csv",
    "Twitter_volume_AAPL.csv"
]


# ============================================================
# GENERATE PARAMETER COMBINATIONS
# ============================================================

def generate_parameter_combinations():

    combinations = []

    for nu in SVM_PARAMS["nu"]:

        for gamma in SVM_PARAMS["gamma"]:

            combinations.append({
                "nu": nu,
                "gamma": gamma
            })

    return combinations


# ============================================================
# PREPARE DATA ONCE
# ============================================================

def prepare_datasets(all_datasets):

    prepared = {}

    for file_path, df in all_datasets.items():

        dataset_name = Path(file_path).name

        if dataset_name not in TUNING_DATASETS:
            continue

        X = (
            df[FEATURES]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .values
        )

        y_true = df["ground_truth"].values

        prepared[dataset_name] = {
            "X": X,
            "y_true": y_true,
            "category": Path(file_path).parent.name
        }

    return prepared


# ============================================================
# RUN ONE CONFIGURATION
# ============================================================

def run_one_configuration(X, y_true, params):

    model = OneClassSVM(
        kernel="rbf",
        nu=params["nu"],
        gamma=params["gamma"]
    )

    predictions = model.fit_predict(X)

    y_pred = (
        predictions == -1
    ).astype(int)

    # Higher score = more anomalous
    scores = -model.decision_function(X)

    metrics = evaluate_predictions(
        y_true,
        y_pred,
        scores
    )

    return metrics


# ============================================================
# MAIN TUNING
# ============================================================

def run_tuning():

    print("=" * 70)
    print("FAST ONE-CLASS SVM HYPERPARAMETER TUNING")
    print("=" * 70)

    print("\nLoading feature-engineered datasets...")

    all_datasets = engineer_all_datasets()

    print(
        f"Total NAB datasets available: "
        f"{len(all_datasets)}"
    )

    datasets = prepare_datasets(
        all_datasets
    )

    print(
        f"Tuning datasets selected: "
        f"{len(datasets)}"
    )

    print("\nDatasets used for tuning:")

    for name in datasets:
        print(f"  - {name}")

    parameter_combinations = (
        generate_parameter_combinations()
    )

    print(
        f"\nParameter combinations: "
        f"{len(parameter_combinations)}"
    )

    print(
        f"Total model fits: "
        f"{len(parameter_combinations) * len(datasets)}"
    )

    results = []

    # ========================================================
    # PARAMETER SEARCH
    # ========================================================

    for config_number, params in enumerate(
        parameter_combinations,
        start=1
    ):

        print("\n" + "-" * 70)

        print(
            f"Configuration "
            f"{config_number}/"
            f"{len(parameter_combinations)}"
        )

        print(params)

        print("-" * 70)

        for dataset_number, (
            dataset_name,
            data
        ) in enumerate(
            datasets.items(),
            start=1
        ):

            print(
                f"   [{dataset_number}/"
                f"{len(datasets)}] "
                f"{dataset_name}...",
                end=" ",
                flush=True
            )

            start_time = time.perf_counter()

            try:

                metrics = run_one_configuration(
                    data["X"],
                    data["y_true"],
                    params
                )

                runtime = (
                    time.perf_counter()
                    - start_time
                )

                results.append({

                    "dataset":
                        dataset_name,

                    "category":
                        data["category"],

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

                print(
                    f"F1={metrics['f1']:.4f}"
                )

            except Exception as e:

                print(
                    f"FAILED: {e}"
                )

    return pd.DataFrame(results)


# ============================================================
# CREATE PARAMETER SUMMARY
# ============================================================

def create_summary(results):

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


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    results = run_tuning()

    results_dir = Path("results")
    results_dir.mkdir(
        exist_ok=True
    )

    # ========================================================
    # SAVE DETAILED RESULTS
    # ========================================================

    detailed_file = (
        results_dir
        / "one_class_svm_tuning_trials.csv"
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
    # PARAMETER SUMMARY
    # ========================================================

    summary = create_summary(
        results
    )

    summary = summary.sort_values(
        "mean_f1",
        ascending=False
    )

    summary_file = (
        results_dir
        / "one_class_svm_tuning_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("ONE-CLASS SVM PARAMETER SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nParameter summary saved to:"
    )

    print(summary_file)

    # ========================================================
    # BEST CONFIGURATION
    # ========================================================

    best = summary.iloc[0]

    best_config = {
        "algorithm": "One-Class SVM",
        "nu": float(best["nu"]),
        "gamma": best["gamma"],
        "selection_metric": "mean_f1",
        "mean_precision": float(
            best["mean_precision"]
        ),
        "mean_recall": float(
            best["mean_recall"]
        ),
        "mean_f1": float(
            best["mean_f1"]
        ),
        "mean_pr_auc": float(
            best["mean_pr_auc"]
        ),
        "mean_runtime": float(
            best["mean_runtime"]
        ),
        "tuning_datasets": len(
            datasets
        )
    }

    best_file = (
        results_dir
        / "one_class_svm_best_configuration.csv"
    )

    pd.DataFrame(
        [best_config]
    ).to_csv(
        best_file,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("BEST ONE-CLASS SVM CONFIGURATION")
    print("=" * 70)

    for key, value in best_config.items():
        print(
            f"{key}: {value}"
        )

    print(
        "\nBest configuration saved to:"
    )

    print(best_file)

    print(
        "\nOne-Class SVM tuning completed."
    )