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
# HYPERPARAMETER SEARCH SPACE
# ============================================================

HST_PARAMS = {
    "n_trees": [10, 25, 50],
    "height": [8, 10, 12],
    "window_size": [100, 250, 500]
}


# ============================================================
# GENERATE PARAMETER COMBINATIONS
# ============================================================

def generate_parameter_combinations():

    combinations = []

    for n_trees in HST_PARAMS["n_trees"]:

        for height in HST_PARAMS["height"]:

            for window_size in HST_PARAMS["window_size"]:

                combinations.append({
                    "n_trees": n_trees,
                    "height": height,
                    "window_size": window_size
                })

    return combinations


# ============================================================
# RUN HALF-SPACE TREES
# ============================================================

def run_half_space_trees(df, params):

    X = (
        df[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    model = anomaly.HalfSpaceTrees(
        n_trees=params["n_trees"],
        height=params["height"],
        window_size=params["window_size"],
        seed=42
    )

    scores = []

    for _, row in X.iterrows():

        x = row.to_dict()

        score = model.score_one(x)

        scores.append(score)

        model.learn_one(x)

    scores = np.asarray(scores)

    # Same 1% prediction threshold used in the
    # baseline Half-Space Trees experiment.
    threshold = np.percentile(
        scores,
        99
    )

    y_pred = (
        scores >= threshold
    ).astype(int)

    return y_pred, scores


# ============================================================
# RUN TUNING
# ============================================================

def run_tuning():

    print("=" * 70)
    print("HALF-SPACE TREES AUTOMATED HYPERPARAMETER TUNING")
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

    # --------------------------------------------------------
    # Use 12 datasets for tuning, matching our previous
    # hyperparameter-tuning methodology.
    # --------------------------------------------------------

    tuning_datasets = list(
        datasets.items()
    )[:12]

    print(
        f"Tuning datasets: {len(tuning_datasets)}"
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
            tuning_datasets,
            start=1
        ):

            dataset_name = Path(file_path).name
            category = Path(file_path).parent.name

            y_true = df["ground_truth"].values

            start_time = time.perf_counter()

            try:

                y_pred, scores = run_half_space_trees(
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
                        "Half-Space Trees",

                    "dataset":
                        dataset_name,

                    "category":
                        category,

                    "n_trees":
                        params["n_trees"],

                    "height":
                        params["height"],

                    "window_size":
                        params["window_size"],

                    "precision":
                        metrics["precision"],

                    "recall":
                        metrics["recall"],

                    "f1":
                        metrics["f1"],

                    "pr_auc":
                        metrics["pr_auc"],

                    "runtime_seconds":
                        runtime
                })

            except Exception as e:

                print(
                    f"FAILED: {dataset_name} | {e}"
                )

    return pd.DataFrame(results)


# ============================================================
# CREATE SUMMARY
# ============================================================

def create_tuning_summary(results):

    summary = (
        results
        .groupby(
            [
                "n_trees",
                "height",
                "window_size"
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

    # --------------------------------------------------------
    # Save individual trials
    # --------------------------------------------------------

    trials_file = (
        results_dir
        / "half_space_trees_hyperparameter_trials.csv"
    )

    results.to_csv(
        trials_file,
        index=False
    )

    print(
        "\nDetailed tuning results saved to:"
    )

    print(trials_file)

    # --------------------------------------------------------
    # Parameter summary
    # --------------------------------------------------------

    summary = create_tuning_summary(
        results
    )

    summary_file = (
        results_dir
        / "half_space_trees_hyperparameter_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("HALF-SPACE TREES PARAMETER SUMMARY")
    print("=" * 70)

    print(
        summary
        .sort_values(
            "mean_f1",
            ascending=False
        )
        .to_string(index=False)
    )

    print(
        "\nParameter summary saved to:"
    )

    print(summary_file)

    # --------------------------------------------------------
    # Best configuration
    # --------------------------------------------------------

    best = (
        summary
        .sort_values(
            "mean_f1",
            ascending=False
        )
        .iloc[0]
    )

    print("\n" + "=" * 70)
    print("BEST HALF-SPACE TREES CONFIGURATION")
    print("=" * 70)

    print(
        best.to_string()
    )

    # --------------------------------------------------------
    # Save best configuration
    # --------------------------------------------------------

    best_file = (
        results_dir
        / "half_space_trees_best_configuration.csv"
    )

    best.to_frame().T.to_csv(
        best_file,
        index=False
    )

    print(
        "\nBest configuration saved to:"
    )

    print(best_file)

    print(
        "\nHalf-Space Trees tuning completed."
    )