from pathlib import Path
import pandas as pd


RESULTS_DIR = Path("results")


def load_summary(filename, stage):
    """
    Load a model summary and label it with the
    corresponding experimental stage.
    """

    file_path = RESULTS_DIR / filename

    df = pd.read_csv(file_path)

    df.insert(0, "stage", stage)

    return df


def main():

    print("=" * 70)
    print("CREATING MASTER EXPERIMENT COMPARISON")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load results from each major experimental stage
    # ---------------------------------------------------------

    baseline = load_summary(
        "model_summary_with_scores.csv",
        "Baseline"
    )

    engineered = load_summary(
        "optimized_engineered_model_summary.csv",
        "Feature Engineering"
    )

    top5 = load_summary(
        "top5_feature_selection_model_summary.csv",
        "Top-5 Feature Selection"
    )

    optimized = load_summary(
        "final_optimized_model_summary.csv",
        "Final Optimized"
    )

    # ---------------------------------------------------------
    # Keep only comparable performance columns
    # ---------------------------------------------------------

    columns = [
        "stage",
        "algorithm",
        "datasets",
        "mean_precision",
        "mean_recall",
        "mean_f1",
        "mean_pr_auc",
        "mean_runtime"
    ]

    baseline = baseline[columns]
    engineered = engineered[columns]
    top5 = top5[columns]
    optimized = optimized[columns]

    # ---------------------------------------------------------
    # Combine all stages
    # ---------------------------------------------------------

    master = pd.concat(
        [
            baseline,
            engineered,
            top5,
            optimized
        ],
        ignore_index=True
    )

    # ---------------------------------------------------------
    # Save master comparison
    # ---------------------------------------------------------

    output_file = (
        RESULTS_DIR
        / "master_experiment_comparison.csv"
    )

    master.to_csv(
        output_file,
        index=False
    )

    # ---------------------------------------------------------
    # Display result
    # ---------------------------------------------------------

    print("\nMASTER EXPERIMENT COMPARISON")
    print("=" * 70)

    print(
        master.to_string(index=False)
    )

    print("\n")
    print("=" * 70)
    print("MASTER COMPARISON SAVED")
    print("=" * 70)

    print(output_file)


if __name__ == "__main__":
    main()