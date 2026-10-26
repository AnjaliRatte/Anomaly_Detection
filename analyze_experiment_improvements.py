from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = Path("results")

INPUT_FILE = (
    RESULTS_DIR /
    "master_experiment_comparison.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR /
    "experiment_improvement_analysis.csv"
)


# ============================================================
# LOAD MASTER COMPARISON
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Master comparison file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    return df


# ============================================================
# CALCULATE IMPROVEMENTS
# ============================================================

def calculate_improvements(df):

    algorithms = df["algorithm"].unique()

    stages = [
        "Baseline",
        "Feature Engineering",
        "Top-5 Feature Selection",
        "Final Optimized"
    ]

    metrics = [
        "mean_precision",
        "mean_recall",
        "mean_f1",
        "mean_pr_auc",
        "mean_runtime"
    ]

    results = []

    for algorithm in algorithms:

        algorithm_data = df[
            df["algorithm"] == algorithm
        ]

        for i in range(len(stages) - 1):

            previous_stage = stages[i]
            current_stage = stages[i + 1]

            previous = algorithm_data[
                algorithm_data["stage"] == previous_stage
            ]

            current = algorithm_data[
                algorithm_data["stage"] == current_stage
            ]

            if previous.empty or current.empty:
                continue

            previous = previous.iloc[0]
            current = current.iloc[0]

            result = {
                "algorithm": algorithm,
                "comparison": (
                    f"{previous_stage} -> "
                    f"{current_stage}"
                )
            }

            for metric in metrics:

                old_value = previous[metric]
                new_value = current[metric]

                absolute_change = (
                    new_value - old_value
                )

                if old_value != 0:
                    percentage_change = (
                        absolute_change /
                        abs(old_value)
                    ) * 100
                else:
                    percentage_change = None

                result[
                    f"{metric}_absolute_change"
                ] = absolute_change

                result[
                    f"{metric}_percentage_change"
                ] = percentage_change

            results.append(result)

        # ----------------------------------------------------
        # Overall Baseline -> Final Optimized
        # ----------------------------------------------------

        baseline = algorithm_data[
            algorithm_data["stage"] == "Baseline"
        ]

        final = algorithm_data[
            algorithm_data["stage"] == "Final Optimized"
        ]

        if not baseline.empty and not final.empty:

            baseline = baseline.iloc[0]
            final = final.iloc[0]

            result = {
                "algorithm": algorithm,
                "comparison": (
                    "Baseline -> Final Optimized"
                )
            }

            for metric in metrics:

                old_value = baseline[metric]
                new_value = final[metric]

                absolute_change = (
                    new_value - old_value
                )

                if old_value != 0:
                    percentage_change = (
                        absolute_change /
                        abs(old_value)
                    ) * 100
                else:
                    percentage_change = None

                result[
                    f"{metric}_absolute_change"
                ] = absolute_change

                result[
                    f"{metric}_percentage_change"
                ] = percentage_change

            results.append(result)

    return pd.DataFrame(results)


# ============================================================
# DISPLAY IMPORTANT FINDINGS
# ============================================================

def display_key_findings(analysis):

    print("\n")
    print("=" * 70)
    print("KEY EXPERIMENTAL IMPROVEMENTS")
    print("=" * 70)

    overall = analysis[
        analysis["comparison"] ==
        "Baseline -> Final Optimized"
    ]

    for _, row in overall.iterrows():

        print("\n" + "-" * 70)

        print(
            f"Algorithm: {row['algorithm']}"
        )

        print(
            f"F1 change: "
            f"{row['mean_f1_absolute_change']:.6f}"
        )

        print(
            f"F1 percentage change: "
            f"{row['mean_f1_percentage_change']:.2f}%"
        )

        print(
            f"PR-AUC change: "
            f"{row['mean_pr_auc_absolute_change']:.6f}"
        )

        print(
            f"PR-AUC percentage change: "
            f"{row['mean_pr_auc_percentage_change']:.2f}%"
        )

        print(
            f"Precision change: "
            f"{row['mean_precision_absolute_change']:.6f}"
        )

        print(
            f"Recall change: "
            f"{row['mean_recall_absolute_change']:.6f}"
        )

        print(
            f"Runtime change: "
            f"{row['mean_runtime_absolute_change']:.6f} seconds"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("EXPERIMENT IMPROVEMENT ANALYSIS")
    print("=" * 70)

    df = load_data()

    print(
        f"\nLoaded master comparison: "
        f"{len(df)} rows"
    )

    analysis = calculate_improvements(df)

    # --------------------------------------------------------
    # Save analysis
    # --------------------------------------------------------

    analysis.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nImprovement analysis saved to:"
    )

    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # Display full analysis
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("EXPERIMENT IMPROVEMENT ANALYSIS")
    print("=" * 70)

    print(
        analysis.to_string(index=False)
    )

    # --------------------------------------------------------
    # Display key findings
    # --------------------------------------------------------

    display_key_findings(
        analysis
    )

    print("\n")
    print("=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)