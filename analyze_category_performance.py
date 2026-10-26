from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = Path("results")

INPUT_FILE = (
    RESULTS_DIR /
    "final_optimized_dataset_results.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR /
    "category_model_comparison.csv"
)

WINNER_FILE = (
    RESULTS_DIR /
    "category_algorithm_winners.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_results():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    return df


# ============================================================
# CATEGORY-WISE SUMMARY
# ============================================================

def create_category_summary(df):

    summary = (
        df
        .groupby(
            ["category", "algorithm"]
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
# FIND CATEGORY WINNERS
# ============================================================

def find_category_winners(summary):

    winners = []

    for category in summary["category"].unique():

        category_data = summary[
            summary["category"] == category
        ].copy()

        # ----------------------------------------------------
        # Best F1
        # ----------------------------------------------------

        best_f1 = category_data.loc[
            category_data["mean_f1"].idxmax()
        ]

        # ----------------------------------------------------
        # Best PR-AUC
        # Ignore NaN values
        # ----------------------------------------------------

        valid_pr_auc = category_data.dropna(
            subset=["mean_pr_auc"]
        )

        if not valid_pr_auc.empty:

            best_pr_auc = valid_pr_auc.loc[
                valid_pr_auc["mean_pr_auc"].idxmax()
            ]

            pr_auc_algorithm = (
                best_pr_auc["algorithm"]
            )

            pr_auc_value = (
                best_pr_auc["mean_pr_auc"]
            )

        else:

            pr_auc_algorithm = "Not available"
            pr_auc_value = None

        winners.append({

            "category":
                category,

            "best_algorithm_by_f1":
                best_f1["algorithm"],

            "best_mean_f1":
                best_f1["mean_f1"],

            "best_algorithm_by_pr_auc":
                pr_auc_algorithm,

            "best_mean_pr_auc":
                pr_auc_value
        })

    return pd.DataFrame(winners)


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(summary, winners):

    print("\n")
    print("=" * 75)
    print("CATEGORY-WISE MODEL PERFORMANCE")
    print("=" * 75)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 75)
    print("BEST ALGORITHM BY CATEGORY")
    print("=" * 75)

    print(
        winners.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 75)
    print("NAB CATEGORY-WISE ALGORITHM COMPARISON")
    print("=" * 75)

    # --------------------------------------------------------
    # Load final optimized dataset-level results
    # --------------------------------------------------------

    df = load_results()

    print(
        f"\nLoaded dataset-level results: "
        f"{len(df)} rows"
    )

    print(
        f"Categories: "
        f"{df['category'].nunique()}"
    )

    print(
        f"Algorithms: "
        f"{df['algorithm'].nunique()}"
    )

    # --------------------------------------------------------
    # Category summary
    # --------------------------------------------------------

    summary = create_category_summary(df)

    summary.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nCategory-wise results saved to:"
    )

    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # Category winners
    # --------------------------------------------------------

    winners = find_category_winners(
        summary
    )

    winners.to_csv(
        WINNER_FILE,
        index=False
    )

    print(
        "\nCategory winners saved to:"
    )

    print(WINNER_FILE)

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_results(
        summary,
        winners
    )

    print("\n")
    print("=" * 75)
    print("CATEGORY ANALYSIS COMPLETED")
    print("=" * 75)