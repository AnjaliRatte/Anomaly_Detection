from pathlib import Path
import pandas as pd


RESULTS_DIR = Path("results")


def main():

    print("=" * 75)
    print("FINAL SIX-ALGORITHM COMPARISON")
    print("=" * 75)

    # ---------------------------------------------------------
    # Load final optimized classical models
    # ---------------------------------------------------------

    classical = pd.read_csv(
        RESULTS_DIR / "final_optimized_model_summary.csv"
    )

    # ---------------------------------------------------------
    # Load Half-Space Trees
    # ---------------------------------------------------------

    half_space = pd.read_csv(
        RESULTS_DIR / "final_half_space_trees_model_summary.csv"
    )

    # ---------------------------------------------------------
    # Load LSTM Autoencoder
    # ---------------------------------------------------------

    lstm = pd.read_csv(
        RESULTS_DIR / "final_lstm_model_summary.csv"
    )

    # ---------------------------------------------------------
    # Correct dataset count
    # ---------------------------------------------------------

    classical["datasets"] = 58
    half_space["datasets"] = 58
    lstm["datasets"] = 58

    # ---------------------------------------------------------
    # Combine all six algorithms
    # ---------------------------------------------------------

    final = pd.concat(
        [
            classical,
            half_space,
            lstm
        ],
        ignore_index=True
    )

    # ---------------------------------------------------------
    # Keep final comparison columns
    # ---------------------------------------------------------

    final = final[
        [
            "algorithm",
            "datasets",
            "mean_precision",
            "mean_recall",
            "mean_f1",
            "mean_pr_auc",
            "mean_runtime"
        ]
    ]

    # ---------------------------------------------------------
    # Rank by F1
    # ---------------------------------------------------------

    final = final.sort_values(
        "mean_f1",
        ascending=False
    ).reset_index(drop=True)

    final.insert(
        0,
        "rank",
        range(1, len(final) + 1)
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output_file = (
        RESULTS_DIR /
        "final_six_algorithm_comparison.csv"
    )

    final.to_csv(
        output_file,
        index=False
    )

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------

    print("\nFINAL SIX-ALGORITHM RESULTS")
    print("-" * 75)

    print(
        final.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print("\nSaved to:")
    print(output_file)

    # ---------------------------------------------------------
    # Best models
    # ---------------------------------------------------------

    best_f1 = final.loc[
        final["mean_f1"].idxmax()
    ]

    best_pr_auc = final.loc[
        final["mean_pr_auc"].idxmax()
    ]

    best_recall = final.loc[
        final["mean_recall"].idxmax()
    ]

    print("\nBEST RESULTS")
    print("-" * 75)

    print(
        f"Best F1      : "
        f"{best_f1['algorithm']} "
        f"({best_f1['mean_f1']:.6f})"
    )

    print(
        f"Best PR-AUC  : "
        f"{best_pr_auc['algorithm']} "
        f"({best_pr_auc['mean_pr_auc']:.6f})"
    )

    print(
        f"Best Recall  : "
        f"{best_recall['algorithm']} "
        f"({best_recall['mean_recall']:.6f})"
    )


if __name__ == "__main__":
    main()