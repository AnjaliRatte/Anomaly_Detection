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
    "dataset_algorithm_comparison.csv"
)

WINNER_FILE = (
    RESULTS_DIR /
    "dataset_algorithm_winners.csv"
)

SUMMARY_FILE = (
    RESULTS_DIR /
    "dataset_performance_summary.csv"
)


# ============================================================
# LOAD RESULTS
# ============================================================

print("=" * 70)
print("NAB DATASET-WISE ALGORITHM PERFORMANCE ANALYSIS")
print("=" * 70)

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Required file not found: {INPUT_FILE}"
    )

results = pd.read_csv(INPUT_FILE)

print(
    f"\nLoaded dataset-level results: "
    f"{len(results)} rows"
)

print(
    f"Datasets: "
    f"{results['dataset'].nunique()}"
)

print(
    f"Algorithms: "
    f"{results['algorithm'].nunique()}"
)


# ============================================================
# DATASET-WISE COMPARISON
# ============================================================

dataset_comparison = (
    results[
        [
            "dataset",
            "category",
            "algorithm",
            "precision",
            "recall",
            "f1",
            "pr_auc",
            "runtime_seconds"
        ]
    ]
    .sort_values(
        ["dataset", "f1"],
        ascending=[True, False]
    )
)


dataset_comparison.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "\nDataset-wise comparison saved to:"
)

print(OUTPUT_FILE)


# ============================================================
# BEST ALGORITHM FOR EACH DATASET
# ============================================================

winners = (
    results
    .sort_values(
        ["dataset", "f1"],
        ascending=[True, False]
    )
    .groupby(
        "dataset",
        as_index=False
    )
    .first()
)

winners = winners[
    [
        "dataset",
        "category",
        "algorithm",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "runtime_seconds"
    ]
]

winners.rename(
    columns={
        "algorithm": "best_algorithm",
        "precision": "best_precision",
        "recall": "best_recall",
        "f1": "best_f1",
        "pr_auc": "best_pr_auc",
        "runtime_seconds": "best_runtime"
    },
    inplace=True
)

winners.to_csv(
    WINNER_FILE,
    index=False
)

print(
    "\nDataset winners saved to:"
)

print(WINNER_FILE)


# ============================================================
# NUMBER OF DATASET WINS
# ============================================================

win_counts = (
    winners["best_algorithm"]
    .value_counts()
    .rename_axis("algorithm")
    .reset_index(name="dataset_wins")
)

win_counts["win_percentage"] = (
    win_counts["dataset_wins"]
    / len(winners)
    * 100
)


print("\n")
print("=" * 70)
print("DATASET WIN COUNT")
print("=" * 70)

print(
    win_counts.to_string(
        index=False
    )
)


# ============================================================
# OVERALL DATASET-LEVEL PERFORMANCE
# ============================================================

performance_summary = (
    results
    .groupby("algorithm")
    .agg(

        datasets=(
            "dataset",
            "nunique"
        ),

        mean_precision=(
            "precision",
            "mean"
        ),

        median_precision=(
            "precision",
            "median"
        ),

        std_precision=(
            "precision",
            "std"
        ),

        mean_recall=(
            "recall",
            "mean"
        ),

        median_recall=(
            "recall",
            "median"
        ),

        std_recall=(
            "recall",
            "std"
        ),

        mean_f1=(
            "f1",
            "mean"
        ),

        median_f1=(
            "f1",
            "median"
        ),

        std_f1=(
            "f1",
            "std"
        ),

        mean_pr_auc=(
            "pr_auc",
            "mean"
        ),

        median_pr_auc=(
            "pr_auc",
            "median"
        ),

        std_pr_auc=(
            "pr_auc",
            "std"
        ),

        mean_runtime=(
            "runtime_seconds",
            "mean"
        )
    )
    .reset_index()
)


performance_summary.to_csv(
    SUMMARY_FILE,
    index=False
)


print("\n")
print("=" * 70)
print("OVERALL DATASET-LEVEL PERFORMANCE")
print("=" * 70)

print(
    performance_summary.to_string(
        index=False
    )
)

print(
    "\nPerformance summary saved to:"
)

print(SUMMARY_FILE)


# ============================================================
# DATASETS WHERE ALL ALGORITHMS FAIL
# ============================================================

dataset_f1 = (
    results
    .groupby("dataset")["f1"]
    .max()
)

failed_datasets = dataset_f1[
    dataset_f1.fillna(0) == 0
]

print("\n")
print("=" * 70)
print("DATASETS WHERE ALL ALGORITHMS HAVE F1 = 0")
print("=" * 70)

print(
    f"Count: {len(failed_datasets)}"
)

if len(failed_datasets) > 0:

    for dataset in failed_datasets.index:

        print(
            f" - {dataset}"
        )

else:

    print(
        "No dataset has F1 = 0 for all algorithms."
    )


# ============================================================
# BEST DATASETS
# ============================================================

best_datasets = (
    winners
    .sort_values(
        "best_f1",
        ascending=False
    )
    .head(10)
)

print("\n")
print("=" * 70)
print("TOP 10 DATASETS BY BEST ACHIEVED F1")
print("=" * 70)

print(
    best_datasets[
        [
            "dataset",
            "category",
            "best_algorithm",
            "best_f1"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# CONSISTENCY: F1 = 0 COUNT
# ============================================================

zero_f1_summary = (
    results
    .assign(
        zero_f1=lambda x:
        x["f1"].fillna(0) == 0
    )
    .groupby("algorithm")
    .agg(
        datasets=("dataset", "nunique"),
        zero_f1_datasets=("zero_f1", "sum")
    )
    .reset_index()
)

zero_f1_summary["nonzero_f1_percentage"] = (
    (
        zero_f1_summary["datasets"]
        - zero_f1_summary["zero_f1_datasets"]
    )
    / zero_f1_summary["datasets"]
    * 100
)


print("\n")
print("=" * 70)
print("ALGORITHM CONSISTENCY")
print("=" * 70)

print(
    zero_f1_summary.to_string(
        index=False
    )
)


# ============================================================
# COMPLETION
# ============================================================

print("\n")
print("=" * 70)
print("DATASET-WISE ANALYSIS COMPLETED")
print("=" * 70)