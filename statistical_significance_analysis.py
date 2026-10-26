from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


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
    "statistical_significance_results.csv"
)


# ============================================================
# LOAD FINAL DATASET-LEVEL RESULTS
# ============================================================

print("=" * 70)
print("STATISTICAL SIGNIFICANCE ANALYSIS")
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
# ALGORITHM LIST
# ============================================================

algorithms = sorted(
    results["algorithm"]
    .dropna()
    .unique()
)

print("\nAlgorithms being compared:")

for algorithm in algorithms:
    print(f" - {algorithm}")


# ============================================================
# CREATE PAIRED F1 TABLE
# ============================================================

f1_table = (
    results[
        [
            "dataset",
            "algorithm",
            "f1"
        ]
    ]
    .pivot(
        index="dataset",
        columns="algorithm",
        values="f1"
    )
)

print(
    f"\nPaired datasets available: "
    f"{len(f1_table)}"
)


# ============================================================
# PAIRWISE WILCOXON TEST
# ============================================================

print("\n")
print("=" * 70)
print("PAIRWISE WILCOXON SIGNED-RANK TEST")
print("=" * 70)

pairwise_results = []


for algorithm_a, algorithm_b in combinations(
    algorithms,
    2
):

    print("\n" + "-" * 70)

    print(
        f"Comparing: "
        f"{algorithm_a} vs {algorithm_b}"
    )

    pair_data = (
        f1_table[
            [
                algorithm_a,
                algorithm_b
            ]
        ]
        .dropna()
    )

    x = pair_data[algorithm_a].values
    y = pair_data[algorithm_b].values

    differences = x - y

    # --------------------------------------------------------
    # Check whether all differences are zero
    # --------------------------------------------------------

    if np.allclose(
        differences,
        0
    ):

        statistic = 0.0
        p_value = 1.0

    else:

        try:

            statistic, p_value = wilcoxon(
                x,
                y,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto"
            )

        except ValueError:

            statistic = np.nan
            p_value = np.nan

    # --------------------------------------------------------
    # Descriptive statistics
    # --------------------------------------------------------

    mean_a = np.mean(x)
    mean_b = np.mean(y)

    median_a = np.median(x)
    median_b = np.median(y)

    mean_difference = np.mean(
        differences
    )

    median_difference = np.median(
        differences
    )

    # --------------------------------------------------------
    # Count wins
    # --------------------------------------------------------

    a_wins = np.sum(
        differences > 0
    )

    b_wins = np.sum(
        differences < 0
    )

    ties = np.sum(
        differences == 0
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    pairwise_results.append({

        "algorithm_1":
            algorithm_a,

        "algorithm_2":
            algorithm_b,

        "paired_datasets":
            len(pair_data),

        "mean_f1_algorithm_1":
            mean_a,

        "mean_f1_algorithm_2":
            mean_b,

        "median_f1_algorithm_1":
            median_a,

        "median_f1_algorithm_2":
            median_b,

        "mean_f1_difference":
            mean_difference,

        "median_f1_difference":
            median_difference,

        "algorithm_1_wins":
            int(a_wins),

        "algorithm_2_wins":
            int(b_wins),

        "ties":
            int(ties),

        "wilcoxon_statistic":
            statistic,

        "p_value":
            p_value
    })

    print(
        f"Paired datasets: {len(pair_data)}"
    )

    print(
        f"Mean F1: "
        f"{algorithm_a} = {mean_a:.6f}, "
        f"{algorithm_b} = {mean_b:.6f}"
    )

    print(
        f"Mean difference: "
        f"{mean_difference:.6f}"
    )

    print(
        f"Wins: "
        f"{algorithm_a}={a_wins}, "
        f"{algorithm_b}={b_wins}, "
        f"Ties={ties}"
    )

    print(
        f"Wilcoxon statistic: "
        f"{statistic}"
    )

    print(
        f"p-value: "
        f"{p_value}"
    )


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

statistical_results = pd.DataFrame(
    pairwise_results
)


# ============================================================
# MULTIPLE COMPARISON CORRECTION
# ============================================================

print("\n")
print("=" * 70)
print("MULTIPLE COMPARISON CORRECTION")
print("=" * 70)

valid_p_values = (
    statistical_results["p_value"]
    .notna()
)

corrected_p_values = np.full(
    len(statistical_results),
    np.nan
)

if valid_p_values.any():

    corrected = multipletests(
        statistical_results.loc[
            valid_p_values,
            "p_value"
        ],
        alpha=0.05,
        method="holm"
    )

    corrected_p_values[
        valid_p_values.values
    ] = corrected[1]

    statistical_results[
        "adjusted_p_value"
    ] = corrected_p_values

    statistical_results[
        "significant_after_correction"
    ] = (
        corrected_p_values < 0.05
    )

else:

    statistical_results[
        "adjusted_p_value"
    ] = np.nan

    statistical_results[
        "significant_after_correction"
    ] = False


# ============================================================
# STATISTICAL INTERPRETATION
# ============================================================

statistical_results[
    "significance_level"
] = 0.05

statistical_results[
    "result"
] = np.where(
    statistical_results[
        "significant_after_correction"
    ],
    "Statistically significant",
    "Not statistically significant"
)


# ============================================================
# SAVE RESULTS
# ============================================================

statistical_results.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "\nStatistical results saved to:"
)

print(OUTPUT_FILE)


# ============================================================
# DISPLAY FINAL TABLE
# ============================================================

print("\n")
print("=" * 70)
print("STATISTICAL SIGNIFICANCE SUMMARY")
print("=" * 70)

display_columns = [

    "algorithm_1",

    "algorithm_2",

    "paired_datasets",

    "mean_f1_algorithm_1",

    "mean_f1_algorithm_2",

    "mean_f1_difference",

    "algorithm_1_wins",

    "algorithm_2_wins",

    "ties",

    "wilcoxon_statistic",

    "p_value",

    "adjusted_p_value",

    "result"
]

print(
    statistical_results[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# COMPLETION
# ============================================================

print("\n")
print("=" * 70)
print("STATISTICAL SIGNIFICANCE ANALYSIS COMPLETED")
print("=" * 70)