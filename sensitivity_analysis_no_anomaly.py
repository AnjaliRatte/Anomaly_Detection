
from pathlib import Path
from itertools import combinations

import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

INPUT_FILE = RESULTS_DIR / "final_optimized_dataset_results.csv"
OUTPUT_FILE = RESULTS_DIR / "sensitivity_statistical_significance_results.csv"


def main():
    # Load existing results; do not modify the source CSV
    results = pd.read_csv(INPUT_FILE)

    required_columns = {
        "dataset", "algorithm", "f1", "actual_anomalies"
    }

    missing = required_columns - set(results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Keep only datasets with at least one labeled anomaly
    filtered = results[
        pd.to_numeric(results["actual_anomalies"], errors="coerce") > 0
    ].copy()

    # Build paired F1 scores for each dataset and algorithm
    f1_table = filtered.pivot(
        index="dataset",
        columns="algorithm",
        values="f1"
    )

    # Retain datasets with valid F1 values for all algorithms
    f1_table = f1_table.dropna()

    algorithms = sorted(f1_table.columns.tolist())

    print("=" * 65)
    print("SENSITIVITY ANALYSIS: DATASETS WITH LABELED ANOMALIES")
    print("=" * 65)
    print(f"Datasets with labeled anomalies in input: "
          f"{filtered['dataset'].nunique()}")
    print(f"Datasets retained for paired tests: {len(f1_table)}")
    print(f"Algorithms: {len(algorithms)}")
    print(f"Algorithm names: {algorithms}")

    if len(algorithms) < 2:
        raise ValueError("At least two algorithms are required.")

    if len(f1_table) < 2:
        raise ValueError("Not enough paired datasets for testing.")

    records = []

    # Pairwise Wilcoxon signed-rank tests
    for algorithm_a, algorithm_b in combinations(algorithms, 2):
        pair = f1_table[[algorithm_a, algorithm_b]].dropna()

        x = pair[algorithm_a].astype(float)
        y = pair[algorithm_b].astype(float)
        differences = x - y

        if (differences == 0).all():
            statistic = 0.0
            p_value = 1.0
        else:
            test = wilcoxon(
                x,
                y,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto"
            )
            statistic = float(test.statistic)
            p_value = float(test.pvalue)

        records.append({
            "algorithm_a": algorithm_a,
            "algorithm_b": algorithm_b,
            "n_datasets": len(pair),
            "mean_f1_a": x.mean(),
            "mean_f1_b": y.mean(),
            "median_f1_a": x.median(),
            "median_f1_b": y.median(),
            "mean_difference_a_minus_b": differences.mean(),
            "median_difference_a_minus_b": differences.median(),
            "wins_a": int((x > y).sum()),
            "wins_b": int((y > x).sum()),
            "ties": int((x == y).sum()),
            "wilcoxon_statistic": statistic,
            "p_value": p_value
        })

    output = pd.DataFrame(records)

    # Correct for multiple pairwise tests using Holm's method
    reject, adjusted_p, _, _ = multipletests(
        output["p_value"],
        alpha=0.05,
        method="holm"
    )

    output["holm_adjusted_p_value"] = adjusted_p
    output["significant_at_0_05"] = reject

    output.to_csv(OUTPUT_FILE, index=False)

    print("\nPAIRWISE RESULTS")
    print(output[
        [
            "algorithm_a",
            "algorithm_b",
            "n_datasets",
            "mean_difference_a_minus_b",
            "p_value",
            "holm_adjusted_p_value",
            "significant_at_0_05"
        ]
    ].to_string(index=False))

    print(f"\nSaved separate results to:\n{OUTPUT_FILE}")
    print("\nOriginal experiment files were not modified.")


if __name__ == "__main__":
    main()
