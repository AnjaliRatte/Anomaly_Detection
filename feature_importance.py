import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance

from feature_engineering import engineer_all_datasets


FEATURES = [
    "value",
    "diff_1",
    "abs_diff",
    "pct_change",
    "rolling_mean_10",
    "rolling_std_10",
    "rolling_mean_30",
    "rolling_std_30",
    "z_score",
    "trend_deviation"
]


def calculate_feature_importance(df):
    """
    Calculate permutation importance using
    Isolation Forest anomaly scores.
    """

    X = df[FEATURES].copy()

    # Replace any remaining problematic values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X)

    # Isolation Forest score
    scores = model.decision_function(X)

    # Convert to anomaly score:
    # higher = more anomalous
    anomaly_scores = -scores

    # Permutation importance requires a scoring function.
    # We measure how much the anomaly-score distribution
    # changes when a feature is shuffled.
    baseline_std = np.std(anomaly_scores)

    importances = []

    rng = np.random.default_rng(42)

    for feature in FEATURES:

        X_permuted = X.copy()

        shuffled = X_permuted[feature].values.copy()

        rng.shuffle(shuffled)

        X_permuted[feature] = shuffled

        permuted_scores = -model.decision_function(
            X_permuted
        )

        importance = abs(
            np.std(permuted_scores) - baseline_std
        )

        importances.append(importance)

    return pd.DataFrame({
        "feature": FEATURES,
        "importance": importances
    })


def run_feature_importance():

    print("Loading feature-engineered datasets...")

    feature_data = engineer_all_datasets()

    print(
        f"Total datasets: {len(feature_data)}"
    )

    all_importance = []

    for index, (file_path, df) in enumerate(
        feature_data.items(),
        start=1
    ):

        print(
            f"[{index}/{len(feature_data)}] "
            f"{file_path}"
        )

        try:

            importance = calculate_feature_importance(
                df
            )

            importance["dataset"] = file_path

            all_importance.append(
                importance
            )

        except Exception as e:

            print(
                f"   Skipped: {e}"
            )

    importance_df = pd.concat(
        all_importance,
        ignore_index=True
    )

    # Aggregate importance across datasets
    summary = (
        importance_df
        .groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    return summary


if __name__ == "__main__":

    summary = run_feature_importance()

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE SUMMARY")
    print("=" * 60)

    print(
        summary.to_string(index=False)
    )

    # Create results directory
    import os

    os.makedirs(
        "results",
        exist_ok=True
    )

    # Save importance values
    summary.to_csv(
        "results/feature_importance.csv",
        index=False
    )

    # Plot
    plt.figure(figsize=(10, 6))

    plt.barh(
        summary["feature"],
        summary["importance"]
    )

    plt.xlabel(
        "Mean Permutation Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Feature Importance Across NAB Datasets"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        "results/feature_importance.png",
        dpi=300
    )

    plt.show()

    print(
        "\nFeature importance saved to:"
    )

    print(
        "results/feature_importance.csv"
    )

    print(
        "results/feature_importance.png"
    )