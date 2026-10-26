import time
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import (
    engineer_all_datasets,
    FEATURES
)

from evaluation import evaluate_predictions

from lstm_autoencoder import (
    train_lstm_autoencoder,
    calculate_reconstruction_errors,
    detect_anomalies_from_errors
)


# ============================================================
# BASELINE CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 20
EPOCHS = 10
BATCH_SIZE = 32
THRESHOLD_PERCENTILE = 95

ALGORITHM_NAME = "LSTM Autoencoder"


# ============================================================
# RUN BASELINE EXPERIMENT
# ============================================================

def run_lstm_baseline():

    print("=" * 70)
    print("LSTM AUTOENCODER BASELINE EXPERIMENT")
    print("=" * 70)

    print("\nLoading feature-engineered datasets...")

    datasets = engineer_all_datasets()

    print(
        f"Datasets loaded: {len(datasets)}"
    )

    results = []

    for dataset_number, (file_path, df) in enumerate(
        datasets.items(),
        start=1
    ):

        dataset_name = Path(file_path).name
        category = Path(file_path).parent.name

        print(
            f"\n[{dataset_number}/{len(datasets)}] "
            f"{dataset_name}"
        )

        y_true = df["ground_truth"].values

        start_time = time.perf_counter()

        try:

            # ------------------------------------------------
            # Prepare feature matrix
            # ------------------------------------------------

            X = (
                df[FEATURES]
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .fillna(0)
                .values
            )

            # ------------------------------------------------
            # Train LSTM Autoencoder
            # ------------------------------------------------

            model, scaler, X_sequences = (
                train_lstm_autoencoder(
                    X,
                    sequence_length=SEQUENCE_LENGTH,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE
                )
            )

            # ------------------------------------------------
            # Reconstruction errors
            # ------------------------------------------------

            errors = calculate_reconstruction_errors(
                model,
                X_sequences
            )

            # ------------------------------------------------
            # Convert sequence errors to point-level errors
            # ------------------------------------------------

            point_errors = np.zeros(
                len(X)
            )

            counts = np.zeros(
                len(X)
            )

            for i, error in enumerate(errors):

                end_index = (
                    i + SEQUENCE_LENGTH
                )

                point_errors[
                    i:end_index
                ] += error

                counts[
                    i:end_index
                ] += 1

            counts[
                counts == 0
            ] = 1

            point_errors = (
                point_errors / counts
            )

            # ------------------------------------------------
            # Threshold
            # ------------------------------------------------

            threshold = np.percentile(
                point_errors,
                THRESHOLD_PERCENTILE
            )

            y_pred = (
                point_errors >= threshold
            ).astype(int)

            # ------------------------------------------------
            # Evaluation
            # ------------------------------------------------

            metrics = evaluate_predictions(
                y_true,
                y_pred,
                point_errors
            )

            runtime = (
                time.perf_counter()
                - start_time
            )

            results.append({

                "dataset":
                    dataset_name,

                "category":
                    category,

                "algorithm":
                    ALGORITHM_NAME,

                "rows":
                    len(df),

                "features_used":
                    len(FEATURES),

                "sequence_length":
                    SEQUENCE_LENGTH,

                "epochs":
                    EPOCHS,

                "batch_size":
                    BATCH_SIZE,

                "threshold_percentile":
                    THRESHOLD_PERCENTILE,

                "actual_anomalies":
                    metrics["actual_anomalies"],

                "predicted_anomalies":
                    metrics["predicted_anomalies"],

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

            results.append({

                "dataset":
                    dataset_name,

                "category":
                    category,

                "algorithm":
                    ALGORITHM_NAME,

                "rows":
                    len(df),

                "features_used":
                    len(FEATURES),

                "sequence_length":
                    SEQUENCE_LENGTH,

                "epochs":
                    EPOCHS,

                "batch_size":
                    BATCH_SIZE,

                "threshold_percentile":
                    THRESHOLD_PERCENTILE,

                "actual_anomalies":
                    int(y_true.sum()),

                "predicted_anomalies":
                    None,

                "precision":
                    None,

                "recall":
                    None,

                "f1":
                    None,

                "pr_auc":
                    None,

                "true_positive":
                    None,

                "false_positive":
                    None,

                "false_negative":
                    None,

                "true_negative":
                    None,

                "runtime_seconds":
                    None
            })

    return pd.DataFrame(results)


# ============================================================
# CREATE SUMMARY
# ============================================================

def create_summary(results):

    summary = (
        results
        .groupby("algorithm")
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

    results = run_lstm_baseline()

    results_dir = Path("results")
    results_dir.mkdir(
        exist_ok=True
    )

    # --------------------------------------------------------
    # Dataset-level results
    # --------------------------------------------------------

    results_file = (
        results_dir
        / "lstm_baseline_dataset_results.csv"
    )

    results.to_csv(
        results_file,
        index=False
    )

    print(
        "\nDataset-level results saved to:"
    )

    print(results_file)

    # --------------------------------------------------------
    # Model summary
    # --------------------------------------------------------

    summary = create_summary(
        results
    )

    summary_file = (
        results_dir
        / "lstm_baseline_model_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("LSTM AUTOENCODER BASELINE MODEL SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nModel summary saved to:"
    )

    print(summary_file)

    print(
        "\nLSTM Autoencoder baseline experiment completed."
    )