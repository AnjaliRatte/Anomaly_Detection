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
    train_lstm_autoencoder
)


# ============================================================
# FINAL OPTIMIZED LSTM CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 10
UNITS = 64
EPOCHS = 10
BATCH_SIZE = 32
THRESHOLD_PERCENTILE = 97

ALGORITHM_NAME = "LSTM Autoencoder"


# ============================================================
# RUN FINAL LSTM EXPERIMENT
# ============================================================

def run_final_experiment():

    print("=" * 70)
    print("FINAL OPTIMIZED LSTM AUTOENCODER EXPERIMENT")
    print("=" * 70)

    print("\nSelected configuration:")
    print(
        f"Sequence length       : {SEQUENCE_LENGTH}"
    )
    print(
        f"LSTM units            : {UNITS}"
    )
    print(
        f"Epochs                : {EPOCHS}"
    )
    print(
        f"Batch size            : {BATCH_SIZE}"
    )
    print(
        f"Threshold percentile  : {THRESHOLD_PERCENTILE}"
    )

    print("\nLoading feature-engineered datasets...")

    datasets = engineer_all_datasets()

    print(
        f"Datasets loaded: {len(datasets)}"
    )

    results = []

    # ========================================================
    # PROCESS ALL 58 DATASETS
    # ========================================================

    for dataset_number, (file_path, df) in enumerate(
        datasets.items(),
        start=1
    ):

        dataset_name = Path(file_path).name
        category = Path(file_path).parent.name

        print("\n" + "-" * 70)

        print(
            f"[{dataset_number}/{len(datasets)}] "
            f"{dataset_name}"
        )

        print(
            f"Category: {category}"
        )

        y_true = df[
            "ground_truth"
        ].values

        start_time = time.perf_counter()

        try:

            # ------------------------------------------------
            # Prepare features
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
            # Train optimized LSTM Autoencoder
            # ------------------------------------------------

            model, scaler, X_sequences = (
                train_lstm_autoencoder(
                    X,
                    sequence_length=SEQUENCE_LENGTH,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    units=UNITS
                )
            )

            # ------------------------------------------------
            # Reconstruction
            # ------------------------------------------------

            reconstructed = model.predict(
                X_sequences,
                verbose=0
            )

            # ------------------------------------------------
            # Sequence-level reconstruction error
            # ------------------------------------------------

            sequence_errors = np.mean(
                np.square(
                    X_sequences -
                    reconstructed
                ),
                axis=(1, 2)
            )

            # ------------------------------------------------
            # Convert sequence errors
            # to point-level errors
            # ------------------------------------------------

            point_errors = np.zeros(
                len(X)
            )

            counts = np.zeros(
                len(X)
            )

            for i, error in enumerate(
                sequence_errors
            ):

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
            # Determine anomaly threshold
            # ------------------------------------------------

            threshold = np.percentile(
                point_errors,
                THRESHOLD_PERCENTILE
            )

            # ------------------------------------------------
            # Generate predictions
            # ------------------------------------------------

            y_pred = (
                point_errors >= threshold
            ).astype(int)

            # ------------------------------------------------
            # Evaluate
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

            # ------------------------------------------------
            # Save results
            # ------------------------------------------------

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

                "units":
                    UNITS,

                "epochs":
                    EPOCHS,

                "batch_size":
                    BATCH_SIZE,

                "threshold_percentile":
                    THRESHOLD_PERCENTILE,

                "threshold":
                    threshold,

                "actual_anomalies":
                    metrics[
                        "actual_anomalies"
                    ],

                "predicted_anomalies":
                    metrics[
                        "predicted_anomalies"
                    ],

                "precision":
                    metrics[
                        "precision"
                    ],

                "recall":
                    metrics[
                        "recall"
                    ],

                "f1":
                    metrics[
                        "f1"
                    ],

                "pr_auc":
                    metrics[
                        "pr_auc"
                    ],

                "true_positive":
                    metrics[
                        "true_positive"
                    ],

                "false_positive":
                    metrics[
                        "false_positive"
                    ],

                "false_negative":
                    metrics[
                        "false_negative"
                    ],

                "true_negative":
                    metrics[
                        "true_negative"
                    ],

                "runtime_seconds":
                    runtime
            })

            print(
                f"F1={metrics['f1']:.4f} | "
                f"Precision={metrics['precision']:.4f} | "
                f"Recall={metrics['recall']:.4f} | "
                f"PR-AUC={metrics['pr_auc']:.4f}"
            )

            print(
                f"Runtime={runtime:.2f} seconds"
            )

        except Exception as e:

            runtime = (
                time.perf_counter()
                - start_time
            )

            print(
                f"FAILED: {dataset_name}"
            )

            print(
                f"Error: {e}"
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

                "units":
                    UNITS,

                "epochs":
                    EPOCHS,

                "batch_size":
                    BATCH_SIZE,

                "threshold_percentile":
                    THRESHOLD_PERCENTILE,

                "threshold":
                    None,

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
                    runtime
            })

    return pd.DataFrame(results)


# ============================================================
# CREATE FINAL SUMMARY
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

    results = run_final_experiment()

    results_dir = Path(
        "results"
    )

    results_dir.mkdir(
        exist_ok=True
    )

    # ========================================================
    # SAVE DATASET-LEVEL RESULTS
    # ========================================================

    results_file = (
        results_dir
        / "final_lstm_dataset_results.csv"
    )

    results.to_csv(
        results_file,
        index=False
    )

    print(
        "\nDataset-level results saved to:"
    )

    print(
        results_file
    )

    # ========================================================
    # CREATE MODEL SUMMARY
    # ========================================================

    summary = create_summary(
        results
    )

    summary_file = (
        results_dir
        / "final_lstm_model_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    # ========================================================
    # DISPLAY FINAL SUMMARY
    # ========================================================

    print("\n")

    print("=" * 70)
    print("FINAL OPTIMIZED LSTM AUTOENCODER SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nModel summary saved to:"
    )

    print(
        summary_file
    )

    print(
        "\nFinal optimized LSTM Autoencoder "
        "experiment completed."
    )