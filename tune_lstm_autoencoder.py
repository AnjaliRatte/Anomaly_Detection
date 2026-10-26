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
    calculate_reconstruction_errors
)


# ============================================================
# SMALL AUTOMATED SEARCH SPACE
# ============================================================

LSTM_PARAMS = {

    "sequence_length": [
        10,
        20
    ],

    "units": [
        32,
        64
    ],

    "epochs": [
        5,
        10
    ],

    "batch_size": [
        32
    ],

    "threshold_percentile": [
        95,
        97
    ]
}


# ============================================================
# GENERATE PARAMETER COMBINATIONS
# ============================================================

def generate_parameter_combinations():

    combinations = []

    for sequence_length in LSTM_PARAMS["sequence_length"]:

        for units in LSTM_PARAMS["units"]:

            for epochs in LSTM_PARAMS["epochs"]:

                for batch_size in LSTM_PARAMS["batch_size"]:

                    for threshold_percentile in LSTM_PARAMS[
                        "threshold_percentile"
                    ]:

                        combinations.append({

                            "sequence_length":
                                sequence_length,

                            "units":
                                units,

                            "epochs":
                                epochs,

                            "batch_size":
                                batch_size,

                            "threshold_percentile":
                                threshold_percentile
                        })

    return combinations


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(
    X,
    sequence_length
):

    sequences = []

    for i in range(
        len(X) - sequence_length + 1
    ):

        sequences.append(
            X[
                i:i + sequence_length
            ]
        )

    return np.asarray(
        sequences,
        dtype=np.float32
    )


# ============================================================
# RUN ONE CONFIGURATION
# ============================================================

def run_one_configuration(
    df,
    params
):

    X = (
        df[FEATURES]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
        .values
    )

    y_true = df[
        "ground_truth"
    ].values

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    model, scaler, X_sequences = (
        train_lstm_autoencoder(
            X,
            sequence_length=params[
                "sequence_length"
            ],
            epochs=params[
                "epochs"
            ],
            batch_size=params[
                "batch_size"
            ],
            units=params[
                "units"
            ]
        )
    )

    # --------------------------------------------------------
    # Reconstruction error
    # --------------------------------------------------------

    reconstructed = model.predict(
        X_sequences,
        verbose=0
    )

    sequence_errors = np.mean(
        np.square(
            X_sequences - reconstructed
        ),
        axis=(1, 2)
    )

    # --------------------------------------------------------
    # Convert sequence errors to point errors
    # --------------------------------------------------------

    point_errors = np.zeros(
        len(X)
    )

    counts = np.zeros(
        len(X)
    )

    sequence_length = params[
        "sequence_length"
    ]

    for i, error in enumerate(
        sequence_errors
    ):

        end_index = (
            i + sequence_length
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

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    threshold = np.percentile(
        point_errors,
        params[
            "threshold_percentile"
        ]
    )

    y_pred = (
        point_errors >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    metrics = evaluate_predictions(
        y_true,
        y_pred,
        point_errors
    )

    runtime = (
        time.perf_counter()
        - start_time
    )

    return metrics, runtime


# ============================================================
# MAIN TUNING
# ============================================================

def run_tuning():

    print("=" * 70)
    print(
        "LSTM AUTOENCODER AUTOMATED "
        "HYPERPARAMETER TUNING"
    )
    print("=" * 70)

    print(
        "\nLoading feature-engineered datasets..."
    )

    datasets = engineer_all_datasets()

    print(
        f"Datasets loaded: {len(datasets)}"
    )

    parameter_combinations = (
        generate_parameter_combinations()
    )

    print(
        "Parameter combinations:",
        len(parameter_combinations)
    )

    print(
        "\nIMPORTANT:"
        "\nTuning uses the first 12 datasets "
        "to keep computation practical."
    )

    tuning_datasets = list(
        datasets.items()
    )[:12]

    results = []

    for combination_number, params in enumerate(
        parameter_combinations,
        start=1
    ):

        print("\n" + "-" * 70)

        print(
            f"Configuration "
            f"{combination_number}/"
            f"{len(parameter_combinations)}"
        )

        print(params)

        print("-" * 70)

        for dataset_number, (
            file_path,
            df
        ) in enumerate(
            tuning_datasets,
            start=1
        ):

            dataset_name = Path(
                file_path
            ).name

            category = Path(
                file_path
            ).parent.name

            print(
                f"   Dataset "
                f"{dataset_number}/"
                f"{len(tuning_datasets)}: "
                f"{dataset_name}"
            )

            try:

                metrics, runtime = (
                    run_one_configuration(
                        df,
                        params
                    )
                )

                results.append({

                    "algorithm":
                        "LSTM Autoencoder",

                    "dataset":
                        dataset_name,

                    "category":
                        category,

                    "sequence_length":
                        params[
                            "sequence_length"
                        ],

                    "units":
                        params[
                            "units"
                        ],

                    "epochs":
                        params[
                            "epochs"
                        ],

                    "batch_size":
                        params[
                            "batch_size"
                        ],

                    "threshold_percentile":
                        params[
                            "threshold_percentile"
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

                    "runtime_seconds":
                        runtime
                })

                print(
                    f"      F1="
                    f"{metrics['f1']:.4f}"
                )

            except Exception as e:

                print(
                    f"      FAILED: {e}"
                )

    return pd.DataFrame(
        results
    )


# ============================================================
# SUMMARY
# ============================================================

def create_summary(
    results
):

    summary = (
        results
        .groupby(
            [
                "sequence_length",
                "units",
                "epochs",
                "batch_size",
                "threshold_percentile"
            ]
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
# EXECUTION
# ============================================================

if __name__ == "__main__":

    results = run_tuning()

    results_dir = Path(
        "results"
    )

    results_dir.mkdir(
        exist_ok=True
    )

    # --------------------------------------------------------
    # Detailed trials
    # --------------------------------------------------------

    trials_file = (
        results_dir
        / "lstm_hyperparameter_trials.csv"
    )

    results.to_csv(
        trials_file,
        index=False
    )

    print(
        "\nDetailed tuning results saved to:"
    )

    print(
        trials_file
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = create_summary(
        results
    )

    summary_file = (
        results_dir
        / "lstm_hyperparameter_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "LSTM HYPERPARAMETER SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        summary
        .sort_values(
            "mean_f1",
            ascending=False
        )
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Best configuration
    # --------------------------------------------------------

    best = (
        summary
        .sort_values(
            "mean_f1",
            ascending=False
        )
        .iloc[0]
    )

    best_file = (
        results_dir
        / "lstm_best_configuration.csv"
    )

    best.to_frame().T.to_csv(
        best_file,
        index=False
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "BEST LSTM AUTOENCODER CONFIGURATION"
    )

    print(
        "=" * 70
    )

    print(
        best.to_string()
    )

    print(
        "\nBest configuration saved to:"
    )

    print(
        best_file
    )

    print(
        "\nLSTM hyperparameter tuning completed."
    )