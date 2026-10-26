import numpy as np

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Input,
    LSTM,
    RepeatVector,
    TimeDistributed,
    Dense
)


# ============================================================
# TRAIN LSTM AUTOENCODER
# ============================================================

def train_lstm_autoencoder(
    X,
    sequence_length=20,
    epochs=10,
    batch_size=32,
    units=64
):

    # --------------------------------------------------------
    # Scale features
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------------
    # Create temporal sequences
    # --------------------------------------------------------

    sequences = []

    for i in range(
        len(X_scaled) - sequence_length + 1
    ):

        sequence = X_scaled[
            i:i + sequence_length
        ]

        sequences.append(sequence)

    X_sequences = np.asarray(
        sequences,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # LSTM Autoencoder
    # --------------------------------------------------------

    n_features = X_sequences.shape[2]

    model = Sequential([

        LSTM(
            64,
            activation="tanh",
            input_shape=(
                sequence_length,
                n_features
            ),
            return_sequences=False
        ),

        RepeatVector(
            sequence_length
        ),

        LSTM(
            64,
            activation="tanh",
            return_sequences=True
        ),

        TimeDistributed(
            Dense(n_features)
        )
    ])

    model.compile(
        optimizer=Adam(
            learning_rate=0.001
        ),
        loss="mse"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_sequences,
        X_sequences,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        shuffle=False,
        verbose=0
    )

    return (
        model,
        scaler,
        X_sequences
    )


# ============================================================
# CALCULATE RECONSTRUCTION ERRORS
# ============================================================

def calculate_reconstruction_errors(
    model,
    X_sequences
):

    reconstructed = model.predict(
        X_sequences,
        verbose=0
    )

    # Mean squared reconstruction error
    errors = np.mean(
        np.square(
            X_sequences - reconstructed
        ),
        axis=(1, 2)
    )

    return errors


# ============================================================
# DETECT ANOMALIES
# ============================================================

def detect_anomalies_from_errors(
    errors,
    threshold_percentile=95
):

    threshold = np.percentile(
        errors,
        threshold_percentile
    )

    predictions = (
        errors >= threshold
    ).astype(int)

    return (
        predictions,
        threshold
    )