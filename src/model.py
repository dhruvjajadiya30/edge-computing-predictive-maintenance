"""
Step 03 — LSTM Autoencoder Model
=================================
Define, train, and save the LSTM Autoencoder for anomaly detection.
Trained on normal data only (unsupervised reconstruction learning).
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_lstm_autoencoder(
    window_size: int,
    n_features: int,
    encoder_units: list = None,
    decoder_units: list = None,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Build an LSTM Autoencoder model for time-series reconstruction.
    
    Architecture:
        Encoder: LSTM(64) → LSTM(32) → latent
        Decoder: RepeatVector → LSTM(32) → LSTM(64) → TimeDistributed(Dense)
    """
    if encoder_units is None:
        encoder_units = [64, 32]
    if decoder_units is None:
        decoder_units = [32, 64]

    # --- Encoder ---
    inputs = keras.Input(shape=(window_size, n_features))

    # First encoder LSTM
    encoded = layers.LSTM(
        encoder_units[0], activation="relu", return_sequences=True
    )(inputs)

    # Second encoder LSTM (bottleneck)
    encoded = layers.LSTM(
        encoder_units[1], activation="relu", return_sequences=False
    )(encoded)

    # --- Decoder ---
    # Repeat the latent vector for each timestep
    decoded = layers.RepeatVector(window_size)(encoded)

    # First decoder LSTM
    decoded = layers.LSTM(
        decoder_units[0], activation="relu", return_sequences=True
    )(decoded)

    # Second decoder LSTM
    decoded = layers.LSTM(
        decoder_units[1], activation="relu", return_sequences=True
    )(decoded)

    # Output layer: reconstruct original features
    outputs = layers.TimeDistributed(layers.Dense(n_features))(decoded)

    model = keras.Model(inputs=inputs, outputs=outputs, name="lstm_autoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )

    logger.info("LSTM Autoencoder built:")
    model.summary(print_fn=logger.info)

    return model


def train_model(
    model: keras.Model,
    X_train_normal: np.ndarray,
    X_val_normal: np.ndarray,
    config: dict,
) -> keras.callbacks.History:
    """
    Train the autoencoder on normal data only.
    Uses reconstruction learning: input = output target.
    """
    model_save_path = config["model"]["model_save_path"]
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)

    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config["model"]["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            model_save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    logger.info(
        "Training LSTM Autoencoder — %d normal train samples, %d normal val samples",
        len(X_train_normal), len(X_val_normal),
    )

    history = model.fit(
        X_train_normal, X_train_normal,  # Reconstruct input
        validation_data=(X_val_normal, X_val_normal),
        epochs=config["model"]["epochs"],
        batch_size=config["model"]["batch_size"],
        callbacks=cb,
        verbose=1,
    )

    logger.info("Training complete. Best val_loss: %.6f", min(history.history["val_loss"]))
    return history


def compute_anomaly_threshold(
    model: keras.Model,
    X_val_normal: np.ndarray,
    sigma_multiplier: float = 2.0,
) -> float:
    """
    Compute anomaly threshold as mean + N*sigma of reconstruction error
    on the validation set (normal data only).
    """
    reconstructions = model.predict(X_val_normal, verbose=0)
    mse_per_sample = np.mean(np.square(X_val_normal - reconstructions), axis=(1, 2))

    mean_error = np.mean(mse_per_sample)
    std_error = np.std(mse_per_sample)
    threshold = mean_error + sigma_multiplier * std_error

    logger.info(
        "Anomaly threshold: %.6f (mean=%.6f, std=%.6f, sigma_mult=%.1f)",
        threshold, mean_error, std_error, sigma_multiplier,
    )
    return threshold


def predict_anomalies(
    model: keras.Model,
    X: np.ndarray,
    threshold: float,
) -> tuple:
    """
    Predict anomalies based on reconstruction error exceeding threshold.
    
    Returns:
        predictions: binary array (1=anomaly, 0=normal)
        reconstruction_errors: per-sample MSE values
    """
    reconstructions = model.predict(X, verbose=0)
    errors = np.mean(np.square(X - reconstructions), axis=(1, 2))
    predictions = (errors > threshold).astype(int)
    return predictions, errors


def train_and_save(config: dict, data: dict) -> dict:
    """
    Full model training pipeline:
    1. Filter normal-only data for training
    2. Build LSTM Autoencoder
    3. Train on normal data
    4. Compute anomaly threshold
    5. Save model + threshold
    
    Returns dict with model, threshold, and history.
    """
    window_size = config["features"]["sliding_window_size"]
    n_features = config["features"]["pca_n_components"]

    # Filter normal data only for training (anomaly_label == 0)
    train_normal_mask = data["train"]["y"] == 0
    X_train_normal = data["train"]["X"][train_normal_mask]

    val_normal_mask = data["val"]["y"] == 0
    X_val_normal = data["val"]["X"][val_normal_mask]

    logger.info(
        "Normal data — Train: %d / %d, Val: %d / %d",
        len(X_train_normal), len(data["train"]["X"]),
        len(X_val_normal), len(data["val"]["X"]),
    )

    # Build model
    model = build_lstm_autoencoder(
        window_size=window_size,
        n_features=n_features,
        encoder_units=config["model"]["encoder_units"],
        decoder_units=config["model"]["decoder_units"],
        learning_rate=config["model"]["learning_rate"],
    )

    # Train
    history = train_model(model, X_train_normal, X_val_normal, config)

    # Compute threshold
    threshold = compute_anomaly_threshold(
        model, X_val_normal,
        sigma_multiplier=config["model"]["anomaly_threshold_sigma"],
    )

    # Save threshold
    threshold_path = Path(config["model"]["model_save_path"]).parent / "anomaly_threshold.json"
    with open(str(threshold_path), "w") as f:
        json.dump({"threshold": float(threshold)}, f, indent=2)
    logger.info("Saved anomaly threshold to %s", threshold_path)

    return {
        "model": model,
        "threshold": threshold,
        "history": history,
    }


def load_trained_model(config: dict) -> tuple:
    """
    Load a previously trained model and its threshold.
    Returns (model, threshold).
    """
    model_path = config["model"]["model_save_path"]
    threshold_path = Path(model_path).parent / "anomaly_threshold.json"

    model = keras.models.load_model(model_path)
    with open(str(threshold_path), "r") as f:
        threshold = json.load(f)["threshold"]

    logger.info("Loaded model from %s (threshold=%.6f)", model_path, threshold)
    return model, threshold
