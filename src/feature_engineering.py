"""
Step 02 — Feature Engineering
==============================
Rolling statistics, PCA dimensionality reduction, and sliding window
construction for LSTM input sequences.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from pathlib import Path
import pickle
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]


def compute_rolling_features(df: pd.DataFrame, sensor_cols: list, window: int = 5) -> pd.DataFrame:
    """
    Compute rolling statistics per sensor per unit_id:
    - Rolling mean (window size)
    - Rolling standard deviation (window size)
    - Rate of change (first difference)
    """
    df = df.copy()
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)

    for col in sensor_cols:
        grouped = df.groupby("unit_id")[col]
        df[f"{col}_rolling_mean"] = grouped.transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f"{col}_rolling_std"] = grouped.transform(
            lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
        )
        df[f"{col}_roc"] = grouped.transform(
            lambda x: x.diff().fillna(0)
        )

    logger.info(
        "Computed rolling features: %d new columns (window=%d)",
        len(sensor_cols) * 3, window,
    )
    return df


def apply_pca(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sensor_cols: list,
    n_components: int = 8,
    variance_threshold: float = 0.95,
    save_path: str = None,
) -> tuple:
    """
    Apply PCA to reduce sensor dimensions.
    Fits PCA on training data only; transforms val/test with the same PCA.
    
    Returns (train_pca, val_pca, test_pca, pca_object, pca_col_names)
    """
    # Collect all feature columns (original sensors + rolling features)
    feature_cols = []
    for col in sensor_cols:
        feature_cols.extend([col, f"{col}_rolling_mean", f"{col}_rolling_std", f"{col}_roc"])

    # Filter to columns that actually exist
    feature_cols = [c for c in feature_cols if c in train_df.columns]

    logger.info("PCA input features: %d columns", len(feature_cols))

    # Handle NaN / Inf
    train_features = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    val_features = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    test_features = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Fit PCA on training data
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_features)

    explained_var = sum(pca.explained_variance_ratio_)
    logger.info(
        "PCA: %d components explain %.2f%% of variance (threshold: %.0f%%)",
        n_components, explained_var * 100, variance_threshold * 100,
    )

    if explained_var < variance_threshold:
        logger.warning(
            "PCA explained variance (%.2f%%) below threshold (%.0f%%). "
            "Consider increasing n_components.",
            explained_var * 100, variance_threshold * 100,
        )

    # Transform val/test
    val_pca = pca.transform(val_features)
    test_pca = pca.transform(test_features)

    # Create column names
    pca_col_names = [f"pca_{i+1}" for i in range(n_components)]

    # Add PCA components back to DataFrames
    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()

    for i, col_name in enumerate(pca_col_names):
        train_out[col_name] = train_pca[:, i]
        val_out[col_name] = val_pca[:, i]
        test_out[col_name] = test_pca[:, i]

    # Save PCA model
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(pca, f)
        logger.info("Saved PCA model to %s", save_path)

    return train_out, val_out, test_out, pca, pca_col_names


def create_sliding_windows(
    df: pd.DataFrame,
    feature_cols: list,
    window_size: int = 30,
    label_col: str = "anomaly_label",
) -> tuple:
    """
    Create sliding window sequences for LSTM input.
    
    For each unit_id, slides a window of `window_size` cycles and extracts:
    - X: feature sequences of shape (n_samples, window_size, n_features)
    - y: label for the last timestep in each window
    - metadata: DataFrame with unit_id, cycle, timestamp for each sample
    
    Returns (X, y, metadata)
    """
    X_list = []
    y_list = []
    meta_list = []

    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)

    for uid in df["unit_id"].unique():
        unit_data = df[df["unit_id"] == uid]
        features = unit_data[feature_cols].values
        labels = unit_data[label_col].values

        if len(unit_data) < window_size:
            # Pad with first row if unit has fewer cycles than window
            pad_length = window_size - len(unit_data)
            features = np.vstack([np.tile(features[0], (pad_length, 1)), features])
            labels = np.concatenate([np.zeros(pad_length), labels])

        for i in range(len(features) - window_size + 1):
            X_list.append(features[i : i + window_size])
            y_list.append(labels[i + window_size - 1])

            # Metadata for the last timestep in the window
            row_idx = min(i + window_size - 1, len(unit_data) - 1)
            meta_list.append({
                "unit_id": uid,
                "cycle": unit_data.iloc[row_idx]["cycle"] if row_idx < len(unit_data) else -1,
                "timestamp_injected": unit_data.iloc[row_idx].get("timestamp_injected", 0) if row_idx < len(unit_data) else 0,
            })

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    metadata = pd.DataFrame(meta_list)

    logger.info(
        "Sliding windows: %d samples, shape=(%d, %d), anomaly_ratio=%.2f%%",
        len(X), window_size, len(feature_cols),
        100 * y.sum() / len(y),
    )
    return X, y, metadata


def engineer_features(config: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Full feature engineering pipeline:
    1. Compute rolling statistics
    2. Apply PCA
    3. Create sliding windows
    
    Returns dict with windowed data for each split.
    """
    rolling_window = config["features"]["rolling_window_size"]
    n_components = config["features"]["pca_n_components"]
    variance_threshold = config["features"]["pca_variance_threshold"]
    sliding_window = config["features"]["sliding_window_size"]

    # Step 1: Rolling features
    logger.info("Computing rolling features...")
    train_df = compute_rolling_features(train_df, SENSOR_COLS, rolling_window)
    val_df = compute_rolling_features(val_df, SENSOR_COLS, rolling_window)
    test_df = compute_rolling_features(test_df, SENSOR_COLS, rolling_window)

    # Step 2: PCA
    logger.info("Applying PCA...")
    train_df, val_df, test_df, pca_model, pca_cols = apply_pca(
        train_df, val_df, test_df,
        SENSOR_COLS, n_components, variance_threshold,
        save_path="models/pca_model.pkl",
    )

    # Step 3: Sliding windows using PCA features
    logger.info("Creating sliding windows...")
    X_train, y_train, meta_train = create_sliding_windows(train_df, pca_cols, sliding_window)
    X_val, y_val, meta_val = create_sliding_windows(val_df, pca_cols, sliding_window)
    X_test, y_test, meta_test = create_sliding_windows(test_df, pca_cols, sliding_window)

    return {
        "train": {"X": X_train, "y": y_train, "meta": meta_train},
        "val": {"X": X_val, "y": y_val, "meta": meta_val},
        "test": {"X": X_test, "y": y_test, "meta": meta_test},
        "pca_model": pca_model,
        "pca_cols": pca_cols,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }
