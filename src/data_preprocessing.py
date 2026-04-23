"""
Step 01 — Data Collection & Preprocessing
==========================================
Download NASA C-MAPSS FD001-FD004, parse, derive RUL and anomaly labels,
normalise per unit_id, and split into train/val/test sets.
"""

import os
import io
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Column names for C-MAPSS datasets (no header in raw files)
CMAPSS_COLUMNS = (
    ["unit_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_config(config_path: str = "config/experiment_config.yaml") -> dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_cmapss(raw_dir: str) -> str:
    """
    Download NASA C-MAPSS dataset if not already present.
    Returns the path to the extracted directory.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    # Check if data already exists (look for train_FD001.txt)
    if (raw_path / "train_FD001.txt").exists():
        logger.info("C-MAPSS data already downloaded at %s", raw_path)
        return str(raw_path)

    # The C-MAPSS dataset URL — try direct download
    zip_path = raw_path / "CMAPSSData.zip"

    # If zip already downloaded but not extracted
    if not zip_path.exists():
        logger.info("Downloading C-MAPSS dataset...")
        url = "https://data.nasa.gov/download/hi77-rrsp/application%2Fx-zip-compressed"
        try:
            urllib.request.urlretrieve(url, str(zip_path))
            logger.info("Download complete: %s", zip_path)
        except Exception as e:
            logger.warning("Auto-download failed: %s", e)
            logger.info(
                "Please manually download the C-MAPSS dataset from:\n"
                "  https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data\n"
                "Extract it to: %s", raw_path
            )
            raise FileNotFoundError(
                f"C-MAPSS data not found at {raw_path}. "
                "Please download manually and extract to this directory."
            )

    # Extract zip
    logger.info("Extracting C-MAPSS dataset...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(raw_path))

    # Some zips have a nested folder — flatten if needed
    nested = raw_path / "CMAPSSData"
    if nested.exists():
        for f in nested.iterdir():
            f.rename(raw_path / f.name)
        nested.rmdir()

    logger.info("Extraction complete. Files at: %s", raw_path)
    return str(raw_path)


def parse_cmapss_file(filepath: str) -> pd.DataFrame:
    """Parse a single C-MAPSS text file into a DataFrame."""
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)
    return df


def load_subset(raw_dir: str, subset: str) -> tuple:
    """
    Load a C-MAPSS subset (e.g., 'FD001').
    Returns (train_df, test_df, rul_df).
    """
    raw_path = Path(raw_dir)
    train_path = raw_path / f"train_{subset}.txt"
    test_path = raw_path / f"test_{subset}.txt"
    rul_path = raw_path / f"RUL_{subset}.txt"

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    train_df = parse_cmapss_file(str(train_path))
    test_df = parse_cmapss_file(str(test_path))
    rul_df = pd.read_csv(str(rul_path), sep=r"\s+", header=None, names=["RUL_remaining"])

    logger.info(
        "Loaded %s — Train: %d rows (%d units), Test: %d rows (%d units)",
        subset,
        len(train_df), train_df["unit_id"].nunique(),
        len(test_df), test_df["unit_id"].nunique(),
    )
    return train_df, test_df, rul_df


def derive_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive Remaining Useful Life (RUL) for each row.
    RUL = max_cycle_for_unit - current_cycle
    """
    df = df.copy()
    max_cycles = df.groupby("unit_id")["cycle"].max().reset_index()
    max_cycles.columns = ["unit_id", "max_cycle"]
    df = df.merge(max_cycles, on="unit_id", how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def derive_anomaly_label(df: pd.DataFrame, threshold: int = 30) -> pd.DataFrame:
    """
    Derive binary anomaly label from RUL.
    anomaly_label = 1 if RUL <= threshold, else 0
    """
    df = df.copy()
    df["anomaly_label"] = (df["RUL"] <= threshold).astype(int)
    normal_count = (df["anomaly_label"] == 0).sum()
    anomaly_count = (df["anomaly_label"] == 1).sum()
    logger.info(
        "Anomaly labels (threshold=%d): Normal=%d (%.1f%%), Anomaly=%d (%.1f%%)",
        threshold, normal_count, 100 * normal_count / len(df),
        anomaly_count, 100 * anomaly_count / len(df),
    )
    return df


def normalize_sensors(df: pd.DataFrame, sensor_cols: list, fit_scalers: dict = None):
    """
    Normalise sensor readings using MinMaxScaler per unit_id.
    Prevents data leakage by fitting per unit.
    
    If fit_scalers is provided, uses those (for val/test sets).
    Returns (normalised_df, scalers_dict).
    """
    df = df.copy()
    scalers = fit_scalers or {}

    # Convert sensor columns to float64 to avoid LossySetitemError
    for col in sensor_cols:
        df[col] = df[col].astype(np.float64)

    for uid in df["unit_id"].unique():
        mask = df["unit_id"] == uid
        if uid not in scalers:
            scaler = MinMaxScaler()
            df.loc[mask, sensor_cols] = scaler.fit_transform(df.loc[mask, sensor_cols])
            scalers[uid] = scaler
        else:
            df.loc[mask, sensor_cols] = scalers[uid].transform(df.loc[mask, sensor_cols])

    return df, scalers


def inject_timestamps(df: pd.DataFrame, freq_hz: int = 1) -> pd.DataFrame:
    """
    Inject synthetic timestamps at the given frequency (Hz) per cycle.
    Each unit starts at t=0.
    """
    df = df.copy()
    df["timestamp_injected"] = df.groupby("unit_id")["cycle"].transform(
        lambda x: (x - x.min()) / freq_hz
    )
    return df


def split_by_unit(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple:
    """
    Split data by unit_id (entire engine trajectories stay in one split).
    Returns (train_df, val_df, test_df).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    unit_ids = df["unit_id"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(unit_ids)

    n = len(unit_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_units = unit_ids[:n_train]
    val_units = unit_ids[n_train : n_train + n_val]
    test_units = unit_ids[n_train + n_val :]

    train_df = df[df["unit_id"].isin(train_units)].copy()
    val_df = df[df["unit_id"].isin(val_units)].copy()
    test_df = df[df["unit_id"].isin(test_units)].copy()

    logger.info(
        "Split — Train: %d units (%d rows), Val: %d units (%d rows), Test: %d units (%d rows)",
        len(train_units), len(train_df),
        len(val_units), len(val_df),
        len(test_units), len(test_df),
    )
    return train_df, val_df, test_df


def preprocess_dataset(config: dict, subset: str = "FD001"):
    """
    Full preprocessing pipeline for a given C-MAPSS subset.
    Returns processed DataFrames and saves to disk.
    """
    raw_dir = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]
    threshold = config["preprocessing"]["rul_anomaly_threshold"]
    seed = config["preprocessing"]["random_seed"]

    # Ensure directories exist
    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Download / verify data
    download_cmapss(raw_dir)

    # Step 2: Load subset
    train_df, test_df_orig, rul_df = load_subset(raw_dir, subset)

    # Step 3: Derive RUL and anomaly labels (for training data)
    train_df = derive_rul(train_df)
    train_df = derive_anomaly_label(train_df, threshold)

    # For test data: use the provided RUL file
    # The RUL file gives the remaining RUL at the last cycle of each unit in test
    # We need to derive per-cycle RUL for test data too
    test_df_orig = test_df_orig.copy()
    max_test_cycles = test_df_orig.groupby("unit_id")["cycle"].max().reset_index()
    max_test_cycles.columns = ["unit_id", "max_cycle"]
    rul_df["unit_id"] = range(1, len(rul_df) + 1)
    max_test_cycles = max_test_cycles.merge(rul_df, on="unit_id")
    max_test_cycles["true_max_cycle"] = max_test_cycles["max_cycle"] + max_test_cycles["RUL_remaining"]

    test_df = test_df_orig.merge(max_test_cycles[["unit_id", "true_max_cycle"]], on="unit_id")
    test_df["RUL"] = test_df["true_max_cycle"] - test_df["cycle"]
    test_df.drop(columns=["true_max_cycle"], inplace=True)
    test_df = derive_anomaly_label(test_df, threshold)

    # Combine for splitting (we'll re-split by unit_id)
    # Actually, C-MAPSS already provides train/test split. We'll use the train data
    # and split it into train/val/test ourselves for the model.
    # The original test set can be used as a held-out evaluation.

    # Step 4: Sensor columns
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]

    # Step 5: Normalise per unit_id
    train_df, scalers = normalize_sensors(train_df, sensor_cols)

    # Step 6: Split training data into train/val/test by unit_id
    train_split, val_split, test_split = split_by_unit(
        train_df,
        train_ratio=config["preprocessing"]["train_ratio"],
        val_ratio=config["preprocessing"]["val_ratio"],
        test_ratio=config["preprocessing"]["test_ratio"],
        seed=seed,
    )

    # Normalize val and test splits using the scalers fitted on their own units
    # (Per-unit normalization means each unit has its own scaler, so no leakage)
    # The scalers are already applied above since we normalized the full train_df

    # Step 7: Inject timestamps
    train_split = inject_timestamps(train_split)
    val_split = inject_timestamps(val_split)
    test_split = inject_timestamps(test_split)

    # Step 8: Save processed data
    train_split.to_csv(Path(processed_dir) / f"{subset}_train.csv", index=False)
    val_split.to_csv(Path(processed_dir) / f"{subset}_val.csv", index=False)
    test_split.to_csv(Path(processed_dir) / f"{subset}_test.csv", index=False)

    logger.info("Saved processed data to %s", processed_dir)

    return train_split, val_split, test_split


# ---- CLI entry point ----
if __name__ == "__main__":
    config = load_config()
    for subset in config["data"]["subsets"]:
        preprocess_dataset(config, subset)
