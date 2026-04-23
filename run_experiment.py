"""
Master Experiment Orchestrator
================================
Runs the full research pipeline end-to-end:
  Step 01: Data preprocessing
  Step 02: Feature engineering
  Step 03: Model training
  Step 04: Deployment strategy inference (edge, cloud, hybrid)
  Step 05: Network degradation simulation (20 conditions × 30 runs)
  Step 06: Evaluation & statistical testing
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yaml
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import load_config, preprocess_dataset
from src.feature_engineering import engineer_features
from src.model import train_and_save, load_trained_model, predict_anomalies
from src.deployment.edge_strategy import EdgeStrategy
from src.deployment.cloud_strategy import CloudStrategy
from src.deployment.hybrid_strategy import HybridStrategy
from src.network_simulation import NetworkSimulator, generate_condition_matrix
from src.metrics_logger import MetricsLogger
from src.evaluation import full_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("experiment.log"),
    ],
)
logger = logging.getLogger(__name__)


def run_experiment(config_path: str = "config/experiment_config.yaml"):
    """
    Run the full experiment pipeline.
    
    Total: 20 network conditions × 3 strategies × 30 runs = 1,800 inference logs
    """
    start_time = time.time()

    # ============================================================
    # Load Configuration
    # ============================================================
    logger.info("=" * 70)
    logger.info("EDGE COMPUTING IN PREDICTIVE MAINTENANCE — EXPERIMENT")
    logger.info("=" * 70)

    config = load_config(config_path)
    logger.info("Configuration loaded from %s", config_path)

    # ============================================================
    # Step 01: Data Preprocessing
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 01: Data Collection & Preprocessing")
    logger.info("=" * 70)

    subset = config["data"]["subsets"][0]  # Start with FD001
    processed_dir = Path(config["data"]["processed_dir"])
    train_csv = processed_dir / f"{subset}_train.csv"

    if train_csv.exists():
        logger.info("Loading preprocessed data from disk...")
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(processed_dir / f"{subset}_val.csv")
        test_df = pd.read_csv(processed_dir / f"{subset}_test.csv")
    else:
        train_df, val_df, test_df = preprocess_dataset(config, subset)

    # ============================================================
    # Step 02: Feature Engineering
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 02: Feature Engineering")
    logger.info("=" * 70)

    data = engineer_features(config, train_df, val_df, test_df)

    # ============================================================
    # Step 03: Model Training
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 03: LSTM Autoencoder Training")
    logger.info("=" * 70)

    model_path = Path(config["model"]["model_save_path"])
    threshold_path = model_path.parent / "anomaly_threshold.json"

    if model_path.exists() and threshold_path.exists():
        logger.info("Loading pre-trained model...")
        model, threshold = load_trained_model(config)
    else:
        logger.info("Training new model...")
        model_result = train_and_save(config, data)
        model = model_result["model"]
        threshold = model_result["threshold"]

    # Quick validation on test set
    test_preds, test_errors = predict_anomalies(model, data["test"]["X"], threshold)
    from sklearn.metrics import f1_score as sk_f1
    test_f1 = sk_f1(data["test"]["y"], test_preds, average="binary", zero_division=0)
    logger.info("Test set baseline F1-Score: %.4f", test_f1)

    # ============================================================
    # Step 04 & 05: Deployment Inference + Network Simulation
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEPS 04-05: Deployment Inference + Network Simulation")
    logger.info("=" * 70)

    # Initialize strategies (same model, same threshold for all)
    strategies = {
        "edge-only": EdgeStrategy(model, threshold, config),
        "cloud-only": CloudStrategy(model, threshold, config),
        "hybrid": HybridStrategy(model, threshold, config),
    }

    # Generate network condition matrix
    conditions = generate_condition_matrix(
        config["network"]["packet_loss_levels"],
        config["network"]["bandwidth_levels"],
    )
    runs_per_condition = config["network"]["runs_per_condition"]

    total_experiments = len(conditions) * len(strategies) * runs_per_condition
    logger.info(
        "Experiment matrix: %d conditions × %d strategies × %d runs = %d total",
        len(conditions), len(strategies), runs_per_condition, total_experiments,
    )

    # Initialize logger
    metrics_logger = MetricsLogger(config["evaluation"]["inference_logs_dir"])
    metrics_logger.reset_file()

    # Select test samples for inference
    # Use a subset of test data for each run to keep experiment feasible
    X_test = data["test"]["X"]
    y_test = data["test"]["y"]
    meta_test = data["test"]["meta"]
    
    # Use up to 50 samples per run for feasibility (or all if fewer)
    n_samples_per_run = min(50, len(X_test))
    logger.info("Using %d test samples per run", n_samples_per_run)

    # Random indices for each run (reproducible)
    rng = np.random.RandomState(config["preprocessing"]["random_seed"])

    # ---- Main Experiment Loop ----
    experiment_counter = 0

    for condition in tqdm(conditions, desc="Network Conditions"):
        cond_id = condition["condition_id"]
        pkt_loss = condition["packet_loss_pct"]
        bw = condition["bandwidth_mbps"]

        for strategy_name, strategy in strategies.items():
            for run in range(1, runs_per_condition + 1):
                # Create network simulator with unique seed per run
                run_seed = hash((cond_id, strategy_name, run)) % (2**31)
                net_sim = NetworkSimulator(
                    packet_loss_pct=pkt_loss,
                    bandwidth_mbps=bw,
                    base_latency_ms=config["network"]["base_latency_ms"],
                    jitter_ms=config["network"]["jitter_ms"],
                    seed=run_seed,
                )

                # Select samples for this run
                sample_indices = rng.choice(len(X_test), n_samples_per_run, replace=False)

                for idx in sample_indices:
                    X_sample = X_test[idx]
                    y_true = int(y_test[idx])
                    uid = int(meta_test.iloc[idx]["unit_id"])
                    cyc = int(meta_test.iloc[idx]["cycle"])

                    log_entry = strategy.run_inference(
                        X_sample=X_sample,
                        y_true=y_true,
                        unit_id=uid,
                        cycle=cyc,
                        network_sim=net_sim,
                        condition_id=cond_id,
                        run_number=run,
                    )

                    metrics_logger.log(log_entry)

                experiment_counter += 1

    logger.info(
        "Experiment complete: %d logs collected (%d runs)",
        metrics_logger.get_logs_count(), experiment_counter,
    )

    # ============================================================
    # Step 06: Evaluation & Statistical Testing
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 06: Evaluation & Statistical Testing")
    logger.info("=" * 70)

    eval_results = full_evaluation(config)

    # ============================================================
    # Summary
    # ============================================================
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETE — Total time: %.1f minutes", elapsed / 60)
    logger.info("=" * 70)
    logger.info("Results saved to: %s", config["evaluation"]["results_dir"])
    logger.info("Inference logs: %s", config["evaluation"]["inference_logs_dir"])

    # Print SQ3 recommendation
    logger.info("\n=== FINAL SQ3 DEPLOYMENT RECOMMENDATION ===")
    sq3 = eval_results["sq3_summary"]
    for _, row in sq3.iterrows():
        logger.info(
            "  %s: %d/%d conditions pass (F1≥%.2f & latency≤%dms), avg comm load=%.1f bytes",
            row["strategy"],
            row["passing_conditions"],
            row["total_conditions"],
            config["evaluation"]["f1_threshold"],
            config["evaluation"]["latency_threshold_ms"],
            row["avg_bytes_passing"] if not pd.isna(row["avg_bytes_passing"]) else -1,
        )

    return eval_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Edge Computing PdM Experiment")
    parser.add_argument(
        "--config", default="config/experiment_config.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 1 condition × 1 run (for testing)",
    )
    args = parser.parse_args()

    if args.quick:
        # Override config for quick testing
        config = load_config(args.config)
        config["network"]["packet_loss_levels"] = [0]
        config["network"]["bandwidth_levels"] = [100]
        config["network"]["runs_per_condition"] = 1
        config["model"]["epochs"] = 5

        # Save temp config
        quick_config_path = "config/quick_test_config.yaml"
        Path(quick_config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(quick_config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        run_experiment(quick_config_path)
    else:
        run_experiment(args.config)
