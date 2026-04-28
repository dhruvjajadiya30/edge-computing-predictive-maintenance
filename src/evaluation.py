"""
Step 06 — Evaluation
=====================
Compute F1-Score, E2E Alert Latency, Communication Load per strategy per condition.
Generate SQ3 deployment recommendation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_inference_logs(logs_path: str = "results/inference_logs/inference_logs.csv") -> pd.DataFrame:
    """Load the inference logs CSV file."""
    df = pd.read_csv(logs_path)
    logger.info("Loaded %d inference logs", len(df))
    return df


def compute_metrics_per_condition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute M1 (F1-Score), M2 (E2E-AL), M3 (Communication Load)
    per strategy × per network condition.
    
    Returns a summary DataFrame.
    """
    results = []

    for (strategy, condition_id), group in df.groupby(["strategy", "network_condition_id"]):
        # M1: F1-Score
        y_true = group["anomaly_label"].values
        y_pred = group["prediction"].values

        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
        precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
        recall = recall_score(y_true, y_pred, average="binary", zero_division=0)

        # M2: E2E Alert Latency
        latency_mean = group["latency_ms"].mean()
        latency_std = group["latency_ms"].std()
        latency_median = group["latency_ms"].median()

        # M3: Communication Load
        bytes_mean = group["bytes_transmitted"].mean()
        bytes_std = group["bytes_transmitted"].std()
        bytes_total = group["bytes_transmitted"].sum()

        # Network condition info
        packet_loss = group["packet_loss_pct"].iloc[0]
        bandwidth = group["bandwidth_mbps"].iloc[0]
        n_runs = group["run_number"].nunique()

        results.append({
            "strategy": strategy,
            "condition_id": condition_id,
            "packet_loss_pct": packet_loss,
            "bandwidth_mbps": bandwidth,
            "n_samples": len(group),
            "n_runs": n_runs,
            # M1
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            # M2
            "latency_mean_ms": latency_mean,
            "latency_std_ms": latency_std,
            "latency_median_ms": latency_median,
            # M3
            "bytes_mean": bytes_mean,
            "bytes_std": bytes_std,
            "bytes_total": bytes_total,
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["strategy", "condition_id"]).reset_index(drop=True)

    logger.info("Computed metrics for %d strategy×condition combinations", len(results_df))
    return results_df


def sq3_decision(
    metrics_df: pd.DataFrame,
    f1_threshold: float = 0.80,
    latency_threshold_ms: float = 200.0,
) -> pd.DataFrame:
    """
    SQ3 Decision Logic:
    1. Filter strategies meeting F1 ≥ threshold AND latency ≤ threshold
    2. Among qualifying, select lowest communication load
    3. Report which conditions cause each strategy to fail
    
    Returns DataFrame with decision results.
    """
    # Add pass/fail columns
    df = metrics_df.copy()
    df["f1_pass"] = df["f1_score"] >= f1_threshold
    df["latency_pass"] = df["latency_mean_ms"] <= latency_threshold_ms
    df["both_pass"] = df["f1_pass"] & df["latency_pass"]

    # Summary per strategy
    strategy_summary = []
    for strategy in df["strategy"].unique():
        s_df = df[df["strategy"] == strategy]
        total_conditions = len(s_df)
        passing_conditions = s_df["both_pass"].sum()
        failing_conditions = total_conditions - passing_conditions

        # Conditions that fail
        failed = s_df[~s_df["both_pass"]]
        failed_list = failed["condition_id"].tolist()

        # Average metrics across all conditions
        avg_f1 = s_df["f1_score"].mean()
        avg_latency = s_df["latency_mean_ms"].mean()
        avg_bytes = s_df["bytes_mean"].mean()

        # Average metrics across passing conditions only
        passing = s_df[s_df["both_pass"]]
        avg_bytes_passing = passing["bytes_mean"].mean() if len(passing) > 0 else np.nan

        strategy_summary.append({
            "strategy": strategy,
            "total_conditions": total_conditions,
            "passing_conditions": int(passing_conditions),
            "failing_conditions": int(failing_conditions),
            "pass_rate_pct": 100 * passing_conditions / total_conditions,
            "avg_f1_all": avg_f1,
            "avg_latency_all_ms": avg_latency,
            "avg_bytes_all": avg_bytes,
            "avg_bytes_passing": avg_bytes_passing,
            "failed_conditions": str(failed_list),
        })

    summary_df = pd.DataFrame(strategy_summary)
    summary_df = summary_df.sort_values(
        ["passing_conditions", "avg_bytes_passing"],
        ascending=[False, True],
    ).reset_index(drop=True)

    logger.info("\n=== SQ3 DEPLOYMENT RECOMMENDATION ===")
    logger.info("\n%s", summary_df.to_string(index=False))

    # Recommended strategy: highest pass rate, then lowest comm load
    recommended = summary_df.iloc[0]
    logger.info(
        "\nRECOMMENDED: %s (passes %d/%d conditions, avg %.1f bytes)",
        recommended["strategy"],
        recommended["passing_conditions"],
        recommended["total_conditions"],
        recommended["avg_bytes_passing"],
    )

    return summary_df


def full_evaluation(config: dict, logs_path: str = None) -> dict:
    """
    Run the complete evaluation pipeline:
    1. Load inference logs
    2. Compute per-condition metrics (F1, Latency, Bytes)
    3. Run SQ3 decision logic
    4. Save all results
    
    Returns dict with all results.
    """
    if logs_path is None:
        logs_path = f"{config['evaluation']['inference_logs_dir']}/inference_logs.csv"

    results_dir = Path(config["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    f1_thresh = config["evaluation"]["f1_threshold"]
    latency_thresh = config["evaluation"]["latency_threshold_ms"]

    # Load logs
    logs_df = load_inference_logs(logs_path)

    # Per-condition metrics
    metrics_df = compute_metrics_per_condition(logs_df)
    metrics_df.to_csv(results_dir / "metrics_per_condition.csv", index=False)

    # SQ3 Decision
    sq3_df = sq3_decision(metrics_df, f1_thresh, latency_thresh)
    sq3_df.to_csv(results_dir / "sq3_recommendation.csv", index=False)

    logger.info("All evaluation results saved to %s", results_dir)

    return {
        "metrics_df": metrics_df,
        "sq3_summary": sq3_df,
        "logs_df": logs_df,
    }
