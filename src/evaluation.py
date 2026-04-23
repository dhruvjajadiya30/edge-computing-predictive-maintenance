"""
Step 06 — Evaluation & Statistical Testing
=============================================
Compute F1-Score, E2E Alert Latency, Communication Load per strategy per condition.
Run one-way ANOVA + Tukey HSD post-hoc tests.
Generate SQ3 deployment recommendation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from scipy import stats
import warnings
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


def run_anova(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str = "strategy",
    alpha: float = 0.05,
) -> dict:
    """
    Run one-way ANOVA across strategies for a given metric.
    
    Returns dict with F-statistic, p-value, and significance.
    """
    groups = [group[metric_col].values for _, group in df.groupby(group_col)]

    # Filter out groups with insufficient data
    groups = [g for g in groups if len(g) > 1]

    if len(groups) < 2:
        logger.warning("Not enough groups for ANOVA on %s", metric_col)
        return {"f_statistic": np.nan, "p_value": np.nan, "significant": False}

    f_stat, p_value = stats.f_oneway(*groups)

    result = {
        "metric": metric_col,
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "alpha": alpha,
    }

    logger.info(
        "ANOVA (%s): F=%.4f, p=%.6f → %s",
        metric_col, f_stat, p_value,
        "SIGNIFICANT" if result["significant"] else "not significant",
    )
    return result


def run_tukey_hsd(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str = "strategy",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run Tukey HSD post-hoc test for pairwise comparisons.
    
    Returns DataFrame with pairwise comparison results.
    """
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        result = pairwise_tukeyhsd(
            endog=df[metric_col].values,
            groups=df[group_col].values,
            alpha=alpha,
        )

        # Parse results into DataFrame
        tukey_df = pd.DataFrame(
            data=result._results_table.data[1:],
            columns=result._results_table.data[0],
        )

        logger.info("Tukey HSD (%s):\n%s", metric_col, result.summary())
        return tukey_df

    except ImportError:
        logger.warning("statsmodels not available for Tukey HSD. Falling back to pairwise t-tests.")
        return _pairwise_t_tests(df, metric_col, group_col, alpha)


def _pairwise_t_tests(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str,
    alpha: float,
) -> pd.DataFrame:
    """Fallback: pairwise independent t-tests with Bonferroni correction."""
    groups = df[group_col].unique()
    results = []
    n_comparisons = len(groups) * (len(groups) - 1) // 2
    bonferroni_alpha = alpha / max(1, n_comparisons)

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1 = df[df[group_col] == groups[i]][metric_col].values
            g2 = df[df[group_col] == groups[j]][metric_col].values

            t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
            results.append({
                "group1": groups[i],
                "group2": groups[j],
                "meandiff": np.mean(g2) - np.mean(g1),
                "t_statistic": t_stat,
                "p-adj": p_val * n_comparisons,  # Bonferroni
                "reject": p_val < bonferroni_alpha,
            })

    return pd.DataFrame(results)


def compute_effect_size(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str = "strategy",
) -> dict:
    """
    Compute eta-squared (η²) effect size for ANOVA.
    η² = SS_between / SS_total
    """
    grand_mean = df[metric_col].mean()
    groups = df.groupby(group_col)[metric_col]

    ss_between = sum(
        len(g) * (g.mean() - grand_mean) ** 2
        for _, g in groups
    )
    ss_total = sum((df[metric_col] - grand_mean) ** 2)

    eta_squared = ss_between / ss_total if ss_total > 0 else 0

    # Interpretation
    if eta_squared < 0.01:
        interpretation = "negligible"
    elif eta_squared < 0.06:
        interpretation = "small"
    elif eta_squared < 0.14:
        interpretation = "medium"
    else:
        interpretation = "large"

    logger.info(
        "Effect size (η²) for %s: %.4f (%s)",
        metric_col, eta_squared, interpretation,
    )

    return {
        "metric": metric_col,
        "eta_squared": float(eta_squared),
        "interpretation": interpretation,
    }


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
    2. Compute per-condition metrics
    3. Run ANOVA for each metric
    4. Run Tukey HSD post-hoc
    5. Compute effect sizes
    6. Run SQ3 decision logic
    7. Save all results
    
    Returns dict with all results.
    """
    if logs_path is None:
        logs_path = f"{config['evaluation']['inference_logs_dir']}/inference_logs.csv"

    results_dir = Path(config["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    alpha = config["evaluation"]["significance_level"]
    f1_thresh = config["evaluation"]["f1_threshold"]
    latency_thresh = config["evaluation"]["latency_threshold_ms"]

    # Load logs
    logs_df = load_inference_logs(logs_path)

    # Per-condition metrics
    metrics_df = compute_metrics_per_condition(logs_df)
    metrics_df.to_csv(results_dir / "metrics_per_condition.csv", index=False)

    # ANOVA for each metric
    anova_results = {}
    for metric in ["f1_score", "latency_mean_ms", "bytes_mean"]:
        # Need per-run data for ANOVA, not per-condition aggregates
        # Create per-run metric data
        if metric == "f1_score":
            # F1 is already per-condition, use it directly
            anova_results[metric] = run_anova(metrics_df, metric, alpha=alpha)
        elif metric == "latency_mean_ms":
            anova_results[metric] = run_anova(metrics_df, metric, alpha=alpha)
        else:
            anova_results[metric] = run_anova(metrics_df, metric, alpha=alpha)

    # Tukey HSD
    tukey_results = {}
    for metric in ["f1_score", "latency_mean_ms", "bytes_mean"]:
        tukey_results[metric] = run_tukey_hsd(metrics_df, metric, alpha=alpha)
        tukey_results[metric].to_csv(
            results_dir / f"tukey_hsd_{metric}.csv", index=False
        )

    # Effect sizes
    effect_sizes = {}
    for metric in ["f1_score", "latency_mean_ms", "bytes_mean"]:
        effect_sizes[metric] = compute_effect_size(metrics_df, metric)

    # SQ3 Decision
    sq3_df = sq3_decision(metrics_df, f1_thresh, latency_thresh)
    sq3_df.to_csv(results_dir / "sq3_recommendation.csv", index=False)

    # Save ANOVA summary
    anova_df = pd.DataFrame([
        {**v, "effect_size_eta2": effect_sizes[k]["eta_squared"],
         "effect_interpretation": effect_sizes[k]["interpretation"]}
        for k, v in anova_results.items()
    ])
    anova_df.to_csv(results_dir / "anova_results.csv", index=False)

    logger.info("All evaluation results saved to %s", results_dir)

    return {
        "metrics_df": metrics_df,
        "anova_results": anova_results,
        "tukey_results": tukey_results,
        "effect_sizes": effect_sizes,
        "sq3_summary": sq3_df,
        "logs_df": logs_df,
    }
