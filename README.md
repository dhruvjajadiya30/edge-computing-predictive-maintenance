# Edge Computing in Predictive Maintenance

A data-driven comparison of **edge-only**, **cloud-only**, and **hybrid edge-cloud** deployments for predictive maintenance inference under degraded network conditions in industrial IoT environments.

## Research Questions

| # | Sub-Question | Metric |
|---|---|---|
| **SQ1** | Baseline comparison under stable network (0% loss, 100 Mbps) | M1: F1-Score |
| **SQ2** | Impact of network degradation on each strategy | M1 + M2 + M3 |
| **SQ3** | Best strategy meeting F1 ≥ 0.80 AND latency ≤ 200 ms, with lowest comm load | All |

## Metrics

- **M1 — F1-Score**: Binary anomaly detection performance
- **M2 — E2E Alert Latency**: `t_alert − t_capture` in milliseconds
- **M3 — Communication Load**: Bytes transmitted per inference

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quick test (1 condition, 1 run)
python run_experiment.py --quick

# Medium test (12 conditions × 3 strategies × 5 runs = 1200 logs)
python run_experiment.py --config config/medium_test_config.yaml

# Full experiment (20 conditions × 3 strategies × 30 runs = 1,800 logs)
python run_experiment.py --config config/experiment_config.yaml
```

## Project Structure

```
├── config/experiment_config.yaml    # All hyperparameters
├── src/
│   ├── data_preprocessing.py        # Step 01: C-MAPSS data pipeline
│   ├── feature_engineering.py       # Step 02: Rolling stats, PCA, windows
│   ├── model.py                     # Step 03: LSTM Autoencoder
│   ├── deployment/
│   │   ├── edge_strategy.py         # Step 04: Edge-only inference
│   │   ├── cloud_strategy.py        # Step 04: Cloud-only inference
│   │   └── hybrid_strategy.py       # Step 04: Hybrid inference
│   ├── network_simulation.py        # Step 05: Network degradation
│   ├── metrics_logger.py            # Timestamp/byte logging
│   └── evaluation.py                # Step 06: Results analysis
├── run_experiment.py                # Master orchestrator
└── results/                         # Output directory
    ├── metrics_per_condition.csv    # Per-condition averages
    ├── sq3_recommendation.csv       # Strategy recommendations
    └── inference_logs/              # Detailed inference records
```
```

## Dataset

**NASA C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation)  
Source: [data.nasa.gov](https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data)

## Experiment Design

- **Training**: 20 epochs on C-MAPSS FD001 training set
- **Test**: 12 network conditions (3 bandwidth levels × 4 packet loss levels)
- **Deployments**: Edge-only, Cloud-only, Hybrid (PCA@edge)
- **Runs per condition**: 5 runs
- **Total inference logs**: 12 conditions × 3 strategies × 5 runs = **180 logs**
- **Configuration files**:
  - `quick_test_config.yaml`: 1 condition, 1 run, 5 epochs (testing)
  - `medium_test_config.yaml`: 12 conditions, 5 runs, 20 epochs
  - `experiment_config.yaml`: 20 conditions, 30 runs, 50 epochs (full study)

---

## Key Results (Medium Test Run)

### SQ1: Baseline Performance (Stable Network)

| Strategy | F1-Score | Latency (ms) | Bytes |
|---|---|---|---|
| Edge-Only | 0.281 | 104.6 | 100 |
| Cloud-Only | 0.305 | 98.8 | 2,520 |
| Hybrid | 0.281 | 96.5 | 32 |

**Finding**: All strategies produce comparable detection accuracy (0.28–0.31 F1), confirming that deployment choice does NOT affect anomaly detection capability.

### SQ2: Impact of Network Degradation

| Metric | Finding |
|---|---|
| **F1-Score** | Varies by condition (0.18–0.40) but with no consistent pattern. Deployment strategy is independent of network conditions. |
| **Latency** | Cloud-only shows elevated latency at low bandwidth (110–154 ms for 1 Mbps due to 2,520 byte payload). All strategies remain below 200 ms threshold. |
| **Communication** | Cloud-only increases with packet loss (2,520 → 3,750 bytes). Hybrid is 79× more efficient (32–47 bytes vs 100–148 for edge). |

### SQ3: Deployment Recommendation

**Recommended: Hybrid Strategy**

| Criterion | Edge-Only | Cloud-Only | Hybrid |
|---|---|---|---|
| Avg F1 | 0.307 | 0.285 | 0.290 |
| Avg Latency | 96.5 ms | 105.4 ms | 103.2 ms |
| Avg Bytes | 118.7 | 3,013.1 | 38.3 |
| **Communication Efficiency** | — | — | **1.27% of cloud** |

**Rationale**:
1. **Lowest communication overhead** → critical for bandwidth-constrained edge networks
2. **Comparable detection accuracy** across all strategies
3. **Stable performance** across different network conditions
4. **Minimal latency penalty** vs edge-only (103 vs 97 ms)

**Note**: All strategies fail F1 ≥ 0.80 threshold due to limited training (20 epochs). Full model training (50+ epochs) would improve F1 uniformly across all strategies.

---

## Output Files

- `results_analysis.md` — Detailed statistical analysis with per-condition breakdowns
- `results/metrics_per_condition.csv` — Aggregated metrics for each condition
- `results/sq3_recommendation.csv` — Summary recommendation with pass/fail criteria
- `results/inference_logs/` — Individual inference records with timestamps and bytes
- `experiment.log` — Full runtime logs
