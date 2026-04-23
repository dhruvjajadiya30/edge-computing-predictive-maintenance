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

# Full experiment (20 conditions × 3 strategies × 30 runs = 1,800 logs)
python run_experiment.py
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
│   └── evaluation.py                # Step 06: ANOVA + Tukey HSD
├── run_experiment.py                # Master orchestrator
└── results/                         # Output directory
```

## Dataset

**NASA C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation)  
Source: [data.nasa.gov](https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data)

## Experiment Matrix

20 network conditions (5 packet loss × 4 bandwidth) × 3 strategies × 30 runs = **1,800 inference logs**
