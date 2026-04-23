"""
Metrics Logger
===============
Shared logging utility for all deployment strategies.
Logs timestamps, byte counts, predictions, and network conditions
for each inference cycle.
"""

import time
import csv
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class InferenceLog:
    """Single inference log entry."""
    strategy: str                    # "edge-only", "cloud-only", "hybrid"
    unit_id: int                     # Engine unit ID
    cycle: int                       # Cycle number
    prediction: int                  # 0=normal, 1=anomaly
    anomaly_label: int               # Ground truth
    reconstruction_error: float      # MSE reconstruction error
    t_capture: float                 # Timestamp: sensor window captured (perf_counter)
    t_alert: float                   # Timestamp: anomaly alert delivered (perf_counter)
    latency_ms: float                # End-to-end alert latency in milliseconds
    bytes_transmitted: int           # Bytes sent over network
    network_condition_id: str        # e.g., "C01"
    packet_loss_pct: float           # Current packet loss percentage
    bandwidth_mbps: float            # Current bandwidth in Mbps
    run_number: int                  # Run index (1-30)


class MetricsLogger:
    """
    Collects and persists inference logs for all experiments.
    Thread-safe CSV writer with automatic file management.
    """

    def __init__(self, output_dir: str = "results/inference_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs: list = []
        self._csv_path = self.output_dir / "inference_logs.csv"
        self._header_written = False

        # Write header if file doesn't exist
        if not self._csv_path.exists():
            self._write_header()

    def _write_header(self):
        """Write CSV header."""
        fieldnames = [
            "strategy", "unit_id", "cycle", "prediction", "anomaly_label",
            "reconstruction_error", "t_capture", "t_alert", "latency_ms",
            "bytes_transmitted", "network_condition_id",
            "packet_loss_pct", "bandwidth_mbps", "run_number",
        ]
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        self._header_written = True

    def log(self, entry: InferenceLog):
        """Log a single inference result."""
        self.logs.append(entry)

        # Append to CSV immediately
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(entry).keys()))
            writer.writerow(asdict(entry))

    def get_logs_count(self) -> int:
        """Return total number of logs collected."""
        return len(self.logs)

    def reset(self):
        """Clear in-memory logs (CSV file is preserved)."""
        self.logs.clear()

    def reset_file(self):
        """Clear both in-memory logs and CSV file."""
        self.logs.clear()
        self._write_header()


class Timer:
    """Context manager for precise timing of inference stages."""

    def __init__(self):
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return (self.end_time - self.start_time) * 1000

    @property
    def elapsed_s(self) -> float:
        """Elapsed time in seconds."""
        return self.end_time - self.start_time
