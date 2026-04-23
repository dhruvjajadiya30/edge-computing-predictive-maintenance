"""
Edge-Only Deployment Strategy
==============================
Full LSTM inference runs at the edge device.
Only anomaly alert flag is transmitted to the maintenance system.
Transmission payload: ~50-200 bytes (alert flag + metadata).
"""

import time
import numpy as np
from ..metrics_logger import InferenceLog, Timer
from ..network_simulation import NetworkSimulator
import logging

logger = logging.getLogger(__name__)


class EdgeStrategy:
    """
    Edge-only inference pipeline.
    
    Flow:
    1. Sensor window captured → t_capture logged
    2. LSTM inference runs locally (edge device)
    3. Only anomaly alert flag transmitted (~50-200 bytes)
    4. t_alert logged on receipt
    """

    STRATEGY_NAME = "edge-only"

    def __init__(self, model, threshold: float, config: dict):
        self.model = model
        self.threshold = threshold
        self.alert_payload_bytes = config["deployment"]["edge"]["alert_payload_bytes"]
        # Simulated edge inference overhead (Raspberry Pi 4 is ~3-10x slower than desktop)
        self.edge_inference_overhead_factor = 3.0

    def run_inference(
        self,
        X_sample: np.ndarray,
        y_true: int,
        unit_id: int,
        cycle: int,
        network_sim: NetworkSimulator,
        condition_id: str,
        run_number: int,
    ) -> InferenceLog:
        """
        Execute edge-only inference for a single sample.
        
        Args:
            X_sample: Input sequence of shape (1, window_size, n_features)
            y_true: Ground truth anomaly label
            unit_id: Engine unit ID
            cycle: Cycle number
            network_sim: Network simulator for transmission
            condition_id: Network condition ID (e.g., "C01")
            run_number: Experiment run number (1-30)
        
        Returns:
            InferenceLog with all metrics
        """
        # --- Step 1: Capture sensor window ---
        t_capture = time.perf_counter()

        # --- Step 2: Run LSTM inference at edge ---
        if len(X_sample.shape) == 2:
            X_sample = X_sample[np.newaxis, :]  # Add batch dimension

        reconstruction = self.model.predict(X_sample, verbose=0)
        recon_error = float(np.mean(np.square(X_sample - reconstruction)))

        # Simulate edge device overhead (slower than desktop)
        edge_delay_ms = self.edge_inference_overhead_factor * 0.5  # ~1.5ms overhead
        time.sleep(edge_delay_ms / 1000.0)

        prediction = 1 if recon_error > self.threshold else 0

        # --- Step 3: Transmit only alert flag ---
        # Even normal results need a status update transmission
        tx_result = network_sim.transmit_fast(self.alert_payload_bytes)

        # --- Step 4: Alert received ---
        t_alert = time.perf_counter()

        # Total latency = inference time + network delay
        actual_elapsed_ms = (t_alert - t_capture) * 1000
        simulated_latency_ms = actual_elapsed_ms + tx_result["total_delay_ms"]

        return InferenceLog(
            strategy=self.STRATEGY_NAME,
            unit_id=unit_id,
            cycle=cycle,
            prediction=prediction,
            anomaly_label=y_true,
            reconstruction_error=recon_error,
            t_capture=t_capture,
            t_alert=t_alert,
            latency_ms=simulated_latency_ms,
            bytes_transmitted=tx_result["bytes_transmitted"],
            network_condition_id=condition_id,
            packet_loss_pct=network_sim.packet_loss_pct,
            bandwidth_mbps=network_sim.bandwidth_mbps,
            run_number=run_number,
        )
