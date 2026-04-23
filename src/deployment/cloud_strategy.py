"""
Cloud-Only Deployment Strategy
================================
Raw sensor window is transmitted to cloud for inference.
Transmission payload: 21 sensors × 30 cycles × 4 bytes = ~2,520 bytes.
"""

import time
import numpy as np
from ..metrics_logger import InferenceLog, Timer
from ..network_simulation import NetworkSimulator
import logging

logger = logging.getLogger(__name__)


class CloudStrategy:
    """
    Cloud-only inference pipeline.
    
    Flow:
    1. Sensor window captured → t_capture logged
    2. Full raw sensor window transmitted to cloud (~2,520 bytes)
    3. LSTM inference runs on cloud server
    4. Anomaly result returned → t_alert logged
    """

    STRATEGY_NAME = "cloud-only"

    def __init__(self, model, threshold: float, config: dict):
        self.model = model
        self.threshold = threshold
        # Payload: 21 sensors × 30 cycles × 4 bytes/float
        cloud_cfg = config["deployment"]["cloud"]
        self.payload_bytes = (
            cloud_cfg["sensors"] * cloud_cfg["window_size"] * cloud_cfg["bytes_per_float"]
        )
        # Cloud inference is fast (powerful hardware)
        self.cloud_inference_overhead_ms = 0.2

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
        Execute cloud-only inference for a single sample.
        
        Args:
            X_sample: Input sequence (PCA features) of shape (1, window_size, n_features)
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

        # --- Step 2: Transmit full raw sensor window to cloud ---
        # The raw window is larger than PCA features
        tx_result = network_sim.transmit_fast(self.payload_bytes)

        # --- Step 3: Cloud inference ---
        if len(X_sample.shape) == 2:
            X_sample = X_sample[np.newaxis, :]

        reconstruction = self.model.predict(X_sample, verbose=0)
        recon_error = float(np.mean(np.square(X_sample - reconstruction)))
        prediction = 1 if recon_error > self.threshold else 0

        # --- Step 4: Return result (small response, ~50 bytes) ---
        response_tx = network_sim.transmit_fast(50)

        t_alert = time.perf_counter()

        # Total latency = upload + inference + download + cloud overhead
        actual_elapsed_ms = (t_alert - t_capture) * 1000
        simulated_latency_ms = (
            actual_elapsed_ms
            + tx_result["total_delay_ms"]
            + self.cloud_inference_overhead_ms
            + response_tx["total_delay_ms"]
        )

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
