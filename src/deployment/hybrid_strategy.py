"""
Hybrid Deployment Strategy
============================
PCA feature extraction at edge, compressed vector transmitted to cloud for LSTM inference.
Transmission payload: 8 PCA components × 4 bytes = ~32 bytes.
"""

import time
import numpy as np
from ..metrics_logger import InferenceLog, Timer
from ..network_simulation import NetworkSimulator
import logging

logger = logging.getLogger(__name__)


class HybridStrategy:
    """
    Hybrid edge-cloud inference pipeline.
    
    Flow:
    1. Sensor window captured → t_capture logged
    2. PCA feature extraction at edge (21 sensors → 8 components)
    3. Compressed PCA vector transmitted to cloud (~32 bytes)
    4. LSTM inference on cloud using PCA features
    5. Result returned → t_alert logged
    """

    STRATEGY_NAME = "hybrid"

    def __init__(self, model, threshold: float, config: dict):
        self.model = model
        self.threshold = threshold
        # Payload: 8 PCA components × 4 bytes/float
        hybrid_cfg = config["deployment"]["hybrid"]
        self.payload_bytes = (
            hybrid_cfg["pca_components"] * hybrid_cfg["bytes_per_float"]
        )
        # Edge PCA extraction overhead (lightweight computation)
        self.edge_pca_overhead_ms = 0.3
        # Cloud inference overhead
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
        Execute hybrid inference for a single sample.
        
        Note: PCA has already been applied during feature engineering.
        The X_sample already contains PCA features. Here we simulate
        the separation of PCA extraction (edge) and LSTM inference (cloud).
        
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

        # --- Step 2: PCA extraction at edge (already done, simulate overhead) ---
        time.sleep(self.edge_pca_overhead_ms / 1000.0)

        # --- Step 3: Transmit compressed PCA vector to cloud ---
        # Payload is much smaller than cloud-only: 32 bytes vs 2,520 bytes
        tx_result = network_sim.transmit_fast(self.payload_bytes)

        # --- Step 4: Cloud LSTM inference ---
        if len(X_sample.shape) == 2:
            X_sample = X_sample[np.newaxis, :]

        reconstruction = self.model.predict(X_sample, verbose=0)
        recon_error = float(np.mean(np.square(X_sample - reconstruction)))
        prediction = 1 if recon_error > self.threshold else 0

        # --- Step 5: Return result (~50 bytes response) ---
        response_tx = network_sim.transmit_fast(50)

        t_alert = time.perf_counter()

        # Total latency = PCA + upload + inference + download
        actual_elapsed_ms = (t_alert - t_capture) * 1000
        simulated_latency_ms = (
            actual_elapsed_ms
            + self.edge_pca_overhead_ms
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
