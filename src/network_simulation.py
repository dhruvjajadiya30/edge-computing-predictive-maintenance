"""
Step 05 — Network Degradation Simulation
==========================================
Python-based network degradation simulator.
Simulates packet loss and bandwidth limitations without requiring Linux tc netem.
"""

import time
import random
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Condition ID mapping: (packet_loss_pct, bandwidth_mbps) → "C01".."C20"
def generate_condition_matrix(
    packet_loss_levels: list,
    bandwidth_levels: list,
) -> list:
    """
    Generate the full experiment condition matrix.
    Returns list of dicts with condition_id, packet_loss_pct, bandwidth_mbps.
    """
    conditions = []
    idx = 1
    for pl in packet_loss_levels:
        for bw in bandwidth_levels:
            conditions.append({
                "condition_id": f"C{idx:02d}",
                "packet_loss_pct": pl,
                "bandwidth_mbps": bw,
            })
            idx += 1
    return conditions


class NetworkSimulator:
    """
    Simulates network effects for data transmission:
    - Bandwidth-induced transmission delay
    - Packet loss (probabilistic data corruption / retry)
    - Jitter (random latency variation)
    
    This is a deterministic, reproducible simulation that doesn't
    require Linux tc netem.
    """

    def __init__(
        self,
        packet_loss_pct: float = 0.0,
        bandwidth_mbps: float = 100.0,
        base_latency_ms: float = 1.0,
        jitter_ms: float = 0.5,
        seed: int = None,
    ):
        self.packet_loss_pct = packet_loss_pct
        self.bandwidth_mbps = bandwidth_mbps
        self.base_latency_ms = base_latency_ms
        self.jitter_ms = jitter_ms
        self.rng = random.Random(seed)

    def compute_transmission_delay_ms(self, payload_bytes: int) -> float:
        """
        Compute the transmission delay for a given payload size.
        
        delay = base_latency + (payload_bytes * 8) / (bandwidth_bps) * 1000 + jitter
        """
        # Convert bandwidth to bits per second
        bandwidth_bps = self.bandwidth_mbps * 1_000_000

        # Transmission time in ms
        transmission_ms = (payload_bytes * 8) / bandwidth_bps * 1000

        # Jitter: uniform random in [-jitter_ms, +jitter_ms]
        jitter = self.rng.uniform(-self.jitter_ms, self.jitter_ms)

        total_delay = self.base_latency_ms + transmission_ms + max(0, jitter)
        return max(0.01, total_delay)  # Minimum 0.01ms

    def simulate_packet_loss(self) -> bool:
        """
        Simulate whether a packet is lost.
        Returns True if packet is lost (needs retry).
        """
        return self.rng.random() * 100 < self.packet_loss_pct

    def transmit(self, payload_bytes: int, max_retries: int = 5) -> dict:
        """
        Simulate transmitting a payload through the degraded network.
        
        Handles:
        - Transmission delay based on bandwidth
        - Packet loss with retries
        - Cumulative delay from retries
        
        Returns dict with:
        - total_delay_ms: total simulated delay
        - bytes_transmitted: total bytes (including retransmissions)
        - retries: number of retransmissions needed
        - success: whether transmission succeeded
        """
        total_delay_ms = 0.0
        total_bytes = 0
        retries = 0
        success = False

        for attempt in range(max_retries + 1):
            # Compute delay for this attempt
            delay = self.compute_transmission_delay_ms(payload_bytes)
            total_delay_ms += delay
            total_bytes += payload_bytes

            # Check for packet loss
            if not self.simulate_packet_loss():
                success = True
                break
            else:
                retries += 1
                # Add retry backoff delay (exponential)
                backoff_ms = min(50, 2 ** retries * 1.0)
                total_delay_ms += backoff_ms

        # Actually sleep to simulate real time passing
        time.sleep(total_delay_ms / 1000.0)

        return {
            "total_delay_ms": total_delay_ms,
            "bytes_transmitted": total_bytes,
            "retries": retries,
            "success": success,
        }

    def transmit_fast(self, payload_bytes: int, max_retries: int = 5) -> dict:
        """
        Same as transmit() but WITHOUT actual sleep — computes delay analytically.
        Use this for faster experiment runs; the delay is recorded but not enacted.
        """
        total_delay_ms = 0.0
        total_bytes = 0
        retries = 0
        success = False

        for attempt in range(max_retries + 1):
            delay = self.compute_transmission_delay_ms(payload_bytes)
            total_delay_ms += delay
            total_bytes += payload_bytes

            if not self.simulate_packet_loss():
                success = True
                break
            else:
                retries += 1
                backoff_ms = min(50, 2 ** retries * 1.0)
                total_delay_ms += backoff_ms

        return {
            "total_delay_ms": total_delay_ms,
            "bytes_transmitted": total_bytes,
            "retries": retries,
            "success": success,
        }

    def __repr__(self):
        return (
            f"NetworkSimulator(loss={self.packet_loss_pct}%, "
            f"bw={self.bandwidth_mbps}Mbps, "
            f"base_lat={self.base_latency_ms}ms)"
        )
