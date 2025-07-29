"""Quantum ML monitoring and metrics tracking."""

import json
import time
from typing import Any, Dict, Optional, Union

import numpy as np


class QuantumMonitor:
    """Monitor quantum ML experiments and metrics."""
    
    def __init__(
        self,
        experiment_name: str = "quantum_experiment",
        tracking_uri: Optional[str] = None,
    ) -> None:
        """Initialize quantum monitor.
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: URI for tracking server (e.g., MLflow)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.metrics: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
        
    def start_run(self) -> "QuantumMonitor":
        """Start monitoring run."""
        self.start_time = time.time()
        self.metrics = {}
        return self
        
    def __enter__(self) -> "QuantumMonitor":
        """Context manager entry."""
        return self.start_run()
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.end_run()
        
    def end_run(self) -> None:
        """End monitoring run."""
        if self.start_time:
            duration = time.time() - self.start_time
            self.log_metrics({"run_duration": duration})
            
    def log_metrics(self, metrics: Dict[str, Union[float, int]]) -> None:
        """Log quantum ML metrics.
        
        Args:
            metrics: Dictionary of metric names and values
        """
        timestamp = time.time()
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append({
                "value": value,
                "timestamp": timestamp
            })
            
    def log_quantum_state(self, state_vector: np.ndarray, step: int) -> None:
        """Log quantum state information.
        
        Args:
            state_vector: Quantum state vector
            step: Training step
        """
        # Calculate quantum state properties
        fidelity = float(np.abs(np.vdot(state_vector, state_vector)))
        entanglement = self._calculate_entanglement(state_vector)
        
        self.log_metrics({
            f"fidelity_step_{step}": fidelity,
            f"entanglement_step_{step}": entanglement,
        })
        
    def _calculate_entanglement(self, state_vector: np.ndarray) -> float:
        """Calculate entanglement measure for quantum state.
        
        Args:
            state_vector: Quantum state vector
            
        Returns:
            Entanglement entropy measure
        """
        # Simplified entanglement calculation
        # In practice, would use proper von Neumann entropy
        n_qubits = int(np.log2(len(state_vector)))
        if n_qubits < 2:
            return 0.0
            
        # Calculate reduced density matrix and entropy
        probabilities = np.abs(state_vector) ** 2
        probabilities = probabilities[probabilities > 1e-10]  # Remove zeros
        
        if len(probabilities) == 0:
            return 0.0
            
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return float(entropy / n_qubits)  # Normalized entropy
        
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of logged metrics.
        
        Returns:
            Dictionary with metric summaries
        """
        summary = {}
        for metric_name, values in self.metrics.items():
            if values:
                latest_values = [v["value"] for v in values]
                summary[metric_name] = {
                    "latest": latest_values[-1],
                    "mean": np.mean(latest_values),
                    "std": np.std(latest_values),
                    "min": np.min(latest_values),
                    "max": np.max(latest_values),
                    "count": len(latest_values)
                }
        return summary
        
    def export_metrics(self, filepath: str) -> None:
        """Export metrics to file.
        
        Args:
            filepath: Path to export metrics
        """
        summary = self.get_metrics_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)