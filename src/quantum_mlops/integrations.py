"""MLOps integrations for quantum machine learning workflows."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import mlflow
    import mlflow.tracking
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .core import QuantumModel, QuantumMetrics
from .exceptions import QuantumMLOpsException

logger = logging.getLogger(__name__)


class QuantumMLflow:
    """MLflow integration for quantum machine learning experiments."""
    
    def __init__(self, experiment_name: str = "quantum_ml_experiment") -> None:
        """Initialize QuantumMLflow integration.
        
        Args:
            experiment_name: Name of the MLflow experiment
        """
        if not MLFLOW_AVAILABLE:
            raise QuantumMLOpsException("MLflow not available. Install with: pip install mlflow")
        
        self.experiment_name = experiment_name
        self._setup_experiment()
    
    def _setup_experiment(self) -> None:
        """Setup MLflow experiment."""
        try:
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.warning(f"Failed to set MLflow experiment: {e}")
    
    @staticmethod
    def autolog() -> None:
        """Enable automatic logging of quantum ML metrics."""
        if not MLFLOW_AVAILABLE:
            return
        
        mlflow.autolog()
        
        # Custom quantum metric logging
        original_log_metric = mlflow.log_metric
        
        def quantum_log_metric(key: str, value: float, step: Optional[int] = None) -> None:
            """Enhanced metric logging with quantum-specific formatting."""
            # Add quantum prefix for quantum-specific metrics
            quantum_metrics = [
                'fidelity', 'entanglement_entropy', 'gradient_variance',
                'circuit_depth', 'coherence_time', 'gate_error_rate'
            ]
            
            if any(qm in key.lower() for qm in quantum_metrics):
                key = f"quantum_{key}"
            
            original_log_metric(key, value, step)
        
        mlflow.log_metric = quantum_log_metric
    
    def log_quantum_model(self, model: QuantumModel, model_name: str = "quantum_model") -> None:
        """Log quantum model artifacts to MLflow.
        
        Args:
            model: Quantum model to log
            model_name: Name for the model artifact
        """
        if not MLFLOW_AVAILABLE:
            return
        
        # Save model to temporary file
        model_path = Path(f"/tmp/{model_name}.json")
        model.save_model(str(model_path))
        
        # Log model artifact
        mlflow.log_artifact(str(model_path), "models")
        
        # Log model metadata
        mlflow.log_params({
            "n_qubits": model.n_qubits,
            "circuit_depth": model.circuit_depth,
            "parameter_count": len(model.parameters) if model.parameters is not None else 0
        })
        
        # Log quantum state vector if available
        if hasattr(model, 'state_vector'):
            state_vector_path = Path(f"/tmp/{model_name}_state.npy")
            np.save(state_vector_path, model.state_vector)
            mlflow.log_artifact(str(state_vector_path), "quantum_states")
    
    def log_quantum_metrics(self, metrics: QuantumMetrics, step: Optional[int] = None) -> None:
        """Log quantum-specific metrics to MLflow.
        
        Args:
            metrics: Quantum metrics to log
            step: Training step for metric
        """
        if not MLFLOW_AVAILABLE:
            return
        
        metrics_dict = metrics.to_dict()
        
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"quantum_{key}", value, step)
            elif isinstance(value, dict):
                # Log nested metrics (e.g., noise analysis)
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, dict):
                        for sub_key, sub_value in nested_value.items():
                            if isinstance(sub_value, (int, float)):
                                mlflow.log_metric(f"quantum_{key}_{nested_key}_{sub_key}", sub_value, step)
    
    def log_quantum_circuit(self, circuit_description: Dict[str, Any], name: str = "quantum_circuit") -> None:
        """Log quantum circuit description to MLflow.
        
        Args:
            circuit_description: Circuit description dictionary
            name: Name for the circuit artifact
        """
        if not MLFLOW_AVAILABLE:
            return
        
        circuit_path = Path(f"/tmp/{name}.json")
        with open(circuit_path, 'w') as f:
            json.dump(circuit_description, f, indent=2)
        
        mlflow.log_artifact(str(circuit_path), "circuits")


class QuantumWandB:
    """Weights & Biases integration for quantum machine learning."""
    
    def __init__(self, project: str = "quantum-ml", config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize QuantumWandB integration.
        
        Args:
            project: W&B project name
            config: Initial configuration dictionary
        """
        if not WANDB_AVAILABLE:
            raise QuantumMLOpsException("Weights & Biases not available. Install with: pip install wandb")
        
        self.project = project
        self.config = config or {}
        self._initialized = False
    
    def init(self, **kwargs: Any) -> None:
        """Initialize W&B run with quantum-specific configuration.
        
        Args:
            **kwargs: Additional arguments for wandb.init()
        """
        if not WANDB_AVAILABLE:
            return
        
        # Merge quantum-specific config
        config = {
            **self.config,
            **kwargs.get('config', {})
        }
        
        wandb.init(
            project=self.project,
            config=config,
            **{k: v for k, v in kwargs.items() if k != 'config'}
        )
        
        self._initialized = True
    
    @staticmethod
    def watch(model: QuantumModel, log_freq: int = 100, log_graph: bool = True) -> None:
        """Watch quantum model for automatic logging.
        
        Args:
            model: Quantum model to watch
            log_freq: Frequency of automatic logging
            log_graph: Whether to log quantum circuit graph
        """
        if not WANDB_AVAILABLE or not hasattr(wandb, 'run') or wandb.run is None:
            return
        
        # Log model architecture
        wandb.config.update({
            "quantum_model": {
                "n_qubits": model.n_qubits,
                "circuit_depth": model.circuit_depth,
                "parameter_count": len(model.parameters) if model.parameters is not None else 0
            }
        })
        
        # Log quantum state visualization
        if log_graph and hasattr(model, 'state_vector'):
            state_vector = model.state_vector
            
            # Create quantum state visualization
            wandb.log({
                "quantum_state_magnitude": wandb.Histogram(np.abs(state_vector)),
                "quantum_state_phase": wandb.Histogram(np.angle(state_vector))
            })
    
    def log_quantum_metrics(self, metrics: QuantumMetrics, step: Optional[int] = None) -> None:
        """Log quantum metrics to W&B.
        
        Args:
            metrics: Quantum metrics to log
            step: Training step
        """
        if not WANDB_AVAILABLE or not self._initialized:
            return
        
        log_dict = {}
        metrics_dict = metrics.to_dict()
        
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                log_dict[f"quantum_{key}"] = value
            elif isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, dict):
                        for sub_key, sub_value in nested_value.items():
                            if isinstance(sub_value, (int, float)):
                                log_dict[f"quantum_{key}_{nested_key}_{sub_key}"] = sub_value
        
        if step is not None:
            log_dict["step"] = step
        
        wandb.log(log_dict)
    
    def log_quantum_circuit_visualization(self, circuit_description: Dict[str, Any]) -> None:
        """Log quantum circuit visualization to W&B.
        
        Args:
            circuit_description: Circuit description dictionary
        """
        if not WANDB_AVAILABLE or not self._initialized:
            return
        
        # Extract circuit statistics
        gates = circuit_description.get('gates', [])
        gate_counts = {}
        
        for gate in gates:
            gate_type = gate.get('type', 'unknown')
            gate_counts[gate_type] = gate_counts.get(gate_type, 0) + 1
        
        # Log gate distribution
        wandb.log({
            "quantum_gate_distribution": wandb.Histogram(list(gate_counts.values()), bins=list(gate_counts.keys())),
            "quantum_circuit_depth": len(gates),
            "quantum_gate_types": len(gate_counts)
        })


class QuantumTensorboard:
    """TensorBoard integration for quantum machine learning visualization."""
    
    def __init__(self, log_dir: str = "logs/quantum_ml") -> None:
        """Initialize QuantumTensorboard integration.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            self.available = True
        except ImportError:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            self.available = False
            self.writer = None
    
    def log_quantum_metrics(self, metrics: QuantumMetrics, step: int) -> None:
        """Log quantum metrics to TensorBoard.
        
        Args:
            metrics: Quantum metrics to log
            step: Training step
        """
        if not self.available or not self.writer:
            return
        
        metrics_dict = metrics.to_dict()
        
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"Quantum/{key}", value, step)
            elif isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, dict):
                        for sub_key, sub_value in nested_value.items():
                            if isinstance(sub_value, (int, float)):
                                self.writer.add_scalar(f"Quantum/{key}/{nested_key}_{sub_key}", sub_value, step)
    
    def log_quantum_state_histogram(self, state_vector: np.ndarray, step: int) -> None:
        """Log quantum state vector histogram to TensorBoard.
        
        Args:
            state_vector: Quantum state vector
            step: Training step
        """
        if not self.available or not self.writer:
            return
        
        # Log state magnitude and phase distributions
        magnitudes = np.abs(state_vector)
        phases = np.angle(state_vector)
        
        self.writer.add_histogram("Quantum/state_magnitudes", magnitudes, step)
        self.writer.add_histogram("Quantum/state_phases", phases, step)
    
    def close(self) -> None:
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()


def get_available_integrations() -> Dict[str, bool]:
    """Get availability status of MLOps integrations.
    
    Returns:
        Dictionary mapping integration names to availability status
    """
    return {
        "mlflow": MLFLOW_AVAILABLE,
        "wandb": WANDB_AVAILABLE,
        "tensorboard": True  # Usually available with PyTorch
    }


def setup_experiment_tracking(
    backend: str = "mlflow",
    experiment_name: str = "quantum_ml_experiment",
    **kwargs: Any
) -> Union[QuantumMLflow, QuantumWandB, QuantumTensorboard]:
    """Setup experiment tracking backend.
    
    Args:
        backend: Tracking backend ("mlflow", "wandb", "tensorboard")
        experiment_name: Name of the experiment
        **kwargs: Additional backend-specific arguments
    
    Returns:
        Configured tracking backend instance
    
    Raises:
        QuantumMLOpsException: If backend is not available
    """
    if backend == "mlflow":
        if not MLFLOW_AVAILABLE:
            raise QuantumMLOpsException("MLflow not available")
        return QuantumMLflow(experiment_name, **kwargs)
    
    elif backend == "wandb":
        if not WANDB_AVAILABLE:
            raise QuantumMLOpsException("Weights & Biases not available")
        tracker = QuantumWandB(project=experiment_name, **kwargs)
        tracker.init()
        return tracker
    
    elif backend == "tensorboard":
        return QuantumTensorboard(log_dir=kwargs.get("log_dir", f"logs/{experiment_name}"))
    
    else:
        raise QuantumMLOpsException(f"Unknown tracking backend: {backend}")