"""Quantum ML monitoring and metrics tracking.

This module provides comprehensive monitoring capabilities for quantum machine learning
workflows, including real-time metrics tracking, experiment logging, visualization,
and alerting systems specifically designed for quantum computing environments.
"""

import json
import logging
import os
import threading
import time
import warnings
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Optional dependencies with graceful fallbacks
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Visualization features will be limited.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive visualizations will be limited.")

try:
    import mlflow
    import mlflow.tracking
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not available. Experiment tracking will be limited.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("Weights & Biases not available. Cloud tracking will be limited.")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import entropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Advanced statistical analysis will be limited.")

# Set up logging
logger = logging.getLogger(__name__)


class QuantumMetricsCalculator:
    """Calculate quantum-specific metrics and properties."""
    
    @staticmethod
    def calculate_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
        """Calculate quantum state fidelity between two states.
        
        Args:
            state1: First quantum state vector
            state2: Second quantum state vector
            
        Returns:
            Fidelity measure (0 to 1)
        """
        try:
            # Normalize states
            state1_norm = state1 / np.linalg.norm(state1)
            state2_norm = state2 / np.linalg.norm(state2)
            
            # Calculate fidelity as |<psi1|psi2>|^2
            overlap = np.abs(np.vdot(state1_norm, state2_norm))
            return float(overlap ** 2)
        except Exception as e:
            logger.error(f"Error calculating fidelity: {e}")
            return 0.0
    
    @staticmethod
    def calculate_entanglement_entropy(state_vector: np.ndarray, 
                                     subsystem_qubits: Optional[List[int]] = None) -> float:
        """Calculate von Neumann entanglement entropy.
        
        Args:
            state_vector: Quantum state vector
            subsystem_qubits: Qubits in subsystem (defaults to first half)
            
        Returns:
            Entanglement entropy
        """
        try:
            n_qubits = int(np.log2(len(state_vector)))
            if n_qubits < 2:
                return 0.0
            
            # Default to bipartition
            if subsystem_qubits is None:
                subsystem_qubits = list(range(n_qubits // 2))
            
            # Reshape state vector to tensor
            state_tensor = state_vector.reshape([2] * n_qubits)
            
            # Trace out complement subsystem
            complement_qubits = [i for i in range(n_qubits) if i not in subsystem_qubits]
            
            # Calculate reduced density matrix (simplified approach)
            rho_reduced = np.abs(state_vector) ** 2
            rho_reduced = rho_reduced.reshape([2] * n_qubits)
            
            # Sum over complement qubits (simplified)
            for qubit in reversed(complement_qubits):
                rho_reduced = np.sum(rho_reduced, axis=qubit)
            
            # Flatten and normalize
            rho_flat = rho_reduced.flatten()
            rho_flat = rho_flat[rho_flat > 1e-10]  # Remove numerical zeros
            
            if len(rho_flat) == 0:
                return 0.0
            
            rho_flat = rho_flat / np.sum(rho_flat)  # Normalize
            
            # Calculate von Neumann entropy
            entropy = -np.sum(rho_flat * np.log2(rho_flat + 1e-10))
            return float(entropy)
        
        except Exception as e:
            logger.error(f"Error calculating entanglement entropy: {e}")
            return 0.0
    
    @staticmethod
    def calculate_circuit_depth(gate_sequence: List[Dict[str, Any]]) -> int:
        """Calculate quantum circuit depth.
        
        Args:
            gate_sequence: List of gate operations
            
        Returns:
            Circuit depth (number of parallel layers)
        """
        if not gate_sequence:
            return 0
        
        try:
            # Track when each qubit is last used
            qubit_times = defaultdict(int)
            max_time = 0
            
            for gate in gate_sequence:
                gate_type = gate.get('type', 'unknown')
                
                if gate_type in ['ry', 'rz', 'rx', 'h', 'x', 'y', 'z']:
                    # Single qubit gate
                    qubit = gate.get('qubit', 0)
                    qubit_times[qubit] += 1
                    max_time = max(max_time, qubit_times[qubit])
                    
                elif gate_type in ['cnot', 'cx', 'cz', 'swap']:
                    # Two qubit gate
                    control = gate.get('control', 0)
                    target = gate.get('target', 1)
                    
                    gate_time = max(qubit_times[control], qubit_times[target]) + 1
                    qubit_times[control] = gate_time
                    qubit_times[target] = gate_time
                    max_time = max(max_time, gate_time)
            
            return max_time
        
        except Exception as e:
            logger.error(f"Error calculating circuit depth: {e}")
            return 0
    
    @staticmethod
    def calculate_gradient_variance(gradients_history: List[np.ndarray]) -> float:
        """Calculate variance in gradient norms over training.
        
        Args:
            gradients_history: List of gradient arrays
            
        Returns:
            Variance of gradient norms
        """
        if len(gradients_history) < 2:
            return 0.0
        
        try:
            gradient_norms = [np.linalg.norm(grad) for grad in gradients_history]
            return float(np.var(gradient_norms))
        except Exception as e:
            logger.error(f"Error calculating gradient variance: {e}")
            return 0.0
    
    @staticmethod
    def calculate_quantum_volume(n_qubits: int, circuit_depth: int, 
                               gate_fidelity: float = 0.99) -> float:
        """Calculate quantum volume metric.
        
        Args:
            n_qubits: Number of qubits
            circuit_depth: Circuit depth
            gate_fidelity: Average gate fidelity
            
        Returns:
            Quantum volume estimate
        """
        try:
            # Simplified quantum volume calculation
            # QV = min(n_qubits, circuit_depth)^2 * gate_fidelity^(n_gates)
            effective_size = min(n_qubits, circuit_depth)
            n_gates = n_qubits * circuit_depth  # Rough estimate
            
            qv = (effective_size ** 2) * (gate_fidelity ** n_gates)
            return float(qv)
        except Exception as e:
            logger.error(f"Error calculating quantum volume: {e}")
            return 0.0


class QuantumAlertSystem:
    """Alert system for quantum hardware issues and performance problems."""
    
    def __init__(self, alert_thresholds: Optional[Dict[str, float]] = None):
        """Initialize alert system.
        
        Args:
            alert_thresholds: Custom alert thresholds
        """
        self.alert_thresholds = alert_thresholds or {
            'fidelity_drop': 0.05,
            'error_rate_increase': 0.02,
            'queue_time_limit': 300.0,  # 5 minutes
            'cost_spike': 2.0,  # 2x normal cost
            'gradient_explosion': 10.0,
            'entanglement_loss': 0.1
        }
        self.alert_history: List[Dict[str, Any]] = []
        self.alert_callbacks: List[Callable] = []
    
    def register_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Register alert callback function.
        
        Args:
            callback: Function to call when alert is triggered
        """
        self.alert_callbacks.append(callback)
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for alert conditions.
        
        Args:
            metrics: Current metrics to check
            
        Returns:
            List of triggered alerts
        """
        alerts = []
        
        try:
            # Check fidelity drop
            if 'fidelity' in metrics:
                fidelity = metrics['fidelity']
                if isinstance(fidelity, (int, float)) and fidelity < (1.0 - self.alert_thresholds['fidelity_drop']):
                    alerts.append({
                        'type': 'fidelity_drop',
                        'severity': 'high',
                        'message': f'Quantum fidelity dropped to {fidelity:.4f}',
                        'timestamp': datetime.utcnow().isoformat(),
                        'value': fidelity,
                        'threshold': 1.0 - self.alert_thresholds['fidelity_drop']
                    })
            
            # Check error rate increase
            if 'error_rate' in metrics:
                error_rate = metrics['error_rate']
                if isinstance(error_rate, (int, float)) and error_rate > self.alert_thresholds['error_rate_increase']:
                    alerts.append({
                        'type': 'error_rate_high',
                        'severity': 'medium',
                        'message': f'Error rate increased to {error_rate:.4f}',
                        'timestamp': datetime.utcnow().isoformat(),
                        'value': error_rate,
                        'threshold': self.alert_thresholds['error_rate_increase']
                    })
            
            # Check queue time
            if 'queue_time' in metrics:
                queue_time = metrics['queue_time']
                if isinstance(queue_time, (int, float)) and queue_time > self.alert_thresholds['queue_time_limit']:
                    alerts.append({
                        'type': 'queue_time_high',
                        'severity': 'low',
                        'message': f'Queue time is {queue_time:.1f} seconds',
                        'timestamp': datetime.utcnow().isoformat(),
                        'value': queue_time,
                        'threshold': self.alert_thresholds['queue_time_limit']
                    })
            
            # Check gradient explosion
            if 'gradient_norm' in metrics:
                grad_norm = metrics['gradient_norm']
                if isinstance(grad_norm, (int, float)) and grad_norm > self.alert_thresholds['gradient_explosion']:
                    alerts.append({
                        'type': 'gradient_explosion',
                        'severity': 'high',
                        'message': f'Gradient norm exploded to {grad_norm:.4f}',
                        'timestamp': datetime.utcnow().isoformat(),
                        'value': grad_norm,
                        'threshold': self.alert_thresholds['gradient_explosion']
                    })
            
            # Store alerts and trigger callbacks
            for alert in alerts:
                self.alert_history.append(alert)
                for callback in self.alert_callbacks:
                    try:
                        callback(alert['type'], alert)
                    except Exception as e:
                        logger.error(f"Error in alert callback: {e}")
        
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
        
        return alerts
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent alerts.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Alert summary statistics
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]
        
        alert_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for alert in recent_alerts:
            alert_counts[alert['type']] += 1
            severity_counts[alert['severity']] += 1
        
        return {
            'total_alerts': len(recent_alerts),
            'alert_types': dict(alert_counts),
            'severity_breakdown': dict(severity_counts),
            'time_window_hours': hours,
            'most_recent': recent_alerts[-1] if recent_alerts else None
        }


class QuantumMonitor:
    """Comprehensive quantum ML monitoring and metrics tracking."""
    
    def __init__(
        self,
        experiment_name: str = "quantum_experiment",
        tracking_uri: Optional[str] = None,
        enable_mlflow: bool = True,
        enable_wandb: bool = False,
        wandb_project: Optional[str] = None,
        alert_config: Optional[Dict[str, float]] = None,
        storage_path: Optional[str] = None,
    ) -> None:
        """Initialize quantum monitor.
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: URI for tracking server (e.g., MLflow)
            enable_mlflow: Enable MLflow integration
            enable_wandb: Enable Weights & Biases integration
            wandb_project: W&B project name
            alert_config: Custom alert thresholds
            storage_path: Path for storing monitoring data
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.storage_path = Path(storage_path or "./quantum_monitoring")
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.metrics_calculator = QuantumMetricsCalculator()
        self.alert_system = QuantumAlertSystem(alert_config)
        
        # Tracking state
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.start_time: Optional[float] = None
        self.current_run_id: Optional[str] = None
        self.is_running = False
        
        # Performance tracking
        self.execution_times: Dict[str, List[float]] = defaultdict(list)
        self.cost_tracking: Dict[str, float] = defaultdict(float)
        self.hardware_status: Dict[str, Any] = {}
        
        # Gradient and training tracking
        self.gradient_history: List[np.ndarray] = []
        self.parameter_history: List[np.ndarray] = []
        self.loss_history: List[float] = []
        
        # State and quantum tracking
        self.state_history: List[np.ndarray] = []
        self.fidelity_history: List[float] = []
        self.entanglement_history: List[float] = []
        
        # Real-time monitoring
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Initialize tracking backends
        self._initialize_mlflow(enable_mlflow)
        self._initialize_wandb(enable_wandb, wandb_project)
        
        # Set up alert callbacks
        self.alert_system.register_callback(self._handle_alert)
        
        logger.info(f"Initialized QuantumMonitor for experiment: {experiment_name}")
    
    def _initialize_mlflow(self, enable: bool) -> None:
        """Initialize MLflow tracking."""
        self.mlflow_enabled = enable and MLFLOW_AVAILABLE
        
        if self.mlflow_enabled:
            try:
                if self.tracking_uri:
                    mlflow.set_tracking_uri(self.tracking_uri)
                
                # Create or get experiment
                try:
                    experiment = mlflow.get_experiment_by_name(self.experiment_name)
                    if experiment is None:
                        self.experiment_id = mlflow.create_experiment(self.experiment_name)
                    else:
                        self.experiment_id = experiment.experiment_id
                except Exception:
                    self.experiment_id = mlflow.create_experiment(self.experiment_name)
                
                mlflow.set_experiment(self.experiment_name)
                logger.info("MLflow integration initialized")
            except Exception as e:
                logger.error(f"Failed to initialize MLflow: {e}")
                self.mlflow_enabled = False
    
    def _initialize_wandb(self, enable: bool, project: Optional[str]) -> None:
        """Initialize Weights & Biases tracking."""
        self.wandb_enabled = enable and WANDB_AVAILABLE
        self.wandb_run = None
        
        if self.wandb_enabled:
            try:
                # Initialize W&B run
                self.wandb_run = wandb.init(
                    project=project or "quantum-ml",
                    name=self.experiment_name,
                    tags=["quantum", "ml", "monitoring"],
                    config={
                        "experiment_name": self.experiment_name,
                        "framework": "quantum_mlops"
                    }
                )
                logger.info("Weights & Biases integration initialized")
            except Exception as e:
                logger.error(f"Failed to initialize W&B: {e}")
                self.wandb_enabled = False
    
    def _handle_alert(self, alert_type: str, alert_data: Dict[str, Any]) -> None:
        """Handle triggered alerts."""
        logger.warning(f"QUANTUM ALERT [{alert_type}]: {alert_data['message']}")
        
        # Log to tracking systems
        if self.mlflow_enabled and mlflow.active_run():
            mlflow.log_metric(f"alert.{alert_type}", 1.0)
        
        if self.wandb_enabled and self.wandb_run:
            self.wandb_run.log({f"alert/{alert_type}": 1.0})
        
        # Save alert to disk
        alert_file = self.storage_path / f"alerts_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(alert_file, 'a') as f:
            f.write(json.dumps(alert_data) + '\n')
        
    def start_run(self, run_name: Optional[str] = None) -> "QuantumMonitor":
        """Start monitoring run.
        
        Args:
            run_name: Optional name for the run
        """
        self.start_time = time.time()
        self.current_run_id = run_name or f"run_{int(self.start_time)}"
        self.is_running = True
        
        # Clear previous run data
        self.metrics.clear()
        self.gradient_history.clear()
        self.parameter_history.clear()
        self.loss_history.clear()
        self.state_history.clear()
        self.fidelity_history.clear()
        self.entanglement_history.clear()
        self.execution_times.clear()
        self.cost_tracking.clear()
        
        # Start MLflow run
        if self.mlflow_enabled:
            try:
                self.mlflow_run = mlflow.start_run(run_name=run_name)
                mlflow.set_tags({
                    "quantum": "true",
                    "framework": "quantum_mlops",
                    "monitoring": "enabled"
                })
            except Exception as e:
                logger.error(f"Failed to start MLflow run: {e}")
        
        logger.info(f"Started monitoring run: {self.current_run_id}")
        return self
        
    def __enter__(self) -> "QuantumMonitor":
        """Context manager entry."""
        return self.start_run()
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        status = "failed" if exc_type is not None else "completed"
        self.end_run(status=status)

    def end_run(self, status: str = "completed") -> None:
        """End monitoring run.
        
        Args:
            status: Status of the run (completed, failed, stopped)
        """
        if not self.is_running:
            logger.warning("No active run to end")
            return
        
        self.is_running = False
        
        if self.start_time:
            duration = time.time() - self.start_time
            self.log_metrics({"run_duration": duration, "run_status": status})
        
        # Stop real-time monitoring
        self.stop_realtime_monitoring()
        
        # Generate final summary
        summary = self.get_comprehensive_summary()
        
        # Save summary to disk
        summary_file = self.storage_path / f"summary_{self.current_run_id}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # End tracking runs
        if self.mlflow_enabled and hasattr(self, 'mlflow_run'):
            try:
                mlflow.end_run(status=status.upper())
            except Exception as e:
                logger.error(f"Failed to end MLflow run: {e}")
        
        if self.wandb_enabled and self.wandb_run:
            try:
                self.wandb_run.finish()
            except Exception as e:
                logger.error(f"Failed to finish W&B run: {e}")
        
        logger.info(f"Ended monitoring run: {self.current_run_id} with status: {status}")
            
    def log_metrics(self, metrics: Dict[str, Union[float, int, np.ndarray]], 
                   step: Optional[int] = None) -> None:
        """Log quantum ML metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for tracking
        """
        timestamp = time.time()
        
        for name, value in metrics.items():
            try:
                # Handle different value types
                if isinstance(value, np.ndarray):
                    # Log array statistics
                    array_stats = {
                        f"{name}_mean": float(np.mean(value)),
                        f"{name}_std": float(np.std(value)),
                        f"{name}_min": float(np.min(value)),
                        f"{name}_max": float(np.max(value))
                    }
                    
                    for stat_name, stat_value in array_stats.items():
                        self.metrics[stat_name].append({
                            "value": stat_value,
                            "timestamp": timestamp,
                            "step": step
                        })
                        
                        # Log to tracking systems
                        self._log_to_backends(stat_name, stat_value, step)
                else:
                    # Handle scalar values
                    scalar_value = float(value) if isinstance(value, (int, float)) else value
                    self.metrics[name].append({
                        "value": scalar_value,
                        "timestamp": timestamp,
                        "step": step
                    })
                    
                    # Log to tracking systems
                    self._log_to_backends(name, scalar_value, step)
            
            except Exception as e:
                logger.error(f"Error logging metric {name}: {e}")
        
        # Check for alerts
        try:
            alert_metrics = {k: v for k, v in metrics.items() 
                           if isinstance(v, (int, float))}
            self.alert_system.check_alerts(alert_metrics)
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def _log_to_backends(self, name: str, value: Union[float, int], 
                        step: Optional[int] = None) -> None:
        """Log metrics to enabled tracking backends."""
        try:
            # Log to MLflow
            if self.mlflow_enabled and mlflow.active_run():
                mlflow.log_metric(name, value, step=step)
            
            # Log to W&B
            if self.wandb_enabled and self.wandb_run:
                log_dict = {name: value}
                if step is not None:
                    log_dict["step"] = step
                self.wandb_run.log(log_dict)
        
        except Exception as e:
            logger.error(f"Error logging to backends: {e}")
            
    def log_quantum_state(self, state_vector: np.ndarray, step: int,
                         reference_state: Optional[np.ndarray] = None) -> None:
        """Log quantum state information with comprehensive analysis.
        
        Args:
            state_vector: Quantum state vector
            step: Training step
            reference_state: Reference state for fidelity calculation
        """
        try:
            # Store state for history
            self.state_history.append(state_vector.copy())
            
            # Calculate quantum properties
            state_norm = float(np.linalg.norm(state_vector))
            
            # Fidelity calculation
            if reference_state is not None:
                fidelity = self.metrics_calculator.calculate_fidelity(state_vector, reference_state)
            else:
                # Self-fidelity (should be 1 for normalized states)
                fidelity = float(np.abs(np.vdot(state_vector, state_vector)))
            
            # Entanglement entropy
            entanglement = self.metrics_calculator.calculate_entanglement_entropy(state_vector)
            
            # Participation ratio (measure of state complexity)
            prob_amplitudes = np.abs(state_vector) ** 2
            participation_ratio = 1.0 / np.sum(prob_amplitudes ** 2) if np.sum(prob_amplitudes ** 2) > 0 else 0.0
            
            # Store fidelity and entanglement history
            self.fidelity_history.append(fidelity)
            self.entanglement_history.append(entanglement)
            
            # Log all metrics
            metrics = {
                "quantum_fidelity": fidelity,
                "entanglement_entropy": entanglement,
                "state_norm": state_norm,
                "participation_ratio": participation_ratio,
                "state_complexity": float(np.std(prob_amplitudes))
            }
            
            self.log_metrics(metrics, step=step)
            
            # Visualize state if requested
            if len(self.state_history) % 10 == 0:  # Every 10 steps
                self._save_state_visualization(state_vector, step)
        
        except Exception as e:
            logger.error(f"Error logging quantum state: {e}")
    
    def log_circuit_execution(self, circuit_description: Dict[str, Any], 
                            execution_time: float, queue_time: float = 0.0,
                            cost: float = 0.0, backend_info: Optional[Dict] = None) -> None:
        """Log quantum circuit execution details.
        
        Args:
            circuit_description: Description of executed circuit
            execution_time: Time taken to execute
            queue_time: Time spent in queue
            cost: Cost of execution
            backend_info: Information about quantum backend
        """
        try:
            # Calculate circuit properties
            gates = circuit_description.get('gates', [])
            circuit_depth = self.metrics_calculator.calculate_circuit_depth(gates)
            n_qubits = circuit_description.get('n_qubits', 0)
            gate_count = len(gates)
            
            # Update execution tracking
            backend_name = backend_info.get('name', 'unknown') if backend_info else 'unknown'
            self.execution_times[backend_name].append(execution_time)
            self.cost_tracking[backend_name] += cost
            
            # Log execution metrics
            metrics = {
                "execution_time": execution_time,
                "queue_time": queue_time,
                "circuit_depth": circuit_depth,
                "gate_count": gate_count,
                "qubits_used": n_qubits,
                "execution_cost": cost,
                "total_cost": self.cost_tracking[backend_name]
            }
            
            self.log_metrics(metrics)
            
            # Update hardware status
            if backend_info:
                self.hardware_status.update({
                    'last_execution': datetime.utcnow().isoformat(),
                    'backend': backend_name,
                    'status': backend_info.get('status', 'unknown'),
                    'avg_execution_time': np.mean(self.execution_times[backend_name]),
                    'total_executions': len(self.execution_times[backend_name])
                })
        
        except Exception as e:
            logger.error(f"Error logging circuit execution: {e}")
    
    def log_training_step(self, loss: float, gradients: np.ndarray, 
                         parameters: np.ndarray, step: int,
                         additional_metrics: Optional[Dict[str, float]] = None) -> None:
        """Log training step information.
        
        Args:
            loss: Current loss value
            gradients: Gradient vector
            parameters: Current parameters
            step: Training step number
            additional_metrics: Additional metrics to log
        """
        try:
            # Store training history
            self.loss_history.append(loss)
            self.gradient_history.append(gradients.copy())
            self.parameter_history.append(parameters.copy())
            
            # Calculate gradient properties
            gradient_norm = float(np.linalg.norm(gradients))
            gradient_variance = self.metrics_calculator.calculate_gradient_variance(
                self.gradient_history[-10:]  # Last 10 gradients
            )
            
            # Parameter properties
            parameter_norm = float(np.linalg.norm(parameters))
            parameter_change = 0.0
            if len(self.parameter_history) > 1:
                param_diff = parameters - self.parameter_history[-2]
                parameter_change = float(np.linalg.norm(param_diff))
            
            # Training metrics
            metrics = {
                "loss": loss,
                "gradient_norm": gradient_norm,
                "gradient_variance": gradient_variance,
                "parameter_norm": parameter_norm,
                "parameter_change": parameter_change,
                "training_step": step
            }
            
            # Add additional metrics
            if additional_metrics:
                metrics.update(additional_metrics)
            
            self.log_metrics(metrics, step=step)
            
            # Check for training issues
            if gradient_norm > 10.0:
                logger.warning(f"Large gradient norm detected: {gradient_norm:.4f}")
            
            if len(self.loss_history) > 10:
                recent_losses = self.loss_history[-10:]
                if all(l == recent_losses[0] for l in recent_losses):
                    logger.warning("Loss has plateaued - training may have stalled")
        
        except Exception as e:
            logger.error(f"Error logging training step: {e}")

    def start_realtime_monitoring(self, update_interval: float = 1.0) -> None:
        """Start real-time monitoring in background thread.
        
        Args:
            update_interval: Update interval in seconds
        """
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Real-time monitoring already active")
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._realtime_monitoring_loop,
            args=(update_interval,),
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("Started real-time monitoring")
    
    def stop_realtime_monitoring(self) -> None:
        """Stop real-time monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=5.0)
            logger.info("Stopped real-time monitoring")
    
    def _realtime_monitoring_loop(self, update_interval: float) -> None:
        """Real-time monitoring loop."""
        while not self._stop_monitoring.wait(update_interval):
            try:
                # Check system metrics
                current_metrics = self._collect_system_metrics()
                if current_metrics:
                    self.log_metrics(current_metrics)
                
                # Check alerts
                self.alert_system.check_alerts(current_metrics)
                
            except Exception as e:
                logger.error(f"Error in real-time monitoring: {e}")
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        metrics = {}
        
        try:
            # Memory usage
            import psutil
            process = psutil.Process()
            metrics['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
            metrics['cpu_percent'] = process.cpu_percent()
        except ImportError:
            pass
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
        
        return metrics
    
    def _save_state_visualization(self, state_vector: np.ndarray, step: int) -> None:
        """Save quantum state visualization."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        try:
            n_qubits = int(np.log2(len(state_vector)))
            if n_qubits > 6:  # Too many qubits to visualize effectively
                return
            
            # Create probability distribution plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Probability amplitudes
            probabilities = np.abs(state_vector) ** 2
            basis_states = [f'|{i:0{n_qubits}b}⟩' for i in range(len(state_vector))]
            
            ax1.bar(range(len(probabilities)), probabilities)
            ax1.set_xlabel('Basis State')
            ax1.set_ylabel('Probability')
            ax1.set_title(f'State Probabilities (Step {step})')
            ax1.set_xticks(range(len(basis_states)))
            ax1.set_xticklabels(basis_states, rotation=45)
            
            # Phase information
            phases = np.angle(state_vector)
            ax2.bar(range(len(phases)), phases)
            ax2.set_xlabel('Basis State')
            ax2.set_ylabel('Phase (radians)')
            ax2.set_title(f'State Phases (Step {step})')
            ax2.set_xticks(range(len(basis_states)))
            ax2.set_xticklabels(basis_states, rotation=45)
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = self.storage_path / f"state_viz_step_{step}.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Log to tracking systems
            if self.mlflow_enabled and mlflow.active_run():
                mlflow.log_artifact(str(viz_path), "visualizations")
            
            if self.wandb_enabled and self.wandb_run:
                self.wandb_run.log({f"state_visualization_step_{step}": wandb.Image(str(viz_path))})
        
        except Exception as e:
            logger.error(f"Error saving state visualization: {e}")
        
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of logged metrics.
        
        Returns:
            Dictionary with metric summaries
        """
        summary = {}
        for metric_name, values in self.metrics.items():
            if values:
                latest_values = [v["value"] for v in values if isinstance(v["value"], (int, float))]
                if latest_values:
                    summary[metric_name] = {
                        "latest": latest_values[-1],
                        "mean": float(np.mean(latest_values)),
                        "std": float(np.std(latest_values)),
                        "min": float(np.min(latest_values)),
                        "max": float(np.max(latest_values)),
                        "count": len(latest_values),
                        "trend": self._calculate_trend(latest_values)
                    }
        return summary
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for metric values."""
        if len(values) < 2:
            return "stable"
        
        try:
            # Simple linear trend
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            
            if slope > 0.01:
                return "increasing"
            elif slope < -0.01:
                return "decreasing"
            else:
                return "stable"
        except Exception:
            return "unknown"
    
    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary including all monitoring data."""
        summary = {
            "experiment_info": {
                "name": self.experiment_name,
                "run_id": self.current_run_id,
                "start_time": self.start_time,
                "duration": time.time() - self.start_time if self.start_time else 0,
                "is_running": self.is_running
            },
            "metrics_summary": self.get_metrics_summary(),
            "training_summary": {
                "total_steps": len(self.loss_history),
                "final_loss": self.loss_history[-1] if self.loss_history else None,
                "best_loss": min(self.loss_history) if self.loss_history else None,
                "gradient_explosion_count": sum(1 for g in self.gradient_history 
                                               if np.linalg.norm(g) > 10.0),
                "convergence_achieved": self._check_convergence()
            },
            "quantum_summary": {
                "states_logged": len(self.state_history),
                "avg_fidelity": np.mean(self.fidelity_history) if self.fidelity_history else None,
                "avg_entanglement": np.mean(self.entanglement_history) if self.entanglement_history else None,
                "fidelity_trend": self._calculate_trend(self.fidelity_history) if self.fidelity_history else None
            },
            "execution_summary": {
                "backends_used": list(self.execution_times.keys()),
                "total_cost": sum(self.cost_tracking.values()),
                "avg_execution_times": {
                    backend: np.mean(times) for backend, times in self.execution_times.items()
                },
                "hardware_status": self.hardware_status
            },
            "alert_summary": self.alert_system.get_alert_summary(),
            "tracking_status": {
                "mlflow_enabled": self.mlflow_enabled,
                "wandb_enabled": self.wandb_enabled,
                "storage_path": str(self.storage_path)
            }
        }
        
        return summary
    
    def _check_convergence(self, window: int = 50, threshold: float = 0.001) -> bool:
        """Check if training has converged."""
        if len(self.loss_history) < window:
            return False
        
        recent_losses = self.loss_history[-window:]
        return np.std(recent_losses) < threshold
        
    def export_metrics(self, filepath: str, format: str = "json") -> None:
        """Export metrics to file.
        
        Args:
            filepath: Path to export metrics
            format: Export format (json, csv, or comprehensive)
        """
        try:
            if format.lower() == "json":
                summary = self.get_metrics_summary()
                with open(filepath, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
            
            elif format.lower() == "csv":
                self._export_to_csv(filepath)
            
            elif format.lower() == "comprehensive":
                summary = self.get_comprehensive_summary()
                with open(filepath, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Exported metrics to {filepath} in {format} format")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            raise
    
    def _export_to_csv(self, filepath: str) -> None:
        """Export metrics to CSV format."""
        import csv
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            header = ['timestamp', 'metric_name', 'value', 'step']
            writer.writerow(header)
            
            # Write data
            for metric_name, values in self.metrics.items():
                for entry in values:
                    row = [
                        entry.get('timestamp', ''),
                        metric_name,
                        entry.get('value', ''),
                        entry.get('step', '')
                    ]
                    writer.writerow(row)
    
    def create_dashboard(self, port: int = 8050, host: str = "127.0.0.1") -> None:
        """Create and launch monitoring dashboard.
        
        Args:
            port: Port to run dashboard on
            host: Host to bind to
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available. Cannot create dashboard.")
            return
        
        try:
            # This would create an interactive dashboard using Plotly Dash
            # For now, we'll create static visualizations
            self.create_static_dashboard()
            logger.info("Static dashboard created. For interactive dashboard, implement Dash integration.")
        
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
    
    def create_static_dashboard(self) -> str:
        """Create static HTML dashboard."""
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available for dashboard creation")
            return ""
        
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Loss History', 'Gradient Norm',
                    'Fidelity Over Time', 'Entanglement Entropy',
                    'Execution Times', 'Cost Tracking'
                ),
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}]
                ]
            )
            
            # Loss history
            if self.loss_history:
                fig.add_trace(
                    go.Scatter(y=self.loss_history, mode='lines', name='Loss'),
                    row=1, col=1
                )
            
            # Gradient norms
            if self.gradient_history:
                gradient_norms = [np.linalg.norm(g) for g in self.gradient_history]
                fig.add_trace(
                    go.Scatter(y=gradient_norms, mode='lines', name='Gradient Norm'),
                    row=1, col=2
                )
            
            # Fidelity
            if self.fidelity_history:
                fig.add_trace(
                    go.Scatter(y=self.fidelity_history, mode='lines', name='Fidelity'),
                    row=2, col=1
                )
            
            # Entanglement
            if self.entanglement_history:
                fig.add_trace(
                    go.Scatter(y=self.entanglement_history, mode='lines', name='Entanglement'),
                    row=2, col=2
                )
            
            # Execution times
            if self.execution_times:
                for backend, times in self.execution_times.items():
                    fig.add_trace(
                        go.Scatter(y=times, mode='markers', name=f'{backend} Exec Time'),
                        row=3, col=1
                    )
            
            # Cost tracking
            if self.cost_tracking:
                backends = list(self.cost_tracking.keys())
                costs = list(self.cost_tracking.values())
                fig.add_trace(
                    go.Bar(x=backends, y=costs, name='Total Cost'),
                    row=3, col=2
                )
            
            # Update layout
            fig.update_layout(
                height=800,
                title_text=f"Quantum ML Monitoring Dashboard - {self.experiment_name}",
                showlegend=True
            )
            
            # Save dashboard
            dashboard_path = self.storage_path / "dashboard.html"
            fig.write_html(str(dashboard_path))
            
            logger.info(f"Dashboard saved to {dashboard_path}")
            return str(dashboard_path)
        
        except Exception as e:
            logger.error(f"Error creating static dashboard: {e}")
            return ""
    
    def generate_report(self, include_visualizations: bool = True) -> str:
        """Generate comprehensive monitoring report.
        
        Args:
            include_visualizations: Whether to include visualizations
            
        Returns:
            Path to generated report
        """
        try:
            summary = self.get_comprehensive_summary()
            
            # Generate markdown report
            report_lines = [
                f"# Quantum ML Monitoring Report",
                f"",
                f"**Experiment:** {self.experiment_name}",
                f"**Run ID:** {self.current_run_id}",
                f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
                f"",
                f"## Experiment Overview",
                f"",
                f"- Duration: {summary['experiment_info']['duration']:.2f} seconds",
                f"- Status: {'Running' if summary['experiment_info']['is_running'] else 'Completed'}",
                f"- Training Steps: {summary['training_summary']['total_steps']}",
                f"- Quantum States Logged: {summary['quantum_summary']['states_logged']}",
                f"",
                f"## Training Summary",
                f"",
            ]
            
            if summary['training_summary']['final_loss'] is not None:
                report_lines.extend([
                    f"- Final Loss: {summary['training_summary']['final_loss']:.6f}",
                    f"- Best Loss: {summary['training_summary']['best_loss']:.6f}",
                    f"- Convergence: {'Yes' if summary['training_summary']['convergence_achieved'] else 'No'}",
                ])
            
            report_lines.extend([
                f"",
                f"## Quantum Metrics",
                f"",
            ])
            
            if summary['quantum_summary']['avg_fidelity'] is not None:
                report_lines.extend([
                    f"- Average Fidelity: {summary['quantum_summary']['avg_fidelity']:.4f}",
                    f"- Average Entanglement: {summary['quantum_summary']['avg_entanglement']:.4f}",
                    f"- Fidelity Trend: {summary['quantum_summary']['fidelity_trend']}",
                ])
            
            # Add alert summary
            alert_summary = summary['alert_summary']
            if alert_summary['total_alerts'] > 0:
                report_lines.extend([
                    f"",
                    f"## Alerts ({alert_summary['total_alerts']} total)",
                    f"",
                ])
                
                for alert_type, count in alert_summary['alert_types'].items():
                    report_lines.append(f"- {alert_type}: {count}")
            
            # Add execution summary
            exec_summary = summary['execution_summary']
            if exec_summary['backends_used']:
                report_lines.extend([
                    f"",
                    f"## Execution Summary",
                    f"",
                    f"- Backends Used: {', '.join(exec_summary['backends_used'])}",
                    f"- Total Cost: ${exec_summary['total_cost']:.4f}",
                ])
            
            # Save report
            report_path = self.storage_path / f"report_{self.current_run_id}.md"
            with open(report_path, 'w') as f:
                f.write('\n'.join(report_lines))
            
            # Create visualizations if requested
            if include_visualizations:
                dashboard_path = self.create_static_dashboard()
                if dashboard_path:
                    report_lines.extend([
                        f"",
                        f"## Visualizations",
                        f"",
                        f"Interactive dashboard: [dashboard.html]({dashboard_path})",
                    ])
            
            logger.info(f"Generated report: {report_path}")
            return str(report_path)
        
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return ""


class QuantumVisualization:
    """Quantum-specific visualization utilities."""
    
    @staticmethod
    def plot_bloch_sphere(state_vector: np.ndarray, save_path: Optional[str] = None) -> Optional[str]:
        """Plot quantum state on Bloch sphere (for single qubit states).
        
        Args:
            state_vector: Single qubit state vector
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot or None
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for Bloch sphere plotting")
            return None
        
        if len(state_vector) != 2:
            logger.warning("Bloch sphere visualization only supports single qubit states")
            return None
        
        try:
            # Normalize state
            state_norm = state_vector / np.linalg.norm(state_vector)
            alpha, beta = state_norm[0], state_norm[1]
            
            # Calculate Bloch vector components
            x = 2 * np.real(np.conj(alpha) * beta)
            y = 2 * np.imag(np.conj(alpha) * beta)
            z = np.abs(alpha)**2 - np.abs(beta)**2
            
            # Create 3D plot
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Draw Bloch sphere
            u = np.linspace(0, 2 * np.pi, 50)
            v = np.linspace(0, np.pi, 50)
            sphere_x = np.outer(np.cos(u), np.sin(v))
            sphere_y = np.outer(np.sin(u), np.sin(v))
            sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
            
            ax.plot_surface(sphere_x, sphere_y, sphere_z, alpha=0.3, color='lightblue')
            
            # Draw state vector
            ax.quiver(0, 0, 0, x, y, z, color='red', arrow_length_ratio=0.1, linewidth=3)
            
            # Add coordinate axes
            ax.quiver(0, 0, 0, 1, 0, 0, color='black', arrow_length_ratio=0.1, alpha=0.5)
            ax.quiver(0, 0, 0, 0, 1, 0, color='black', arrow_length_ratio=0.1, alpha=0.5)
            ax.quiver(0, 0, 0, 0, 0, 1, color='black', arrow_length_ratio=0.1, alpha=0.5)
            
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Quantum State on Bloch Sphere')
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                plt.show()
                return None
        
        except Exception as e:
            logger.error(f"Error plotting Bloch sphere: {e}")
            return None
    
    @staticmethod
    def plot_state_evolution(state_history: List[np.ndarray], 
                           save_path: Optional[str] = None) -> Optional[str]:
        """Plot evolution of quantum state probabilities.
        
        Args:
            state_history: List of state vectors over time
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot or None
        """
        if not MATPLOTLIB_AVAILABLE or not state_history:
            return None
        
        try:
            n_states = len(state_history[0])
            n_steps = len(state_history)
            
            # Calculate probability evolution
            prob_evolution = np.zeros((n_steps, n_states))
            for i, state in enumerate(state_history):
                prob_evolution[i] = np.abs(state) ** 2
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot each basis state probability
            for j in range(min(n_states, 8)):  # Limit to first 8 states for readability
                ax.plot(prob_evolution[:, j], label=f'|{j:03b}⟩')
            
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Probability')
            ax.set_title('Quantum State Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                plt.show()
                return None
        
        except Exception as e:
            logger.error(f"Error plotting state evolution: {e}")
            return None


# Convenience functions for quick monitoring setup
def create_quantum_monitor(experiment_name: str, 
                          enable_mlflow: bool = True,
                          enable_wandb: bool = False,
                          **kwargs) -> QuantumMonitor:
    """Create and configure a quantum monitor.
    
    Args:
        experiment_name: Name of the experiment
        enable_mlflow: Enable MLflow tracking
        enable_wandb: Enable W&B tracking
        **kwargs: Additional configuration options
        
    Returns:
        Configured QuantumMonitor instance
    """
    return QuantumMonitor(
        experiment_name=experiment_name,
        enable_mlflow=enable_mlflow,
        enable_wandb=enable_wandb,
        **kwargs
    )


def monitor_quantum_training(monitor: QuantumMonitor, 
                           training_func: Callable,
                           *args, **kwargs) -> Any:
    """Decorator-style monitoring for training functions.
    
    Args:
        monitor: QuantumMonitor instance
        training_func: Training function to monitor
        *args, **kwargs: Arguments for training function
        
    Returns:
        Result of training function
    """
    with monitor:
        monitor.start_realtime_monitoring()
        try:
            result = training_func(*args, **kwargs)
            return result
        finally:
            monitor.stop_realtime_monitoring()


# Export key classes and functions
__all__ = [
    'QuantumMonitor',
    'QuantumMetricsCalculator', 
    'QuantumAlertSystem',
    'QuantumVisualization',
    'create_quantum_monitor',
    'monitor_quantum_training'
]