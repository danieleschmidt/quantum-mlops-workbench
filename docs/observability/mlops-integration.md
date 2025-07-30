# MLOps Integration Guide

## Overview

This guide provides comprehensive integration patterns for quantum MLOps with popular MLOps platforms and tools. It covers experiment tracking, model versioning, deployment pipelines, and monitoring for quantum machine learning workflows.

## 🎯 MLOps Integration Objectives

### Core Goals
- **Experiment Reproducibility**: Track quantum experiments with full reproducibility
- **Model Lifecycle Management**: Version and manage quantum models effectively
- **Automated Deployment**: Deploy quantum models to production environments
- **Performance Monitoring**: Monitor quantum model performance in production
- **Resource Optimization**: Optimize quantum computing resource usage

### Quantum-Specific Challenges
- **Stochastic Results**: Handle probabilistic quantum measurement outcomes
- **Hardware Variability**: Account for quantum hardware noise and drift
- **Complex Dependencies**: Manage quantum framework and hardware dependencies
- **Cost Management**: Track and optimize expensive quantum computing costs
- **Hybrid Workflows**: Integrate classical and quantum components

## 🔬 MLflow Integration

### 1. Quantum Experiment Tracking

#### Enhanced MLflow Client
```python
#!/usr/bin/env python3
"""
Enhanced MLflow integration for quantum machine learning experiments
"""
import mlflow
import mlflow.pytorch
import numpy as np
import json
import pickle
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import io
import base64

@dataclass
class QuantumExperimentConfig:
    """Configuration for quantum experiments"""
    algorithm_name: str
    n_qubits: int
    n_layers: int
    shots: int
    provider: str
    device: str
    noise_model: Optional[str] = None
    optimization_method: Optional[str] = None
    convergence_threshold: Optional[float] = None

class QuantumMLflowClient:
    def __init__(self, tracking_uri: Optional[str] = None, experiment_name: str = "quantum-ml-experiments"):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except Exception:
            experiment_id = mlflow.create_experiment(experiment_name)
        
        mlflow.set_experiment(experiment_name)
        self.experiment_id = experiment_id
    
    def start_quantum_run(self, 
                         config: QuantumExperimentConfig,
                         run_name: Optional[str] = None,
                         nested: bool = False) -> mlflow.ActiveRun:
        """Start a quantum ML experiment run"""
        run = mlflow.start_run(run_name=run_name, nested=nested)
        
        # Log quantum configuration
        self.log_quantum_config(config)
        
        # Set quantum-specific tags
        mlflow.set_tags({
            "quantum.framework": "pennylane",  # or detect automatically
            "quantum.type": "variational",
            "quantum.provider": config.provider,
            "quantum.device": config.device,
            "mlops.stage": "experiment"
        })
        
        return run
    
    def log_quantum_config(self, config: QuantumExperimentConfig):
        """Log quantum experiment configuration"""
        config_dict = asdict(config)
        
        # Log as parameters
        for key, value in config_dict.items():
            if value is not None:
                mlflow.log_param(f"quantum.{key}", value)
        
        # Log as artifact
        with open("quantum_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        mlflow.log_artifact("quantum_config.json", "config")
    
    def log_quantum_circuit(self, 
                           circuit_func, 
                           params: np.ndarray,
                           circuit_name: str = "quantum_circuit"):
        """Log quantum circuit information"""
        try:
            # Create circuit visualization (framework-dependent)
            circuit_diagram = self._create_circuit_diagram(circuit_func, params)
            
            # Log circuit parameters
            mlflow.log_param("quantum.circuit.parameter_count", len(params))
            
            # Log circuit as artifact
            if circuit_diagram:
                mlflow.log_text(circuit_diagram, f"{circuit_name}_diagram.txt")
            
            # Log parameters
            np.save("circuit_params.npy", params)
            mlflow.log_artifact("circuit_params.npy", "circuit")
            
        except Exception as e:
            mlflow.log_param("quantum.circuit.logging_error", str(e))
    
    def log_quantum_metrics(self, 
                           metrics: Dict[str, Union[float, int, np.ndarray]],
                           step: Optional[int] = None):
        """Log quantum-specific metrics"""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"quantum.{key}", value, step=step)
            elif isinstance(value, np.ndarray):
                # Log array metrics as statistics
                mlflow.log_metric(f"quantum.{key}.mean", float(np.mean(value)), step=step)
                mlflow.log_metric(f"quantum.{key}.std", float(np.std(value)), step=step)
                mlflow.log_metric(f"quantum.{key}.min", float(np.min(value)), step=step)
                mlflow.log_metric(f"quantum.{key}.max", float(np.max(value)), step=step)
    
    def log_quantum_results(self, 
                           measurement_results: Dict[str, Any],
                           training_history: Optional[Dict[str, List[float]]] = None):
        """Log quantum computation results"""
        # Log measurement statistics
        if "measurements" in measurement_results:
            measurements = measurement_results["measurements"]
            if isinstance(measurements, (list, np.ndarray)):
                measurements = np.array(measurements)
                self.log_quantum_metrics({
                    "measurement.mean": np.mean(measurements),
                    "measurement.variance": np.var(measurements),
                    "measurement.fidelity": measurement_results.get("fidelity", 0)
                })
        
        # Log training history
        if training_history:
            for metric_name, values in training_history.items():
                for step, value in enumerate(values):
                    mlflow.log_metric(f"training.{metric_name}", value, step=step)
        
        # Log results as artifact
        with open("quantum_results.json", "w") as f:
            json.dump(measurement_results, f, indent=2, default=str)
        mlflow.log_artifact("quantum_results.json", "results")
    
    def log_quantum_cost(self, 
                        provider: str,
                        device: str,
                        cost: float,
                        cost_breakdown: Optional[Dict[str, float]] = None):
        """Log quantum computing costs"""
        mlflow.log_metric("quantum.cost.total", cost)
        mlflow.log_param("quantum.cost.provider", provider)
        mlflow.log_param("quantum.cost.device", device)
        
        if cost_breakdown:
            for cost_type, amount in cost_breakdown.items():
                mlflow.log_metric(f"quantum.cost.{cost_type}", amount)
    
    def log_quantum_model(self, 
                         model_func,
                         optimal_params: np.ndarray,
                         model_name: str = "quantum_model"):
        """Log quantum model with parameters"""
        # Save model parameters
        model_data = {
            "optimal_parameters": optimal_params.tolist(),
            "parameter_shape": optimal_params.shape,
            "model_type": "variational_quantum",
            "framework": "pennylane"  # or detect automatically
        }
        
        with open(f"{model_name}.json", "w") as f:
            json.dump(model_data, f, indent=2)
        
        # Log as MLflow model
        mlflow.log_artifact(f"{model_name}.json", "model")
        
        # Register model if in experiment mode
        try:
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model/{model_name}.json"
            mlflow.register_model(model_uri, model_name)
        except Exception as e:
            print(f"Could not register model: {e}")
    
    def log_hardware_info(self, 
                         provider: str,
                         device: str,
                         hardware_info: Dict[str, Any]):
        """Log quantum hardware information"""
        # Log basic hardware info as parameters
        mlflow.log_param("quantum.hardware.provider", provider)
        mlflow.log_param("quantum.hardware.device", device)
        
        # Log detailed hardware info
        for key, value in hardware_info.items():
            if isinstance(value, (str, int, float)):
                mlflow.log_param(f"quantum.hardware.{key}", value)
        
        # Log full hardware info as artifact
        with open("hardware_info.json", "w") as f:
            json.dump(hardware_info, f, indent=2, default=str)
        mlflow.log_artifact("hardware_info.json", "hardware")
    
    def log_comparison_plot(self, 
                           quantum_results: List[float],
                           classical_results: List[float],
                           title: str = "Quantum vs Classical Comparison"):
        """Log quantum vs classical comparison plot"""
        plt.figure(figsize=(10, 6))
        
        x = range(len(quantum_results))
        plt.plot(x, quantum_results, 'b-', label='Quantum', marker='o')
        plt.plot(x, classical_results, 'r-', label='Classical', marker='s')
        
        plt.xlabel('Iteration')
        plt.ylabel('Performance Metric')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("quantum_classical_comparison.png", dpi=300, bbox_inches='tight')
        mlflow.log_artifact("quantum_classical_comparison.png", "plots")
        plt.close()
    
    def _create_circuit_diagram(self, circuit_func, params: np.ndarray) -> Optional[str]:
        """Create quantum circuit diagram (framework-dependent)"""
        try:
            # This would be implemented based on the quantum framework
            # For now, return a simple text representation
            return f"Quantum Circuit with {len(params)} parameters\nParameters: {params[:5]}..."
        except Exception:
            return None
    
    def end_run(self):
        """End the current MLflow run"""
        mlflow.end_run()

# Example usage
def example_quantum_experiment():
    """Example quantum ML experiment with MLflow tracking"""
    # Initialize quantum MLflow client
    client = QuantumMLflowClient(experiment_name="quantum-vqe-optimization")
    
    # Define experiment configuration
    config = QuantumExperimentConfig(
        algorithm_name="VQE",
        n_qubits=4,
        n_layers=3,
        shots=1024,
        provider="pennylane",
        device="default.qubit",
        optimization_method="COBYLA",
        convergence_threshold=1e-6
    )
    
    # Start experiment run
    with client.start_quantum_run(config, run_name="vqe_h2_molecule"):
        # Simulate quantum circuit
        def quantum_circuit(params):
            # Simulate VQE circuit
            return np.random.random() - 0.5  # Random energy
        
        # Log circuit
        initial_params = np.random.random(12)  # 4 qubits * 3 layers
        client.log_quantum_circuit(quantum_circuit, initial_params)
        
        # Simulate optimization
        training_history = {"energy": [], "gradient_norm": []}
        current_params = initial_params.copy()
        
        for iteration in range(50):
            # Simulate optimization step
            energy = quantum_circuit(current_params)
            gradient_norm = np.random.random()
            
            training_history["energy"].append(energy)
            training_history["gradient_norm"].append(gradient_norm)
            
            # Update parameters (simulate)
            current_params += np.random.normal(0, 0.01, size=current_params.shape)
            
            # Log metrics
            client.log_quantum_metrics({
                "energy": energy,
                "gradient_norm": gradient_norm,
                "parameter_norm": np.linalg.norm(current_params)
            }, step=iteration)
        
        # Log final results
        final_results = {
            "final_energy": training_history["energy"][-1],
            "convergence_iterations": 50,
            "measurements": training_history["energy"],
            "fidelity": 0.95
        }
        
        client.log_quantum_results(final_results, training_history)
        
        # Log model
        client.log_quantum_model(quantum_circuit, current_params, "vqe_h2_model")
        
        # Log hardware info
        hardware_info = {
            "qubit_count": 4,
            "connectivity": "all-to-all",
            "gate_error_rate": 0.001,
            "readout_error_rate": 0.02,
            "coherence_time": 100  # microseconds
        }
        client.log_hardware_info("pennylane", "default.qubit", hardware_info)
        
        # Log cost
        client.log_quantum_cost("pennylane", "default.qubit", 0.0)  # Free simulator
        
        # Log comparison plot
        classical_results = [r + np.random.normal(0, 0.1) for r in training_history["energy"]]
        client.log_comparison_plot(training_history["energy"], classical_results)

if __name__ == "__main__":
    example_quantum_experiment()
```

### 2. Model Registry Integration

#### Quantum Model Management
```python
#!/usr/bin/env python3
"""
Quantum model registry and versioning with MLflow
"""
import mlflow
from mlflow.models import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec
import numpy as np
import json
import pickle
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import joblib

@dataclass
class QuantumModelMetadata:
    """Metadata for quantum models"""
    framework: str
    n_qubits: int
    n_parameters: int
    circuit_depth: int
    provider: str
    device: str
    optimization_method: str
    training_shots: int
    final_fidelity: float
    convergence_iterations: int

class QuantumModelRegistry:
    def __init__(self, registry_uri: Optional[str] = None):
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
        
        self.client = mlflow.tracking.MlflowClient()
    
    def register_quantum_model(self,
                              model_name: str,
                              model_params: np.ndarray,
                              circuit_func,
                              metadata: QuantumModelMetadata,
                              run_id: Optional[str] = None,
                              model_version: Optional[str] = None) -> str:
        """Register a quantum model in MLflow registry"""
        
        # Create model artifacts
        model_artifacts = self._create_model_artifacts(
            model_params, circuit_func, metadata
        )
        
        # Create model signature
        signature = self._create_quantum_model_signature(metadata.n_qubits)
        
        # Log model with MLflow
        with mlflow.start_run(run_id=run_id):
            # Log model artifacts
            for artifact_name, artifact_path in model_artifacts.items():
                mlflow.log_artifact(artifact_path, f"model/{artifact_name}")
            
            # Log model metadata
            self._log_model_metadata(metadata)
            
            # Create MLflow model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            
            # Register model
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags={
                    "quantum": "true",
                    "framework": metadata.framework,
                    "qubits": str(metadata.n_qubits),
                    "provider": metadata.provider
                }
            )
            
            return registered_model.version
    
    def _create_model_artifacts(self,
                               params: np.ndarray,
                               circuit_func,
                               metadata: QuantumModelMetadata) -> Dict[str, str]:
        """Create model artifacts for registration"""
        artifacts = {}
        
        # Save parameters
        params_file = "quantum_parameters.npy"
        np.save(params_file, params)
        artifacts["parameters"] = params_file
        
        # Save circuit definition (simplified)
        circuit_file = "circuit_definition.py"
        circuit_code = self._serialize_circuit(circuit_func)
        with open(circuit_file, "w") as f:
            f.write(circuit_code)
        artifacts["circuit"] = circuit_file
        
        # Save metadata
        metadata_file = "model_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata.__dict__, f, indent=2)
        artifacts["metadata"] = metadata_file
        
        # Create model wrapper
        wrapper_file = "quantum_model_wrapper.py"
        wrapper_code = self._create_model_wrapper(metadata)
        with open(wrapper_file, "w") as f:
            f.write(wrapper_code)
        artifacts["wrapper"] = wrapper_file
        
        return artifacts
    
    def _create_quantum_model_signature(self, n_qubits: int) -> ModelSignature:
        """Create MLflow model signature for quantum model"""
        # Input schema: quantum feature vector
        input_schema = Schema([
            ColSpec("double", f"feature_{i}") for i in range(n_qubits)
        ])
        
        # Output schema: quantum prediction (probability or expectation)
        output_schema = Schema([
            ColSpec("double", "quantum_prediction"),
            ColSpec("double", "confidence")
        ])
        
        return ModelSignature(inputs=input_schema, outputs=output_schema)
    
    def _serialize_circuit(self, circuit_func) -> str:
        """Serialize quantum circuit function (simplified)"""
        return f"""
# Quantum Circuit Definition
# This is a simplified serialization - in practice, you would use
# framework-specific serialization methods

def quantum_circuit(params, x):
    # Circuit implementation would be here
    # This is a placeholder
    return 0.0

# Circuit metadata
circuit_info = {{
    "type": "variational",
    "framework": "pennylane",
    "serialization_method": "source_code"
}}
"""
    
    def _create_model_wrapper(self, metadata: QuantumModelMetadata) -> str:
        """Create model wrapper for deployment"""
        return f"""
import numpy as np
import json
from typing import Dict, Any, List

class QuantumModelWrapper:
    def __init__(self, model_path: str):
        # Load model artifacts
        self.parameters = np.load(f"{{model_path}}/parameters/quantum_parameters.npy")
        
        with open(f"{{model_path}}/metadata/model_metadata.json", "r") as f:
            self.metadata = json.load(f)
        
        # Initialize quantum framework (framework-specific)
        self._initialize_quantum_backend()
    
    def _initialize_quantum_backend(self):
        '''Initialize quantum computing backend'''
        # Framework-specific initialization
        pass
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        '''Make predictions using quantum model'''
        predictions = []
        confidences = []
        
        for x in X:
            # Execute quantum circuit with current parameters
            result = self._execute_quantum_circuit(x)
            predictions.append(result['prediction'])
            confidences.append(result['confidence'])
        
        return {{
            'predictions': np.array(predictions),
            'confidences': np.array(confidences),
            'model_metadata': self.metadata
        }}
    
    def _execute_quantum_circuit(self, x: np.ndarray) -> Dict[str, float]:
        '''Execute quantum circuit for single input'''
        # This would contain the actual quantum execution logic
        # For now, return mock results
        return {{
            'prediction': np.random.random(),
            'confidence': 0.95
        }}

# Model metadata
MODEL_METADATA = {metadata.__dict__}
"""
    
    def _log_model_metadata(self, metadata: QuantumModelMetadata):
        """Log model metadata as MLflow parameters"""
        for key, value in metadata.__dict__.items():
            mlflow.log_param(f"model.{key}", value)
    
    def get_model_version(self, model_name: str, version: str = "latest"):
        """Get specific version of registered quantum model"""
        if version == "latest":
            return self.client.get_latest_versions(model_name, stages=["Production"])
        else:
            return self.client.get_model_version(model_name, version)
    
    def deploy_model(self,
                    model_name: str,
                    version: str,
                    deployment_target: str = "staging") -> str:
        """Deploy quantum model to target environment"""
        # Transition model to deployment stage
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=deployment_target.title(),
            archive_existing_versions=True
        )
        
        # Create deployment configuration
        deployment_config = {
            "model_name": model_name,
            "model_version": version,
            "deployment_target": deployment_target,
            "quantum_backend": "cloud",  # or "local", "simulator"
            "scaling_config": {
                "min_instances": 1,
                "max_instances": 5,
                "cpu_per_instance": 2,
                "memory_per_instance": "4Gi"
            }
        }
        
        # Log deployment configuration
        with mlflow.start_run():
            mlflow.log_dict(deployment_config, "deployment_config.json")
            deployment_run_id = mlflow.active_run().info.run_id
        
        return deployment_run_id
    
    def monitor_model_performance(self,
                                 model_name: str,
                                 version: str,
                                 performance_metrics: Dict[str, float]):
        """Monitor deployed model performance"""
        # Log performance metrics
        with mlflow.start_run():
            mlflow.set_tags({
                "monitoring": "true",
                "model_name": model_name,
                "model_version": version
            })
            
            for metric_name, value in performance_metrics.items():
                mlflow.log_metric(f"production.{metric_name}", value)
            
            # Check for performance degradation
            self._check_performance_drift(model_name, version, performance_metrics)
    
    def _check_performance_drift(self,
                                model_name: str,
                                version: str,
                                current_metrics: Dict[str, float]):
        """Check for model performance drift"""
        # Get historical performance data
        # This would query historical metrics and compare
        
        # Simple threshold-based drift detection
        thresholds = {
            "accuracy": 0.05,  # 5% degradation threshold
            "fidelity": 0.02,  # 2% fidelity degradation
            "latency": 2.0     # 2x latency increase
        }
        
        drift_detected = False
        drift_details = {}
        
        for metric, current_value in current_metrics.items():
            if metric in thresholds:
                # This would compare with historical baseline
                historical_baseline = 0.9  # Mock baseline
                
                if abs(current_value - historical_baseline) > thresholds[metric]:
                    drift_detected = True
                    drift_details[metric] = {
                        "current": current_value,
                        "baseline": historical_baseline,
                        "drift": abs(current_value - historical_baseline),
                        "threshold": thresholds[metric]
                    }
        
        if drift_detected:
            # Log drift detection
            with mlflow.start_run():
                mlflow.set_tags({
                    "alert": "performance_drift",
                    "model_name": model_name,
                    "model_version": version
                })
                mlflow.log_dict(drift_details, "drift_analysis.json")

# Example usage
def example_model_registry():
    """Example quantum model registration and management"""
    registry = QuantumModelRegistry()
    
    # Create example quantum model
    model_params = np.random.random(20)  # Example parameters
    
    def example_circuit(params, x):
        # Example quantum circuit
        return np.sum(params * x)  # Simplified
    
    # Create model metadata
    metadata = QuantumModelMetadata(
        framework="pennylane",
        n_qubits=4,
        n_parameters=20,
        circuit_depth=5,
        provider="ibm",
        device="ibmq_qasm_simulator",
        optimization_method="COBYLA",
        training_shots=1024,
        final_fidelity=0.95,
        convergence_iterations=100
    )
    
    # Register model
    model_version = registry.register_quantum_model(
        model_name="quantum_classifier_v1",
        model_params=model_params,
        circuit_func=example_circuit,
        metadata=metadata
    )
    
    print(f"Registered model version: {model_version}")
    
    # Deploy model
    deployment_id = registry.deploy_model(
        model_name="quantum_classifier_v1",
        version=model_version,
        deployment_target="staging"
    )
    
    print(f"Deployment ID: {deployment_id}")
    
    # Monitor performance
    performance_metrics = {
        "accuracy": 0.88,
        "fidelity": 0.93,
        "latency": 1.5,
        "cost_per_prediction": 0.05
    }
    
    registry.monitor_model_performance(
        model_name="quantum_classifier_v1",
        version=model_version,
        performance_metrics=performance_metrics
    )

if __name__ == "__main__":
    example_model_registry()
```

This MLOps integration guide provides comprehensive patterns for integrating quantum machine learning with popular MLOps platforms. The enhanced MLflow client and model registry specifically address quantum computing requirements while maintaining compatibility with existing MLOps workflows.