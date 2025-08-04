"""Core quantum ML pipeline components."""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from .backends import QuantumExecutor, BackendManager
    BACKENDS_AVAILABLE = True
except ImportError:
    BACKENDS_AVAILABLE = False


class QuantumDevice(Enum):
    """Supported quantum computing backends."""
    
    SIMULATOR = "simulator"
    AWS_BRAKET = "aws_braket"
    IBM_QUANTUM = "ibm_quantum"
    IONQ = "ionq"


class QuantumMLPipeline:
    """Main quantum machine learning pipeline."""
    
    def __init__(
        self,
        circuit: Callable,
        n_qubits: int,
        device: QuantumDevice = QuantumDevice.SIMULATOR,
        **kwargs: Any
    ) -> None:
        """Initialize quantum ML pipeline.
        
        Args:
            circuit: Quantum circuit function
            n_qubits: Number of qubits
            device: Quantum backend device
            **kwargs: Additional backend-specific parameters
        """
        self.circuit = circuit
        self.n_qubits = n_qubits
        self.device = device
        self.config = kwargs
        self._validate_circuit()
        self._initialize_backend()
        
        # Initialize quantum executor if backends are available
        if BACKENDS_AVAILABLE:
            self.quantum_executor = QuantumExecutor()
            self._setup_backend()
        else:
            self.quantum_executor = None
    
    def _validate_circuit(self) -> None:
        """Validate circuit requirements."""
        if not callable(self.circuit):
            raise ValueError("Circuit must be callable")
        if self.n_qubits <= 0:
            raise ValueError("Number of qubits must be positive")
        if self.n_qubits > 30 and self.device == QuantumDevice.SIMULATOR:
            print(f"Warning: {self.n_qubits} qubits may be slow on simulator")
    
    def _initialize_backend(self) -> None:
        """Initialize quantum backend connection."""
        backend_configs = {
            QuantumDevice.SIMULATOR: {'shots': 1024, 'noise_model': None},
            QuantumDevice.AWS_BRAKET: {'shots': 1000, 'device_arn': 'local:braket/braket.local.qubit'},
            QuantumDevice.IBM_QUANTUM: {'shots': 1024, 'backend': 'ibmq_qasm_simulator'},
            QuantumDevice.IONQ: {'shots': 1000, 'backend': 'ionq_simulator'}
        }
        
        self.backend_config = backend_configs.get(self.device, {})
        self.backend_config.update(self.config)
        
    def _setup_backend(self) -> None:
        """Setup quantum backend for real execution."""
        if not BACKENDS_AVAILABLE or not self.quantum_executor:
            return
            
        try:
            # Create backend based on device type
            backend_name = self.quantum_executor.create_backend(
                self.device, 
                **self.config
            )
            self.backend_name = backend_name
        except Exception as e:
            print(f"Warning: Failed to setup backend {self.device}: {e}")
            print("Falling back to simulation")
            self.backend_name = None
    
    def _estimate_parameter_count(self) -> int:
        """Estimate number of trainable parameters."""
        # Rough estimate: 2 parameters per qubit per layer
        layers = self.config.get('layers', 3)
        return 2 * self.n_qubits * layers
    
    def _forward_pass(self, model: "QuantumModel", X: np.ndarray) -> np.ndarray:
        """Execute forward pass through quantum circuit."""
        predictions = []
        
        # Use real quantum backends if available
        if BACKENDS_AVAILABLE and self.quantum_executor and hasattr(self, 'backend_name'):
            try:
                predictions = self._execute_with_quantum_backend(model, X)
            except Exception as e:
                print(f"Warning: Real backend execution failed: {e}")
                print("Falling back to simulation")
                predictions = self._execute_with_simulation(model, X)
        else:
            predictions = self._execute_with_simulation(model, X)
        
        return np.array(predictions)
        
    def _execute_with_quantum_backend(self, model: "QuantumModel", X: np.ndarray) -> List[float]:
        """Execute circuits using real quantum backends."""
        predictions = []
        
        for sample in X:
            # Create circuit description for backend
            circuit = self._create_circuit_description(model.parameters, sample)
            
            # Execute on quantum backend
            try:
                result = self.quantum_executor.execute(
                    circuit,
                    backend_names=[self.backend_name] if hasattr(self, 'backend_name') else None,
                    shots=self.backend_config.get('shots', 1024)
                )
                
                predictions.append(result.expectation_value or 0.0)
                
            except Exception as e:
                print(f"Warning: Backend execution failed for sample: {e}")
                # Fallback to simulation
                result = self._simulate_circuit(model.parameters, sample)
                predictions.append(result)
                
        return predictions
        
    def _execute_with_simulation(self, model: "QuantumModel", X: np.ndarray) -> List[float]:
        """Execute circuits using local simulation."""
        predictions = []
        
        for sample in X:
            if self.device == QuantumDevice.SIMULATOR:
                result = self._simulate_circuit(model.parameters, sample)
            else:
                result = self._execute_on_hardware(model.parameters, sample)
            
            predictions.append(result)
            
        return predictions
        
    def _create_circuit_description(self, params: np.ndarray, x: np.ndarray) -> Dict[str, Any]:
        """Create circuit description for backend execution."""
        # Create a simple parameterized circuit
        gates = []
        
        # Data encoding layer
        for i in range(min(self.n_qubits, len(x))):
            gates.append({
                "type": "ry",
                "qubit": i,
                "angle": x[i] * np.pi  # Feature encoding
            })
            
        # Parameterized layer
        param_idx = 0
        layers = self.config.get('layers', 3)
        
        for layer in range(layers):
            for qubit in range(self.n_qubits):
                if param_idx < len(params):
                    gates.append({
                        "type": "ry",
                        "qubit": qubit,
                        "angle": params[param_idx]
                    })
                    param_idx += 1
                    
                if param_idx < len(params):
                    gates.append({
                        "type": "rz", 
                        "qubit": qubit,
                        "angle": params[param_idx]
                    })
                    param_idx += 1
                    
            # Entangling layer
            for qubit in range(self.n_qubits - 1):
                gates.append({
                    "type": "cnot",
                    "control": qubit,
                    "target": qubit + 1
                })
                
        return {
            "gates": gates,
            "n_qubits": self.n_qubits,
            "measurements": [{"type": "expectation", "wires": 0, "observable": "Z"}]
        }
    
    def _simulate_circuit(self, params: np.ndarray, x: np.ndarray) -> float:
        """Simulate quantum circuit locally."""
        # Simplified quantum circuit simulation
        # In practice, this would use PennyLane, Qiskit, etc.
        state = np.zeros(2**self.n_qubits, dtype=complex)
        state[0] = 1.0  # |00...0> initial state
        
        # Apply parameterized gates (simplified)
        for i, param in enumerate(params[:self.n_qubits]):
            rotation = np.cos(param + x[i % len(x)])
        
        # Measure expectation value
        return np.real(np.sum(state * np.conj(state)))
    
    def _execute_on_hardware(self, params: np.ndarray, x: np.ndarray) -> float:
        """Execute circuit on quantum hardware."""
        # In production, this would submit to actual hardware
        # For now, simulate with added noise
        base_result = self._simulate_circuit(params, x)
        noise_level = {'aws_braket': 0.01, 'ibm_quantum': 0.02, 'ionq': 0.005}.get(self.device.value, 0.0)
        noise = np.random.normal(0, noise_level)
        return base_result + noise
    
    def _compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute loss function."""
        return np.mean((predictions - targets) ** 2)
    
    def _compute_accuracy(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute classification accuracy."""
        pred_labels = (predictions > 0.5).astype(int)
        target_labels = (targets > 0.5).astype(int)
        return np.mean(pred_labels == target_labels)
    
    def _compute_gradients(self, model: "QuantumModel", X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute parameter gradients using parameter shift rule."""
        gradients = np.zeros_like(model.parameters)
        shift = np.pi / 2  # Parameter shift rule
        
        for i in range(len(model.parameters)):
            # Forward shift
            model.parameters[i] += shift
            forward_pred = self._forward_pass(model, X)
            forward_loss = self._compute_loss(forward_pred, y)
            
            # Backward shift
            model.parameters[i] -= 2 * shift
            backward_pred = self._forward_pass(model, X)
            backward_loss = self._compute_loss(backward_pred, y)
            
            # Gradient estimate
            gradients[i] = (forward_loss - backward_loss) / 2
            
            # Restore parameter
            model.parameters[i] += shift
        
        return gradients
    
    def _compute_fidelity(self, model: "QuantumModel") -> float:
        """Compute quantum state fidelity."""
        # Simplified fidelity calculation
        state_norm = np.linalg.norm(model.state_vector)
        return min(1.0, state_norm)
    
    def _evaluate_with_noise(self, model: "QuantumModel", X: np.ndarray, noise_model: str) -> np.ndarray:
        """Evaluate model with specific noise model."""
        noise_levels = {
            'depolarizing': 0.01,
            'amplitude_damping': 0.005,
            'phase_damping': 0.008,
            'bit_flip': 0.002
        }
        
        noise_level = noise_levels.get(noise_model, 0.01)
        
        # Add noise to predictions
        clean_predictions = self._forward_pass(model, X)
        noise = np.random.normal(0, noise_level, len(clean_predictions))
        
        return clean_predictions + noise
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.01,
        track_gradients: bool = False,
    ) -> "QuantumModel":
        """Train quantum ML model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            epochs: Number of training epochs
            learning_rate: Optimizer learning rate
            track_gradients: Whether to track gradient statistics
            
        Returns:
            Trained quantum model
        """
        model = QuantumModel(self.circuit, self.n_qubits)
        
        # Initialize parameters randomly
        n_params = self._estimate_parameter_count()
        model.parameters = np.random.uniform(-np.pi, np.pi, n_params)
        
        # Training metrics tracking
        loss_history = []
        gradient_variances = [] if track_gradients else None
        
        for epoch in range(epochs):
            # Compute predictions and loss
            predictions = self._forward_pass(model, X_train)
            loss = self._compute_loss(predictions, y_train)
            loss_history.append(loss)
            
            # Compute gradients
            gradients = self._compute_gradients(model, X_train, y_train)
            
            if track_gradients:
                gradient_variances.append(np.var(gradients))
            
            # Update parameters
            model.parameters -= learning_rate * gradients
            
            # Log progress
            if epoch % 10 == 0:
                accuracy = self._compute_accuracy(predictions, y_train)
                print(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        
        # Store training metadata
        model.training_history = {
            'loss_history': loss_history,
            'gradient_variances': gradient_variances,
            'final_accuracy': self._compute_accuracy(
                self._forward_pass(model, X_train), y_train
            )
        }
        
        return model
        
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the current quantum backend."""
        info = {
            "device": self.device.value,
            "n_qubits": self.n_qubits,
            "config": self.backend_config,
            "real_backend_available": BACKENDS_AVAILABLE and self.quantum_executor is not None,
        }
        
        if BACKENDS_AVAILABLE and self.quantum_executor:
            try:
                status = self.quantum_executor.get_backend_status()
                available_backends = self.quantum_executor.list_available_backends()
                
                info.update({
                    "available_backends": available_backends,
                    "backend_status": status,
                    "current_backend": getattr(self, 'backend_name', None),
                })
            except Exception as e:
                info["backend_error"] = str(e)
                
        return info
        
    def benchmark_execution(self, test_samples: int = 10) -> Dict[str, Any]:
        """Benchmark the current quantum backend setup."""
        if not BACKENDS_AVAILABLE or not self.quantum_executor:
            return {"error": "Real backends not available"}
            
        # Create test data
        X_test = np.random.random((test_samples, self.n_qubits))
        model = QuantumModel(self.circuit, self.n_qubits)
        model.parameters = np.random.uniform(-np.pi, np.pi, self._estimate_parameter_count())
        
        import time
        
        # Benchmark simulation
        start_time = time.time()
        sim_predictions = self._execute_with_simulation(model, X_test)
        sim_time = time.time() - start_time
        
        # Benchmark real backend if available
        backend_time = None
        backend_predictions = None
        backend_error = None
        
        if hasattr(self, 'backend_name'):
            try:
                start_time = time.time()
                backend_predictions = self._execute_with_quantum_backend(model, X_test)
                backend_time = time.time() - start_time
            except Exception as e:
                backend_error = str(e)
                
        return {
            "test_samples": test_samples,
            "simulation_time": sim_time,
            "backend_time": backend_time,
            "backend_error": backend_error,
            "simulation_available": True,
            "backend_available": hasattr(self, 'backend_name'),
            "predictions_match": (
                np.allclose(sim_predictions, backend_predictions, atol=0.1) 
                if backend_predictions else None
            ),
        }
        
    def evaluate(
        self,
        model: "QuantumModel",
        X_test: np.ndarray,
        y_test: np.ndarray,
        noise_models: Optional[List[str]] = None,
    ) -> "QuantumMetrics":
        """Evaluate quantum model with noise analysis.
        
        Args:
            model: Trained quantum model
            X_test: Test features
            y_test: Test labels
            noise_models: List of noise models to simulate
            
        Returns:
            Evaluation metrics including noise analysis
        """
        # Base evaluation without noise
        predictions = self._forward_pass(model, X_test)
        base_accuracy = self._compute_accuracy(predictions, y_test)
        base_loss = self._compute_loss(predictions, y_test)
        
        # Compute fidelity (quantum state overlap)
        fidelity = self._compute_fidelity(model)
        
        # Gradient variance from training
        gradient_variance = 0.0
        if hasattr(model, 'training_history') and model.training_history.get('gradient_variances'):
            gradient_variance = np.mean(model.training_history['gradient_variances'])
        
        # Noise analysis if requested
        noise_results = {}
        if noise_models:
            for noise_model in noise_models:
                noisy_predictions = self._evaluate_with_noise(model, X_test, noise_model)
                noise_accuracy = self._compute_accuracy(noisy_predictions, y_test)
                noise_results[noise_model] = {
                    'accuracy': noise_accuracy,
                    'degradation': base_accuracy - noise_accuracy
                }
        
        return QuantumMetrics(
            accuracy=base_accuracy,
            loss=base_loss,
            gradient_variance=gradient_variance,
            fidelity=fidelity,
            noise_analysis=noise_results
        )


class QuantumModel:
    """Trained quantum machine learning model."""
    
    def __init__(self, circuit: Callable, n_qubits: int) -> None:
        self.circuit = circuit
        self.n_qubits = n_qubits
        self.parameters: Optional[np.ndarray] = None
        self.training_history: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        
    @property
    def circuit_depth(self) -> int:
        """Get circuit depth based on parameter structure."""
        if self.parameters is None:
            return 0
        # Estimate depth from parameter count
        return max(1, len(self.parameters) // (2 * self.n_qubits))
        
    @property
    def state_vector(self) -> np.ndarray:
        """Get current quantum state vector."""
        if self.parameters is None:
            # Return ground state
            state = np.zeros(2**self.n_qubits, dtype=complex)
            state[0] = 1.0
            return state
        
        # Simulate evolved state (simplified)
        state = np.zeros(2**self.n_qubits, dtype=complex)
        state[0] = 1.0
        
        # Apply parameterized evolution
        for i, param in enumerate(self.parameters[:self.n_qubits]):
            # Simplified rotation (in practice would use quantum simulator)
            phase = np.exp(1j * param)
            state = state * phase
        
        # Normalize
        state = state / np.linalg.norm(state)
        return state
    
    def save_model(self, filepath: str) -> None:
        """Save model parameters and metadata."""
        model_data = {
            'parameters': self.parameters.tolist() if self.parameters is not None else None,
            'n_qubits': self.n_qubits,
            'training_history': self.training_history,
            'metadata': self.metadata
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filepath: str) -> None:
        """Load model parameters and metadata."""
        import json
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        if model_data['parameters']:
            self.parameters = np.array(model_data['parameters'])
        self.n_qubits = model_data['n_qubits']
        self.training_history = model_data.get('training_history', {})
        self.metadata = model_data.get('metadata', {})
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for input data."""
        if self.parameters is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        for sample in X:
            # Simplified prediction using circuit simulation
            result = self._execute_circuit(sample)
            predictions.append(result)
        
        return np.array(predictions)
    
    def _execute_circuit(self, x: np.ndarray) -> float:
        """Execute quantum circuit for single sample."""
        # Simplified circuit execution
        state = self.state_vector
        
        # Apply data encoding
        for i, feature in enumerate(x[:self.n_qubits]):
            encoding_param = feature * np.pi  # Feature encoding
            state = state * np.exp(1j * encoding_param)
        
        # Measure expectation value
        return np.real(np.sum(state * np.conj(state)))


class QuantumMetrics:
    """Quantum-specific evaluation metrics."""
    
    def __init__(
        self,
        accuracy: float,
        gradient_variance: float,
        loss: Optional[float] = None,
        fidelity: Optional[float] = None,
        noise_analysis: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.accuracy = accuracy
        self.loss = loss or 0.0
        self.gradient_variance = gradient_variance
        self.fidelity = fidelity or 1.0
        self.noise_analysis = noise_analysis or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            'accuracy': self.accuracy,
            'loss': self.loss,
            'gradient_variance': self.gradient_variance,
            'fidelity': self.fidelity,
            'noise_analysis': self.noise_analysis
        }
    
    def __str__(self) -> str:
        """String representation of metrics."""
        metrics_str = f"Accuracy: {self.accuracy:.4f}\n"
        metrics_str += f"Loss: {self.loss:.4f}\n"
        metrics_str += f"Gradient Variance: {self.gradient_variance:.6f}\n"
        metrics_str += f"Fidelity: {self.fidelity:.4f}"
        
        if self.noise_analysis:
            metrics_str += "\n\nNoise Analysis:"
            for noise_model, results in self.noise_analysis.items():
                metrics_str += f"\n  {noise_model}: Accuracy={results['accuracy']:.4f}, Degradation={results['degradation']:.4f}"
        
        return metrics_str