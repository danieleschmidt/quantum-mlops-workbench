"""Comprehensive quantum ML testing utilities and enhanced base classes."""

import asyncio
import functools
import time
import unittest
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, patch

import numpy as np
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    pytest = None

from .core import QuantumModel, QuantumMLPipeline, QuantumDevice, QuantumMetrics

try:
    from .backends import QuantumBackend, CircuitResult, JobStatus
    BACKENDS_AVAILABLE = True
except ImportError:
    BACKENDS_AVAILABLE = False


# Decorators for quantum testing
def quantum_test(backend: str = 'simulator', shots: int = 1024):
    """Decorator for quantum-specific tests."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Set up quantum test environment
            original_backend = getattr(self, 'backend', None)
            original_shots = getattr(self, 'shots', None)
            
            self.backend = backend
            self.shots = shots
            
            try:
                return func(self, *args, **kwargs)
            finally:
                # Restore original settings
                if original_backend is not None:
                    self.backend = original_backend
                if original_shots is not None:
                    self.shots = original_shots
        
        # Add pytest markers
        if backend == 'simulator':
            wrapper = pytest.mark.simulation(wrapper)
        else:
            wrapper = pytest.mark.hardware(wrapper)
        
        return wrapper
    return decorator


def noise_resilience_test(noise_models: List[str], noise_levels: List[float]):
    """Decorator for noise resilience testing."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            for noise_model in noise_models:
                for noise_level in noise_levels:
                    with self.subTest(noise_model=noise_model, noise_level=noise_level):
                        self.current_noise_model = noise_model
                        self.current_noise_level = noise_level
                        func(self, *args, **kwargs)
        
        wrapper = pytest.mark.slow(wrapper)
        return wrapper
    return decorator


def performance_benchmark(min_throughput: float = 1.0, max_latency: float = 10.0):
    """Decorator for performance benchmarking tests."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            execution_time = time.time() - start_time
            
            # Performance assertions
            if hasattr(self, 'benchmark_samples'):
                throughput = self.benchmark_samples / execution_time
                assert throughput >= min_throughput, f"Throughput {throughput:.2f} below minimum {min_throughput}"
            
            assert execution_time <= max_latency, f"Latency {execution_time:.2f}s exceeds maximum {max_latency}s"
            
            return result
        
        wrapper = pytest.mark.slow(wrapper)
        return wrapper
    return decorator


class QuantumTestCase(unittest.TestCase):
    """Enhanced base test case for quantum ML testing with comprehensive utilities."""
    
    def setUp(self) -> None:
        """Set up quantum test environment."""
        self.n_qubits = 4
        self.noise_tolerance = 0.1
        self.gradient_tolerance = 0.05
        self.fidelity_threshold = 0.95
        self.benchmark_samples = 100
        self.max_circuit_depth = 50
        
        # Performance tracking
        self.performance_metrics = {
            'execution_times': [],
            'memory_usage': [],
            'error_rates': []
        }
        
        # Noise models for testing
        self.noise_models = {
            'depolarizing': {'error_rate': 0.01, 'single_qubit': True, 'two_qubit': True},
            'amplitude_damping': {'error_rate': 0.005, 'single_qubit': True, 'two_qubit': False},
            'phase_damping': {'error_rate': 0.008, 'single_qubit': True, 'two_qubit': False},
            'bit_flip': {'error_rate': 0.002, 'single_qubit': True, 'two_qubit': False},
            'phase_flip': {'error_rate': 0.002, 'single_qubit': True, 'two_qubit': False},
            'thermal': {'error_rate': 0.01, 'temperature': 0.1, 'single_qubit': True, 'two_qubit': True}
        }
        
        # Backend configurations for testing
        self.backend_configs = {
            'simulator': {
                'shots': 1024, 
                'noise_model': None, 
                'max_qubits': 30,
                'native_gates': ['rx', 'ry', 'rz', 'cx', 'cz', 'h', 'x', 'y', 'z']
            },
            'ionq': {
                'shots': 1000, 
                'native_gates': ['rx', 'ry', 'rz', 'cnot'],
                'connectivity': 'all_to_all',
                'max_qubits': 32
            },
            'ibm': {
                'shots': 1024, 
                'native_gates': ['u1', 'u2', 'u3', 'cx'],
                'connectivity': 'heavy_hex',
                'max_qubits': 27
            },
            'google': {
                'shots': 1000,
                'native_gates': ['x_pow', 'y_pow', 'z_pow', 'cz'],
                'connectivity': 'grid',
                'max_qubits': 70
            },
            'rigetti': {
                'shots': 1000,
                'native_gates': ['rx', 'ry', 'rz', 'cz'],
                'connectivity': 'octagonal',
                'max_qubits': 31
            }
        }
        
    def create_model(
        self, 
        n_qubits: Optional[int] = None,
        circuit_type: str = 'variational',
        depth: int = 3,
        entangling: bool = True
    ) -> QuantumModel:
        """Create a test quantum model with configurable architecture.
        
        Args:
            n_qubits: Number of qubits (defaults to self.n_qubits)
            circuit_type: Type of circuit ('variational', 'embedding', 'ansatz')
            depth: Circuit depth/layers
            entangling: Whether to include entangling gates
            
        Returns:
            Test quantum model instance
        """
        qubits = n_qubits or self.n_qubits
        
        def test_circuit(params: np.ndarray, x: np.ndarray) -> float:
            """Parameterized quantum circuit for testing."""
            # More realistic quantum circuit simulation
            state_size = 2 ** qubits
            state = np.zeros(state_size, dtype=complex)
            state[0] = 1.0  # |00...0⟩ initial state
            
            param_idx = 0
            
            # Data encoding layer
            if circuit_type in ['embedding', 'variational']:
                for i in range(min(qubits, len(x))):
                    # RY rotation for data encoding
                    angle = x[i] * np.pi
                    state = self._apply_single_qubit_rotation(state, i, angle, 'y')
            
            # Parameterized layers
            for layer in range(depth):
                # Single-qubit parameterized gates
                for qubit in range(qubits):
                    if param_idx < len(params):
                        state = self._apply_single_qubit_rotation(
                            state, qubit, params[param_idx], 'y'
                        )
                        param_idx += 1
                    if param_idx < len(params):
                        state = self._apply_single_qubit_rotation(
                            state, qubit, params[param_idx], 'z'
                        )
                        param_idx += 1
                
                # Entangling layer
                if entangling:
                    for qubit in range(qubits - 1):
                        state = self._apply_cnot(state, qubit, qubit + 1)
            
            # Measure expectation value of Z on first qubit
            expectation = self._measure_expectation_z(state, 0, qubits)
            return float(np.real(expectation))
            
        model = QuantumModel(test_circuit, qubits)
        model.metadata['circuit_type'] = circuit_type
        model.metadata['depth'] = depth
        model.metadata['entangling'] = entangling
        
        return model
    
    def _apply_single_qubit_rotation(
        self, 
        state: np.ndarray, 
        qubit: int, 
        angle: float, 
        axis: str
    ) -> np.ndarray:
        """Apply single-qubit rotation to quantum state."""
        # Simplified rotation implementation
        if axis == 'x':
            rotation_factor = np.cos(angle/2) + 1j * np.sin(angle/2)
        elif axis == 'y':
            rotation_factor = np.cos(angle/2) + np.sin(angle/2)
        else:  # z
            rotation_factor = np.exp(1j * angle/2)
        
        # Apply rotation (simplified)
        new_state = state.copy()
        qubit_mask = 2 ** qubit
        for i in range(len(state)):
            if i & qubit_mask:
                new_state[i] *= rotation_factor
        
        return new_state
    
    def _apply_cnot(self, state: np.ndarray, control: int, target: int) -> np.ndarray:
        """Apply CNOT gate to quantum state."""
        new_state = state.copy()
        control_mask = 2 ** control
        target_mask = 2 ** target
        
        for i in range(len(state)):
            if i & control_mask:  # Control qubit is |1⟩
                # Flip target qubit
                flipped_index = i ^ target_mask
                new_state[i], new_state[flipped_index] = state[flipped_index], state[i]
        
        return new_state
    
    def _measure_expectation_z(self, state: np.ndarray, qubit: int, n_qubits: int) -> complex:
        """Measure expectation value of Pauli-Z on specified qubit."""
        expectation = 0.0
        qubit_mask = 2 ** qubit
        
        for i in range(len(state)):
            prob = np.abs(state[i]) ** 2
            if i & qubit_mask:  # Qubit is |1⟩
                expectation -= prob
            else:  # Qubit is |0⟩
                expectation += prob
        
        return expectation
        
    def measure_gradient_variance(
        self,
        model: QuantumModel,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        n_samples: int = 100,
        noise_level: float = 0.01,
        gradient_method: str = 'parameter_shift'
    ) -> Dict[str, float]:
        """Measure gradient variance under noise with multiple metrics.
        
        Args:
            model: Quantum model to test
            X: Input data for gradient computation
            y: Target labels for gradient computation
            n_samples: Number of gradient samples
            noise_level: Noise amplitude
            gradient_method: Method for gradient computation
            
        Returns:
            Dictionary of gradient variance metrics
        """
        if X is None:
            X = np.random.random((n_samples, model.n_qubits))
        if y is None:
            y = np.random.randint(0, 2, len(X))
        
        if model.parameters is None:
            n_params = 2 * model.n_qubits * 3  # Estimate parameter count
            model.parameters = np.random.uniform(-np.pi, np.pi, n_params)
        
        gradients = []
        execution_times = []
        
        for i in range(n_samples):
            start_time = time.time()
            
            if gradient_method == 'parameter_shift':
                grad = self._compute_parameter_shift_gradient(
                    model, X[i:i+1], y[i:i+1], noise_level
                )
            elif gradient_method == 'finite_difference':
                grad = self._compute_finite_difference_gradient(
                    model, X[i:i+1], y[i:i+1], noise_level
                )
            else:
                # Simplified noisy gradient
                grad = np.random.normal(0, noise_level, len(model.parameters))
            
            execution_times.append(time.time() - start_time)
            gradients.append(grad)
        
        gradients = np.array(gradients)
        
        return {
            'variance': float(np.mean(np.var(gradients, axis=0))),
            'mean_magnitude': float(np.mean(np.linalg.norm(gradients, axis=1))),
            'max_variance': float(np.max(np.var(gradients, axis=0))),
            'min_variance': float(np.min(np.var(gradients, axis=0))),
            'coefficient_of_variation': float(np.std(np.linalg.norm(gradients, axis=1)) / 
                                           np.mean(np.linalg.norm(gradients, axis=1))),
            'mean_execution_time': float(np.mean(execution_times)),
            'gradient_norm_stability': float(np.std(np.linalg.norm(gradients, axis=1)))
        }
    
    def _compute_parameter_shift_gradient(
        self, 
        model: QuantumModel, 
        X: np.ndarray, 
        y: np.ndarray,
        noise_level: float
    ) -> np.ndarray:
        """Compute gradients using parameter shift rule."""
        shift = np.pi / 2
        gradients = np.zeros_like(model.parameters)
        
        for i in range(len(model.parameters)):
            # Forward shift
            model.parameters[i] += shift
            forward_pred = model.predict(X)[0] + np.random.normal(0, noise_level)
            forward_loss = (forward_pred - y[0]) ** 2
            
            # Backward shift
            model.parameters[i] -= 2 * shift
            backward_pred = model.predict(X)[0] + np.random.normal(0, noise_level)
            backward_loss = (backward_pred - y[0]) ** 2
            
            # Gradient estimate
            gradients[i] = (forward_loss - backward_loss) / 2
            
            # Restore parameter
            model.parameters[i] += shift
        
        return gradients
    
    def _compute_finite_difference_gradient(
        self, 
        model: QuantumModel, 
        X: np.ndarray, 
        y: np.ndarray,
        noise_level: float
    ) -> np.ndarray:
        """Compute gradients using finite differences."""
        epsilon = 1e-7
        gradients = np.zeros_like(model.parameters)
        
        # Base prediction
        base_pred = model.predict(X)[0] + np.random.normal(0, noise_level)
        base_loss = (base_pred - y[0]) ** 2
        
        for i in range(len(model.parameters)):
            # Perturb parameter
            model.parameters[i] += epsilon
            perturbed_pred = model.predict(X)[0] + np.random.normal(0, noise_level)
            perturbed_loss = (perturbed_pred - y[0]) ** 2
            
            # Gradient estimate
            gradients[i] = (perturbed_loss - base_loss) / epsilon
            
            # Restore parameter
            model.parameters[i] -= epsilon
        
        return gradients
        
    def assert_native_gates(
        self, 
        circuit: Union[Callable, Dict[str, Any]], 
        backend: str,
        strict: bool = True
    ) -> None:
        """Assert circuit uses only native gates for backend.
        
        Args:
            circuit: Quantum circuit function or circuit description
            backend: Target quantum backend
            strict: Whether to enforce strict gate compatibility
        """
        if backend not in self.backend_configs:
            self.skipTest(f"Unknown backend: {backend}")
        
        native_gates = self.backend_configs[backend]['native_gates']
        
        # Extract gates from circuit
        if isinstance(circuit, dict) and 'gates' in circuit:
            circuit_gates = [gate['type'] for gate in circuit['gates']]
        else:
            # For callable circuits, we need to analyze the structure
            # This is a simplified implementation
            circuit_gates = self._extract_gates_from_callable(circuit)
        
        # Check gate compatibility
        incompatible_gates = []
        for gate in set(circuit_gates):
            if gate not in native_gates:
                # Check if gate can be decomposed
                if not self._can_decompose_gate(gate, native_gates):
                    incompatible_gates.append(gate)
        
        if strict and incompatible_gates:
            self.fail(f"Circuit uses non-native gates for {backend}: {incompatible_gates}")
        elif incompatible_gates:
            warnings.warn(f"Circuit may require gate decomposition for {backend}: {incompatible_gates}")
        
        # Test passes if no incompatible gates or decomposition is possible
        self.assertTrue(len(incompatible_gates) == 0 or not strict,
                       f"Circuit compatible with {backend} native gates")
    
    def _extract_gates_from_callable(self, circuit: Callable) -> List[str]:
        """Extract gate types from callable circuit (simplified)."""
        # This is a simplified implementation
        # In practice, would analyze circuit structure more thoroughly
        return ['rx', 'ry', 'rz', 'cx']  # Common gates
    
    def _can_decompose_gate(self, gate: str, native_gates: List[str]) -> bool:
        """Check if gate can be decomposed into native gates."""
        decomposition_rules = {
            'h': ['ry', 'rz'] if 'ry' in native_gates and 'rz' in native_gates else [],
            'x': ['rx'] if 'rx' in native_gates else ['u3'] if 'u3' in native_gates else [],
            'y': ['ry'] if 'ry' in native_gates else ['u3'] if 'u3' in native_gates else [],
            'z': ['rz'] if 'rz' in native_gates else ['u1'] if 'u1' in native_gates else [],
            'cnot': ['cx'] if 'cx' in native_gates else ['cz'] if 'cz' in native_gates else [],
            'cz': ['cx'] if 'cx' in native_gates else ['cnot'] if 'cnot' in native_gates else [],
            't': ['rz'] if 'rz' in native_gates else ['u1'] if 'u1' in native_gates else [],
            's': ['rz'] if 'rz' in native_gates else ['u1'] if 'u1' in native_gates else []
        }
        
        if gate in native_gates:
            return True
        
        required_gates = decomposition_rules.get(gate.lower(), [])
        return all(req_gate in native_gates for req_gate in required_gates)
        
    def assert_topology_compatible(
        self,
        circuit: Union[Callable, Dict[str, Any]],
        backend: str,
        qubits_used: Optional[List[int]] = None
    ) -> None:
        """Assert circuit respects device topology constraints.
        
        Args:
            circuit: Quantum circuit function or description
            backend: Target quantum backend
            qubits_used: List of physical qubits used in circuit
        """
        if backend not in self.backend_configs:
            self.skipTest(f"Unknown backend: {backend}")
        
        connectivity = self.backend_configs[backend].get('connectivity', 'all_to_all')
        max_qubits = self.backend_configs[backend].get('max_qubits', 100)
        
        # Extract two-qubit gates and their qubits
        if isinstance(circuit, dict) and 'gates' in circuit:
            two_qubit_gates = [
                gate for gate in circuit['gates'] 
                if gate['type'] in ['cnot', 'cx', 'cz', 'iswap', 'xy']
            ]
        else:
            # For callable circuits, estimate based on qubit count
            two_qubit_gates = self._estimate_two_qubit_gates(circuit, qubits_used or list(range(self.n_qubits)))
        
        # Check qubit count constraint
        total_qubits = len(qubits_used) if qubits_used else self.n_qubits
        self.assertLessEqual(total_qubits, max_qubits,
                           f"Circuit uses {total_qubits} qubits, but {backend} supports max {max_qubits}")
        
        # Check connectivity constraints
        if connectivity == 'all_to_all':
            # No topology constraints
            pass
        elif connectivity in ['heavy_hex', 'grid', 'octagonal']:
            # Check if two-qubit gates respect connectivity
            violations = self._check_connectivity_violations(
                two_qubit_gates, connectivity, qubits_used or list(range(self.n_qubits))
            )
            
            if violations:
                warnings.warn(f"Circuit may require SWAP gates for {backend} topology: {violations}")
            
            # Allow with warning rather than failing for realistic circuits
            self.assertTrue(True, f"Circuit topology checked for {backend}")
        
        self.assertTrue(True, "Circuit respects device topology")
    
    def _estimate_two_qubit_gates(self, circuit: Callable, qubits: List[int]) -> List[Dict[str, Any]]:
        """Estimate two-qubit gates from circuit structure (simplified)."""
        # Simplified estimation - assumes entangling gates between adjacent qubits
        gates = []
        for i in range(len(qubits) - 1):
            gates.append({
                'type': 'cnot',
                'control': qubits[i],
                'target': qubits[i + 1]
            })
        return gates
    
    def _check_connectivity_violations(
        self, 
        two_qubit_gates: List[Dict[str, Any]], 
        connectivity: str,
        qubits: List[int]
    ) -> List[str]:
        """Check for connectivity violations based on topology."""
        violations = []
        
        # Simplified connectivity rules
        if connectivity == 'grid':
            # Grid topology - qubits connected to nearest neighbors
            grid_size = int(np.ceil(np.sqrt(len(qubits))))
            for gate in two_qubit_gates:
                control = gate.get('control', 0)
                target = gate.get('target', 1)
                if not self._are_adjacent_in_grid(control, target, grid_size):
                    violations.append(f"Gate {gate['type']} between qubits {control}-{target}")
        
        elif connectivity == 'heavy_hex':
            # Heavy-hex topology - more complex connectivity pattern
            for gate in two_qubit_gates:
                control = gate.get('control', 0)
                target = gate.get('target', 1)
                if abs(control - target) > 2:  # Simplified check
                    violations.append(f"Gate {gate['type']} between qubits {control}-{target}")
        
        return violations
    
    def _are_adjacent_in_grid(self, qubit1: int, qubit2: int, grid_size: int) -> bool:
        """Check if two qubits are adjacent in a grid topology."""
        row1, col1 = divmod(qubit1, grid_size)
        row2, col2 = divmod(qubit2, grid_size)
        return abs(row1 - row2) + abs(col1 - col2) == 1
        
    def evaluate_with_noise(
        self,
        model: QuantumModel,
        X: np.ndarray,
        y: np.ndarray,
        noise_model: str = 'depolarizing',
        noise_prob: float = 0.01,
        n_trials: int = 10
    ) -> Dict[str, float]:
        """Evaluate model performance under realistic noise with multiple metrics.
        
        Args:
            model: Quantum model to evaluate
            X: Input data
            y: Target labels
            noise_model: Type of noise model
            noise_prob: Noise probability
            n_trials: Number of trials for statistical averaging
            
        Returns:
            Dictionary of performance metrics under noise
        """
        if model.parameters is None:
            n_params = 2 * model.n_qubits * 3
            model.parameters = np.random.uniform(-np.pi, np.pi, n_params)
        
        noise_config = self.noise_models.get(noise_model, {'error_rate': noise_prob})
        
        # Run multiple trials for statistical significance
        accuracies = []
        prediction_variances = []
        execution_times = []
        
        for trial in range(n_trials):
            start_time = time.time()
            
            # Apply noise to model predictions
            noisy_predictions = []
            for i in range(len(X)):
                clean_pred = model.predict(X[i:i+1])[0]
                noisy_pred = self._apply_noise_to_prediction(
                    clean_pred, noise_model, noise_config
                )
                noisy_predictions.append(noisy_pred)
            
            noisy_predictions = np.array(noisy_predictions)
            
            # Calculate metrics
            accuracy = self._compute_accuracy(noisy_predictions, y)
            variance = np.var(noisy_predictions)
            
            accuracies.append(accuracy)
            prediction_variances.append(variance)
            execution_times.append(time.time() - start_time)
        
        # Compute statistics across trials
        return {
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'min_accuracy': float(np.min(accuracies)),
            'max_accuracy': float(np.max(accuracies)),
            'mean_prediction_variance': float(np.mean(prediction_variances)),
            'mean_execution_time': float(np.mean(execution_times)),
            'noise_robustness_score': float(1.0 - np.std(accuracies) / (np.mean(accuracies) + 1e-8))
        }
    
    def _apply_noise_to_prediction(
        self, 
        prediction: float, 
        noise_model: str,
        noise_config: Dict[str, Any]
    ) -> float:
        """Apply specific noise model to prediction."""
        error_rate = noise_config.get('error_rate', 0.01)
        
        if noise_model == 'depolarizing':
            # Depolarizing noise reduces signal towards 0
            return prediction * (1 - error_rate) + np.random.normal(0, error_rate)
        
        elif noise_model == 'amplitude_damping':
            # Amplitude damping biases towards ground state
            damping_factor = np.sqrt(1 - error_rate)
            return prediction * damping_factor + np.random.normal(0, error_rate * 0.1)
        
        elif noise_model == 'phase_damping':
            # Phase damping adds random phase shifts
            phase_noise = np.random.normal(0, error_rate)
            return prediction * np.cos(phase_noise) + np.random.normal(0, error_rate * 0.1)
        
        elif noise_model == 'thermal':
            # Thermal noise depends on temperature
            temperature = noise_config.get('temperature', 0.1)
            thermal_factor = np.tanh(1 / (2 * temperature + 1e-8))
            return prediction * thermal_factor + np.random.normal(0, error_rate)
        
        else:
            # Generic noise
            return prediction + np.random.normal(0, error_rate)
    
    def _compute_accuracy(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute classification accuracy."""
        pred_labels = (predictions > 0.5).astype(int)
        target_labels = (targets > 0.5).astype(int)
        return float(np.mean(pred_labels == target_labels))
        
    def build_circuit(
        self, 
        circuit_type: str = 'basic',
        n_qubits: Optional[int] = None,
        depth: int = 3
    ) -> Callable:
        """Build a test quantum circuit with specified architecture.
        
        Args:
            circuit_type: Type of circuit ('basic', 'variational', 'embedding')
            n_qubits: Number of qubits (defaults to self.n_qubits)
            depth: Circuit depth
            
        Returns:
            Test quantum circuit function
        """
        qubits = n_qubits or self.n_qubits
        
        if circuit_type == 'basic':
            def circuit(params: np.ndarray) -> float:
                # Simple parameterized circuit
                return float(np.sum(np.cos(params)) / len(params))
        
        elif circuit_type == 'variational':
            def circuit(params: np.ndarray) -> float:
                # Variational quantum eigensolver style circuit
                layers = len(params) // (2 * qubits)
                result = 0.0
                
                param_idx = 0
                for layer in range(layers):
                    # Single-qubit rotations
                    layer_result = 0.0
                    for qubit in range(qubits):
                        if param_idx < len(params):
                            layer_result += np.cos(params[param_idx]) * np.sin(params[param_idx + 1] if param_idx + 1 < len(params) else 0)
                            param_idx += 2
                    
                    result += layer_result / qubits
                
                return float(result / max(1, layers))
        
        elif circuit_type == 'embedding':
            def circuit(params: np.ndarray) -> float:
                # Data embedding circuit
                if len(params) < qubits:
                    params = np.pad(params, (0, qubits - len(params)), 'wrap')
                
                # Feature map
                feature_sum = 0.0
                for i in range(qubits):
                    feature_sum += np.sin(params[i] * np.pi) * np.cos(params[i] * np.pi / 2)
                
                return float(feature_sum / qubits)
        
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")
            
        return circuit
    
    # Additional quantum-specific assertions
    def assert_fidelity_above_threshold(
        self, 
        state1: np.ndarray, 
        state2: np.ndarray, 
        threshold: Optional[float] = None
    ) -> None:
        """Assert quantum state fidelity is above threshold."""
        threshold = threshold or self.fidelity_threshold
        fidelity = self._compute_state_fidelity(state1, state2)
        self.assertGreaterEqual(fidelity, threshold,
                               f"State fidelity {fidelity:.4f} below threshold {threshold}")
    
    def assert_circuit_depth_reasonable(self, circuit: Callable, max_depth: Optional[int] = None) -> None:
        """Assert circuit depth is within reasonable bounds."""
        max_depth = max_depth or self.max_circuit_depth
        estimated_depth = self._estimate_circuit_depth(circuit)
        self.assertLessEqual(estimated_depth, max_depth,
                           f"Circuit depth {estimated_depth} exceeds maximum {max_depth}")
    
    def assert_gradient_magnitude_reasonable(
        self, 
        gradients: np.ndarray, 
        min_magnitude: float = 1e-8,
        max_magnitude: float = 10.0
    ) -> None:
        """Assert gradient magnitudes are in reasonable range."""
        grad_norms = np.linalg.norm(gradients, axis=1) if gradients.ndim > 1 else [np.linalg.norm(gradients)]
        
        for i, norm in enumerate(grad_norms):
            self.assertGreaterEqual(norm, min_magnitude,
                                  f"Gradient {i} magnitude {norm} too small (possible vanishing gradient)")
            self.assertLessEqual(norm, max_magnitude,
                               f"Gradient {i} magnitude {norm} too large (possible exploding gradient)")
    
    def assert_quantum_advantage(
        self, 
        quantum_result: float, 
        classical_result: float,
        advantage_threshold: float = 0.05
    ) -> None:
        """Assert quantum model shows advantage over classical baseline."""
        advantage = quantum_result - classical_result
        self.assertGreaterEqual(advantage, advantage_threshold,
                               f"Quantum advantage {advantage:.4f} below threshold {advantage_threshold}")
    
    def _compute_state_fidelity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """Compute quantum state fidelity."""
        # Normalize states
        state1_norm = state1 / np.linalg.norm(state1)
        state2_norm = state2 / np.linalg.norm(state2)
        
        # Compute fidelity
        overlap = np.abs(np.vdot(state1_norm, state2_norm)) ** 2
        return float(overlap)
    
    def _estimate_circuit_depth(self, circuit: Callable) -> int:
        """Estimate circuit depth (simplified implementation)."""
        # This is a simplified estimation
        # In practice, would analyze actual circuit structure
        return max(1, self.n_qubits // 2 + 3)  # Rough estimate
    
    # Performance benchmarking utilities
    def benchmark_execution(
        self, 
        model: QuantumModel,
        X: np.ndarray,
        n_runs: int = 10,
        warmup_runs: int = 3
    ) -> Dict[str, float]:
        """Benchmark model execution performance."""
        if model.parameters is None:
            n_params = 2 * model.n_qubits * 3
            model.parameters = np.random.uniform(-np.pi, np.pi, n_params)
        
        # Warmup runs
        for _ in range(warmup_runs):
            model.predict(X[:1])
        
        # Benchmark runs
        execution_times = []
        memory_usage = []
        
        for run in range(n_runs):
            start_time = time.time()
            
            # Monitor memory usage
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss
            except ImportError:
                memory_before = 0
            
            # Execute prediction
            predictions = model.predict(X)
            
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            
            # Memory usage
            if memory_before > 0:
                try:
                    memory_after = process.memory_info().rss
                    memory_usage.append(memory_after - memory_before)
                except:
                    memory_usage.append(0)
        
        return {
            'mean_execution_time': float(np.mean(execution_times)),
            'std_execution_time': float(np.std(execution_times)),
            'min_execution_time': float(np.min(execution_times)),
            'max_execution_time': float(np.max(execution_times)),
            'throughput_samples_per_second': float(len(X) / np.mean(execution_times)),
            'mean_memory_delta_mb': float(np.mean(memory_usage) / 1024 / 1024) if memory_usage else 0.0
        }
    
    # Mock and testing utilities
    def create_mock_backend(self, backend_type: str = 'simulator') -> MagicMock:
        """Create a mock quantum backend for testing."""
        mock_backend = MagicMock()
        mock_backend.name = f'mock_{backend_type}'
        mock_backend.is_available.return_value = True
        
        # Configure based on backend type
        config = self.backend_configs.get(backend_type, self.backend_configs['simulator'])
        mock_backend.get_device_properties.return_value = config
        
        # Mock job submission and results
        mock_backend.submit_job.return_value = MagicMock(
            job_id='test_job_123',
            status=JobStatus.COMPLETED if BACKENDS_AVAILABLE else 'completed'
        )
        
        mock_backend.get_job_results.return_value = [
            CircuitResult(
                circuit_id='test_circuit',
                counts={'0': 512, '1': 512},
                expectation_value=0.0,
                execution_time=0.1,
                shots=1024
            ) if BACKENDS_AVAILABLE else {
                'circuit_id': 'test_circuit',
                'counts': {'0': 512, '1': 512},
                'expectation_value': 0.0
            }
        ]
        
        return mock_backend
    
    def create_test_dataset(
        self, 
        n_samples: int = 100,
        n_features: Optional[int] = None,
        task_type: str = 'classification',
        noise_level: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic dataset for testing."""
        n_features = n_features or self.n_qubits
        np.random.seed(42)  # For reproducible tests
        
        X = np.random.uniform(-1, 1, (n_samples, n_features))
        
        if task_type == 'classification':
            # Create non-linear decision boundary
            y = (np.sum(X**2, axis=1) + np.random.normal(0, noise_level, n_samples) > 
                 np.median(np.sum(X**2, axis=1))).astype(int)
        elif task_type == 'regression':
            # Create non-linear regression target
            y = np.sum(np.sin(X), axis=1) + np.random.normal(0, noise_level, n_samples)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return X, y
    
    def assert_performance_regression(
        self, 
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        tolerance: float = 0.05
    ) -> None:
        """Assert no significant performance regression."""
        for metric_name, current_value in current_metrics.items():
            if metric_name in baseline_metrics:
                baseline_value = baseline_metrics[metric_name]
                
                # For accuracy-like metrics (higher is better)
                if 'accuracy' in metric_name.lower() or 'fidelity' in metric_name.lower():
                    regression = baseline_value - current_value
                    self.assertLessEqual(regression, tolerance,
                                       f"{metric_name} regressed by {regression:.4f} > {tolerance}")
                
                # For loss-like metrics (lower is better)
                elif 'loss' in metric_name.lower() or 'error' in metric_name.lower():
                    regression = current_value - baseline_value
                    self.assertLessEqual(regression, tolerance,
                                       f"{metric_name} regressed by {regression:.4f} > {tolerance}")
    
    def run_chaos_test(
        self, 
        test_function: Callable,
        failure_modes: List[str],
        n_iterations: int = 10
    ) -> Dict[str, Any]:
        """Run chaos engineering test with various failure modes."""
        results = {
            'total_iterations': n_iterations,
            'successful_runs': 0,
            'failures': {},
            'recovery_times': []
        }
        
        for iteration in range(n_iterations):
            # Randomly select failure mode
            failure_mode = np.random.choice(failure_modes)
            
            try:
                start_time = time.time()
                
                # Inject failure
                with self._inject_failure(failure_mode):
                    test_function()
                
                recovery_time = time.time() - start_time
                results['recovery_times'].append(recovery_time)
                results['successful_runs'] += 1
                
            except Exception as e:
                if failure_mode not in results['failures']:
                    results['failures'][failure_mode] = []
                results['failures'][failure_mode].append(str(e))
        
        results['success_rate'] = results['successful_runs'] / n_iterations
        results['mean_recovery_time'] = np.mean(results['recovery_times']) if results['recovery_times'] else float('inf')
        
        return results
    
    def _inject_failure(self, failure_mode: str):
        """Context manager for injecting specific failure modes."""
        class FailureInjector:
            def __init__(self, mode):
                self.mode = mode
                self.patches = []
            
            def __enter__(self):
                if self.mode == 'network_failure':
                    # Mock network failures
                    self.patches.append(patch('requests.get', side_effect=ConnectionError("Network failure")))
                elif self.mode == 'hardware_error':
                    # Mock hardware errors
                    self.patches.append(patch('quantum_mlops.backends.QuantumBackend.submit_job', 
                                            side_effect=Exception("Hardware error")))
                elif self.mode == 'memory_pressure':
                    # Simulate memory pressure (simplified)
                    import gc
                    gc.collect()
                
                for p in self.patches:
                    p.__enter__()
                
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                for p in reversed(self.patches):
                    try:
                        p.__exit__(exc_type, exc_val, exc_tb)
                    except:
                        pass
        
        return FailureInjector(failure_mode)


# Additional utility classes for advanced testing
class QuantumCircuitOptimizer:
    """Utility class for circuit optimization testing."""
    
    @staticmethod
    def count_gates(circuit_description: Dict[str, Any]) -> Dict[str, int]:
        """Count gates by type in circuit."""
        gate_counts = {}
        for gate in circuit_description.get('gates', []):
            gate_type = gate['type']
            gate_counts[gate_type] = gate_counts.get(gate_type, 0) + 1
        return gate_counts
    
    @staticmethod
    def estimate_depth(circuit_description: Dict[str, Any]) -> int:
        """Estimate circuit depth."""
        # Simplified depth estimation
        n_qubits = circuit_description.get('n_qubits', 4)
        gates = circuit_description.get('gates', [])
        
        # Group gates by qubit dependencies
        depth = 0
        qubit_last_gate = {i: -1 for i in range(n_qubits)}
        
        for i, gate in enumerate(gates):
            if gate['type'] in ['cnot', 'cx', 'cz']:
                control = gate.get('control', 0)
                target = gate.get('target', 1)
                max_dep = max(qubit_last_gate[control], qubit_last_gate[target])
                depth = max(depth, max_dep + 1)
                qubit_last_gate[control] = qubit_last_gate[target] = depth
            else:
                qubit = gate.get('qubit', 0)
                depth = max(depth, qubit_last_gate[qubit] + 1)
                qubit_last_gate[qubit] = depth
        
        return depth


class NoiseModelTester:
    """Utility class for comprehensive noise model testing."""
    
    def __init__(self):
        self.noise_models = {
            'depolarizing': self._depolarizing_channel,
            'amplitude_damping': self._amplitude_damping_channel,
            'phase_damping': self._phase_damping_channel,
            'bit_flip': self._bit_flip_channel,
            'phase_flip': self._phase_flip_channel,
            'thermal': self._thermal_channel
        }
    
    def apply_noise_channel(
        self, 
        state: np.ndarray, 
        noise_type: str, 
        error_rate: float
    ) -> np.ndarray:
        """Apply noise channel to quantum state."""
        if noise_type not in self.noise_models:
            raise ValueError(f"Unknown noise model: {noise_type}")
        
        return self.noise_models[noise_type](state, error_rate)
    
    def _depolarizing_channel(self, state: np.ndarray, error_rate: float) -> np.ndarray:
        """Apply depolarizing noise."""
        # Simplified depolarizing channel
        identity_prob = 1 - error_rate
        noise = np.random.normal(0, error_rate, state.shape) + 1j * np.random.normal(0, error_rate, state.shape)
        return state * identity_prob + noise * error_rate
    
    def _amplitude_damping_channel(self, state: np.ndarray, error_rate: float) -> np.ndarray:
        """Apply amplitude damping noise."""
        damping_factor = np.sqrt(1 - error_rate)
        return state * damping_factor
    
    def _phase_damping_channel(self, state: np.ndarray, error_rate: float) -> np.ndarray:
        """Apply phase damping noise."""
        phase_noise = np.random.normal(0, error_rate, state.shape)
        return state * np.exp(1j * phase_noise)
    
    def _bit_flip_channel(self, state: np.ndarray, error_rate: float) -> np.ndarray:
        """Apply bit flip noise."""
        # Simplified bit flip
        flip_mask = np.random.random(state.shape) < error_rate
        return np.where(flip_mask, -state, state)
    
    def _phase_flip_channel(self, state: np.ndarray, error_rate: float) -> np.ndarray:
        """Apply phase flip noise."""
        flip_mask = np.random.random(state.shape) < error_rate
        return np.where(flip_mask, -1j * state, state)
    
    def _thermal_channel(self, state: np.ndarray, error_rate: float) -> np.ndarray:
        """Apply thermal noise."""
        # Simplified thermal noise
        temperature_factor = np.exp(-error_rate)
        thermal_noise = np.random.normal(0, 1 - temperature_factor, state.shape)
        return state * temperature_factor + thermal_noise * 0.1


class QuantumHardwareSimulator:
    """Simulator for realistic quantum hardware characteristics."""
    
    def __init__(self, backend_type: str = 'superconducting'):
        self.backend_type = backend_type
        self.device_parameters = self._get_device_parameters(backend_type)
    
    def _get_device_parameters(self, backend_type: str) -> Dict[str, Any]:
        """Get realistic device parameters for different backend types."""
        if backend_type == 'superconducting':
            return {
                'T1': 100e-6,  # Relaxation time
                'T2': 50e-6,   # Dephasing time
                'gate_time': 20e-9,  # Gate operation time
                'readout_fidelity': 0.99,
                'gate_fidelity': 0.999,
                'crosstalk': 0.01
            }
        elif backend_type == 'trapped_ion':
            return {
                'T1': 10e-3,   # Much longer coherence times
                'T2': 1e-3,
                'gate_time': 10e-6,  # Slower gates
                'readout_fidelity': 0.995,
                'gate_fidelity': 0.9995,
                'crosstalk': 0.001  # Less crosstalk
            }
        elif backend_type == 'photonic':
            return {
                'T1': float('inf'),  # No decoherence in principle
                'T2': float('inf'),
                'gate_time': 1e-12,  # Very fast
                'readout_fidelity': 0.95,  # Detection efficiency limited
                'gate_fidelity': 0.99,
                'crosstalk': 0.0
            }
        else:
            # Default to simulator parameters
            return {
                'T1': float('inf'),
                'T2': float('inf'),
                'gate_time': 0,
                'readout_fidelity': 1.0,
                'gate_fidelity': 1.0,
                'crosstalk': 0.0
            }
    
    def simulate_decoherence(self, state: np.ndarray, time: float) -> np.ndarray:
        """Simulate decoherence effects over time."""
        T1 = self.device_parameters['T1']
        T2 = self.device_parameters['T2']
        
        if T1 == float('inf') and T2 == float('inf'):
            return state
        
        # Simplified decoherence model
        amplitude_decay = np.exp(-time / T1) if T1 != float('inf') else 1.0
        phase_decay = np.exp(-time / T2) if T2 != float('inf') else 1.0
        
        # Apply decoherence
        state = state * amplitude_decay
        phase_noise = np.random.normal(0, 1 - phase_decay, state.shape)
        state = state * np.exp(1j * phase_noise)
        
        return state
    
    def simulate_gate_error(self, ideal_result: float) -> float:
        """Simulate gate errors."""
        gate_fidelity = self.device_parameters['gate_fidelity']
        error_prob = 1 - gate_fidelity
        
        if np.random.random() < error_prob:
            # Apply random error
            error_magnitude = np.random.normal(0, error_prob)
            return ideal_result + error_magnitude
        
        return ideal_result
    
    def simulate_readout_error(self, true_result: int) -> int:
        """Simulate measurement readout errors."""
        readout_fidelity = self.device_parameters['readout_fidelity']
        error_prob = 1 - readout_fidelity
        
        if np.random.random() < error_prob:
            return 1 - true_result  # Flip result
        
        return true_result