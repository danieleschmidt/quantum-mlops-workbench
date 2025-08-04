"""Pytest configuration and fixtures for quantum ML testing."""

import pytest
import numpy as np
import time
import tempfile
import os
from typing import Generator, Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from quantum_mlops.core import QuantumMLPipeline, QuantumDevice
from quantum_mlops.testing import QuantumTestCase


@pytest.fixture
def quantum_device() -> QuantumDevice:
    """Fixture providing default quantum device."""
    return QuantumDevice.SIMULATOR


@pytest.fixture
def n_qubits() -> int:
    """Fixture providing default number of qubits."""
    return 4


@pytest.fixture
def simple_circuit():
    """Fixture providing a simple test quantum circuit."""
    def circuit(params: np.ndarray, x: np.ndarray) -> float:
        # Simple parameterized quantum circuit for testing
        return float(np.sum(np.sin(params) * np.cos(x)))
    return circuit


@pytest.fixture
def quantum_pipeline(
    simple_circuit, 
    n_qubits: int, 
    quantum_device: QuantumDevice
) -> QuantumMLPipeline:
    """Fixture providing a quantum ML pipeline."""
    return QuantumMLPipeline(
        circuit=simple_circuit,
        n_qubits=n_qubits,
        device=quantum_device
    )


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray]:
    """Fixture providing sample training data."""
    np.random.seed(42)  # For reproducible tests
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture
def quantum_params(n_qubits: int) -> np.ndarray:
    """Fixture providing sample quantum circuit parameters."""
    np.random.seed(42)
    return np.random.uniform(0, 2*np.pi, n_qubits * 2)


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
    """Configure the test environment for quantum ML testing."""
    # Set up test environment variables
    import os
    os.environ['QUANTUM_TEST_MODE'] = 'true'
    os.environ['QUANTUM_BACKEND'] = 'simulator'
    
    yield
    
    # Cleanup after tests
    if 'QUANTUM_TEST_MODE' in os.environ:
        del os.environ['QUANTUM_TEST_MODE']
    if 'QUANTUM_BACKEND' in os.environ:
        del os.environ['QUANTUM_BACKEND']


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "quantum: marks tests that require quantum backends"
    )
    config.addinivalue_line(
        "markers", "hardware: marks tests that require real quantum hardware"
    )
    config.addinivalue_line(
        "markers", "simulation: marks tests that use quantum simulators only"
    )
    config.addinivalue_line(
        "markers", "load: marks load testing tests"
    )
    config.addinivalue_line(
        "markers", "chaos: marks chaos engineering tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks performance benchmarking tests"
    )
    config.addinivalue_line(
        "markers", "noise_resilience: marks noise resilience testing"
    )
    config.addinivalue_line(
        "markers", "gradient_stability: marks gradient stability tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks unit tests"
    )
    config.addinivalue_line(
        "markers", "backend_compatibility: marks backend compatibility tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle quantum-specific markers."""
    if config.getoption("--quantum-backend") == "simulator":
        # Skip hardware tests when using simulator
        skip_hardware = pytest.mark.skip(reason="hardware backend not selected")
        for item in items:
            if "hardware" in item.keywords:
                item.add_marker(skip_hardware)
    
    if not config.getoption("--run-slow"):
        # Skip slow tests unless explicitly requested
        skip_slow = pytest.mark.skip(reason="slow tests not enabled (use --run-slow)")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    if not config.getoption("--quantum-hardware"):
        # Skip hardware tests unless quantum hardware is enabled
        skip_hardware = pytest.mark.skip(reason="quantum hardware tests not enabled (use --quantum-hardware)")
        for item in items:
            if "hardware" in item.keywords:
                item.add_marker(skip_hardware)
    
    if not config.getoption("--performance-mode"):
        # Skip performance tests unless performance mode is enabled
        skip_performance = pytest.mark.skip(reason="performance tests not enabled (use --performance-mode)")
        for item in items:
            if "performance" in item.keywords:
                item.add_marker(skip_performance)


# Advanced Quantum Testing Fixtures

@pytest.fixture
def quantum_test_case() -> QuantumTestCase:
    """Fixture providing an enhanced QuantumTestCase instance."""
    return QuantumTestCase()


@pytest.fixture(params=[
    QuantumDevice.SIMULATOR,
    QuantumDevice.IBM_QUANTUM,
    QuantumDevice.AWS_BRAKET
])
def quantum_backend_device(request) -> QuantumDevice:
    """Parametrized fixture for testing across multiple quantum backends."""
    return request.param


@pytest.fixture
def mock_quantum_backend():
    """Mock quantum backend for testing without real hardware."""
    backend = MagicMock()
    backend.name = "mock_backend"
    backend.is_available.return_value = True
    backend.get_device_properties.return_value = {
        "name": "mock_device",
        "provider": "Mock",
        "qubit_count": 30,
        "connectivity": "all_to_all",
        "native_gate_set": ["rx", "ry", "rz", "cnot"]
    }
    
    # Mock circuit execution
    def mock_execute(circuits, shots=1024):
        results = []
        for i, circuit in enumerate(circuits):
            # Generate realistic mock results
            counts = {f"{i:0{circuit.get('n_qubits', 4)}b}": shots // (2**i + 1) for i in range(4)}
            results.append({
                "circuit_id": f"mock_circuit_{i}",
                "counts": counts,
                "expectation_value": np.random.uniform(-1, 1),
                "execution_time": np.random.uniform(0.1, 2.0),
                "shots": shots
            })
        return results
    
    backend.execute.side_effect = mock_execute
    return backend


@pytest.fixture
def noise_models() -> Dict[str, Dict]:
    """Fixture providing comprehensive noise model configurations."""
    return {
        "depolarizing": {
            "type": "depolarizing",
            "probability": 0.01,
            "qubits": "all"
        },
        "amplitude_damping": {
            "type": "amplitude_damping", 
            "gamma": 0.05,
            "qubits": "all"
        },
        "phase_damping": {
            "type": "phase_damping",
            "gamma": 0.02,
            "qubits": "all"
        },
        "bit_flip": {
            "type": "bit_flip",
            "probability": 0.001,
            "qubits": "all"
        },
        "phase_flip": {
            "type": "phase_flip",
            "probability": 0.001,
            "qubits": "all"
        },
        "thermal": {
            "type": "thermal",
            "temperature": 0.01,
            "t1": 50e-6,  # T1 time in seconds
            "t2": 70e-6,  # T2 time in seconds
            "qubits": "all"
        }
    }


@pytest.fixture
def hardware_configs() -> Dict[str, Dict]:
    """Fixture providing hardware configuration templates."""
    return {
        "ibm_quantum": {
            "backend_name": "ibmq_qasm_simulator",
            "coupling_map": [[0, 1], [1, 2], [2, 3], [0, 3]],
            "basis_gates": ["id", "rz", "sx", "x", "cx"],
            "max_shots": 8192,
            "max_experiments": 300
        },
        "aws_braket": {
            "device_arn": "local:braket/braket.local.qubit",
            "max_shots": 100000,
            "supported_gates": ["rx", "ry", "rz", "h", "cnot", "swap"],
            "qubit_count": 30
        },
        "google_quantum": {
            "processor_id": "rainbow",
            "gate_set": "sqrt_iswap",
            "max_qubits": 23,
            "topology": "grid"
        },
        "rigetti": {
            "qpu": "Aspen-M-3",
            "compiler": "quilc",
            "max_qubits": 80,
            "topology": "octagonal"
        }
    }


@pytest.fixture
def circuit_templates() -> Dict[str, Dict]:
    """Fixture providing various quantum circuit templates for testing."""
    return {
        "variational": {
            "gates": [
                {"type": "ry", "qubit": 0, "angle": "param_0"},
                {"type": "ry", "qubit": 1, "angle": "param_1"},
                {"type": "cnot", "control": 0, "target": 1},
                {"type": "ry", "qubit": 0, "angle": "param_2"},
                {"type": "ry", "qubit": 1, "angle": "param_3"}
            ],
            "n_qubits": 2,
            "n_parameters": 4,
            "depth": 3
        },
        "entangled": {
            "gates": [
                {"type": "h", "qubit": 0},
                {"type": "cnot", "control": 0, "target": 1},
                {"type": "cnot", "control": 1, "target": 2},
                {"type": "cnot", "control": 2, "target": 3}
            ],
            "n_qubits": 4,
            "n_parameters": 0,
            "depth": 2
        },
        "deep_circuit": {
            "gates": [
                {"type": "ry", "qubit": i % 4, "angle": f"param_{i}"}
                for i in range(20)
            ] + [
                {"type": "cnot", "control": i % 4, "target": (i + 1) % 4}
                for i in range(10)
            ],
            "n_qubits": 4,
            "n_parameters": 20,
            "depth": 10
        },
        "qft": {
            "gates": [
                {"type": "h", "qubit": 0},
                {"type": "cphase", "control": 1, "target": 0, "angle": np.pi/2},
                {"type": "h", "qubit": 1},
                {"type": "cphase", "control": 2, "target": 0, "angle": np.pi/4},
                {"type": "cphase", "control": 2, "target": 1, "angle": np.pi/2},
                {"type": "h", "qubit": 2}
            ],
            "n_qubits": 3,
            "n_parameters": 0,
            "depth": 3
        }
    }


@pytest.fixture
def performance_thresholds() -> Dict[str, float]:
    """Fixture providing performance threshold configurations."""
    return {
        "max_execution_time": 10.0,  # seconds
        "min_fidelity": 0.85,
        "max_gradient_variance": 1.0,
        "min_convergence_rate": 0.01,
        "max_memory_usage_mb": 500,
        "min_throughput_ops_per_sec": 1.0,
        "max_error_rate": 0.05
    }


@pytest.fixture
def load_test_config() -> Dict[str, Any]:
    """Fixture providing load testing configuration."""
    return {
        "max_concurrent_circuits": 50,
        "test_duration_seconds": 30,
        "ramp_up_time_seconds": 5,
        "target_throughput": 10,  # operations per second
        "stress_multiplier": 2.0,
        "memory_limit_mb": 1000
    }


@pytest.fixture
def chaos_test_scenarios() -> List[Dict[str, Any]]:
    """Fixture providing chaos engineering test scenarios."""
    return [
        {
            "name": "decoherence_spike",
            "type": "noise_injection",
            "parameters": {"t1_reduction": 0.5, "t2_reduction": 0.3},
            "duration": 10,
            "severity": "medium"
        },
        {
            "name": "backend_failure",
            "type": "backend_unavailable",
            "parameters": {"failure_probability": 0.2},
            "duration": 15,
            "severity": "high"
        },
        {
            "name": "measurement_errors",
            "type": "readout_noise",
            "parameters": {"error_rate": 0.1},
            "duration": 20,
            "severity": "low"
        },
        {
            "name": "crosstalk_interference",
            "type": "crosstalk",
            "parameters": {"coupling_strength": 0.05},
            "duration": 25,
            "severity": "medium"
        }
    ]


@pytest.fixture
def temp_storage_dir():
    """Fixture providing temporary storage directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def monitoring_config(temp_storage_dir) -> Dict[str, Any]:
    """Fixture providing monitoring configuration for tests."""
    return {
        "enabled": True,
        "storage_path": temp_storage_dir,
        "alert_thresholds": {
            "fidelity_drop": 0.05,
            "gradient_explosion": 5.0,
            "execution_timeout": 30.0,
            "memory_usage_mb": 500
        },
        "metrics_collection_interval": 1.0,
        "enable_real_time_monitoring": False
    }


@pytest.fixture
def quantum_states() -> Dict[str, np.ndarray]:
    """Fixture providing various quantum states for testing."""
    states = {}
    
    # |0⟩ state
    states["zero"] = np.array([1.0, 0.0], dtype=complex)
    
    # |1⟩ state  
    states["one"] = np.array([0.0, 1.0], dtype=complex)
    
    # |+⟩ state (superposition)
    states["plus"] = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    
    # |-⟩ state
    states["minus"] = np.array([1.0, -1.0], dtype=complex) / np.sqrt(2)
    
    # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    states["bell_phi_plus"] = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2)
    
    # Random mixed state
    np.random.seed(42)
    random_state = np.random.complex128(4)
    states["random"] = random_state / np.linalg.norm(random_state)
    
    return states


@pytest.fixture
def gradient_test_functions():
    """Fixture providing functions for gradient testing."""
    functions = {}
    
    # Quadratic function
    functions["quadratic"] = lambda x: np.sum(x**2)
    
    # Rosenbrock function (has narrow valley)
    functions["rosenbrock"] = lambda x: np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    # Rastrigin function (many local minima)
    functions["rastrigin"] = lambda x: 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    
    # Simple sinusoidal (for parameter shift testing)
    functions["sinusoidal"] = lambda x: np.sum(np.sin(x))
    
    return functions


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Set up logging for tests."""
    import logging
    
    # Configure logging for tests
    logging.basicConfig(
        level=logging.WARNING,  # Reduce log noise during tests
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('requests').setLevel(logging.ERROR)
    
    yield
    
    # Clean up logging configuration
    logging.getLogger().handlers.clear()


@pytest.fixture
def performance_baseline() -> Dict[str, float]:
    """Fixture providing performance baseline measurements."""
    return {
        "circuit_compilation_time": 0.001,  # seconds
        "single_shot_execution_time": 0.1,   # seconds
        "gradient_computation_time": 0.5,    # seconds
        "parameter_update_time": 0.001,      # seconds
        "memory_usage_per_qubit_mb": 2.0,    # MB
        "throughput_circuits_per_second": 10.0
    }


def pytest_addoption(parser):
    """Add quantum-specific command line options."""
    parser.addoption(
        "--quantum-backend",
        action="store",
        default="simulator",
        help="Quantum backend to use for tests"
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--quantum-hardware",
        action="store_true",
        default=False,
        help="Enable tests on real quantum hardware"
    )
    parser.addoption(
        "--load-test-duration",
        action="store",
        type=int,
        default=30,
        help="Duration for load tests in seconds"
    )
    parser.addoption(
        "--performance-mode",
        action="store_true",
        default=False,
        help="Run performance benchmarking tests"
    )