"""Pytest configuration and fixtures for quantum ML testing."""

import pytest
import numpy as np
from typing import Generator, Any

from quantum_mlops.core import QuantumMLPipeline, QuantumDevice


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


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle quantum-specific markers."""
    if config.getoption("--quantum-backend") == "simulator":
        # Skip hardware tests when using simulator
        skip_hardware = pytest.mark.skip(reason="hardware backend not selected")
        for item in items:
            if "hardware" in item.keywords:
                item.add_marker(skip_hardware)


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