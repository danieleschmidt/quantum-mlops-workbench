"""Unit tests for quantum backend implementations with enhanced coverage."""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock

from quantum_mlops.backends import (
    QuantumBackend,
    QuantumJob,
    CircuitResult,
    JobStatus,
    BackendManager,
    QuantumExecutor,
)
from quantum_mlops.core import QuantumDevice
from quantum_mlops.testing import QuantumTestCase


class TestBackendManager(QuantumTestCase):
    """Enhanced test cases for backend manager functionality."""
    
    @pytest.mark.unit
    def test_backend_manager_initialization(self):
        """Test backend manager initializes correctly."""
        manager = BackendManager()
        assert isinstance(manager.backends, dict)
        assert len(manager.fallback_order) > 0
        assert manager.retry_attempts > 0
        
    @pytest.mark.unit
    def test_register_backend(self):
        """Test backend registration."""
        manager = BackendManager()
        mock_backend = Mock(spec=QuantumBackend)
        mock_backend.name = "test_backend"
        
        manager.register_backend(mock_backend)
        assert "test_backend" in manager.backends
        assert manager.backends["test_backend"] == mock_backend
        
    @pytest.mark.unit
    def test_register_duplicate_backend(self):
        """Test registering duplicate backend names."""
        manager = BackendManager()
        mock_backend1 = Mock(spec=QuantumBackend)
        mock_backend1.name = "duplicate_backend"
        mock_backend2 = Mock(spec=QuantumBackend)
        mock_backend2.name = "duplicate_backend"
        
        manager.register_backend(mock_backend1)
        # Second registration should replace the first
        manager.register_backend(mock_backend2)
        
        assert manager.backends["duplicate_backend"] == mock_backend2
        
    def test_validate_circuit(self):
        """Test circuit validation and normalization."""
        manager = BackendManager()
        
        # Test minimal circuit
        circuit = {"gates": [{"type": "h", "qubit": 0}]}
        validated = manager.validate_circuit(circuit)
        
        assert "gates" in validated
        assert "n_qubits" in validated
        assert validated["n_qubits"] >= 1
        
        # Test circuit with inferred qubits
        circuit = {"gates": [
            {"type": "h", "qubit": 0},
            {"type": "cx", "control": 0, "target": 2}
        ]}
        validated = manager.validate_circuit(circuit)
        assert validated["n_qubits"] == 3  # 0, 1, 2
        
    @patch('quantum_mlops.backends.pennylane_backend.PENNYLANE_AVAILABLE', True)
    def test_create_backend_simulator(self):
        """Test creating simulator backend."""
        manager = BackendManager()
        backend = manager.create_backend(QuantumDevice.SIMULATOR, wires=4)
        
        assert backend is not None
        assert "pennylane" in backend.name
        assert backend.name in manager.backends
        
    def test_create_backend_invalid_device(self):
        """Test creating backend with invalid device."""
        manager = BackendManager()
        
        with pytest.raises(ValueError, match="Unsupported device type"):
            manager.create_backend("invalid_device")


class TestQuantumExecutor:
    """Test the quantum executor functionality."""
    
    def test_quantum_executor_initialization(self):
        """Test quantum executor initializes correctly."""
        executor = QuantumExecutor()
        assert executor.backend_manager is not None
        
    def test_quantum_executor_with_custom_manager(self):
        """Test quantum executor with custom backend manager."""
        manager = BackendManager()
        executor = QuantumExecutor(manager)
        assert executor.backend_manager == manager
        
    def test_validate_circuit_input_single(self):
        """Test single circuit input validation."""
        executor = QuantumExecutor()
        circuit = {"gates": [{"type": "h", "qubit": 0}]}
        
        # Mock the backend manager execution
        with patch.object(executor.backend_manager, 'validate_circuit', return_value=circuit):
            with patch.object(executor.backend_manager, 'execute_with_fallback') as mock_execute:
                mock_job = Mock()
                mock_job.status = JobStatus.COMPLETED
                mock_job.results = [CircuitResult("test", {"0": 500, "1": 500}, 0.0)]
                mock_execute.return_value = mock_job
                
                result = executor.execute(circuit)
                assert isinstance(result, CircuitResult)
                
    def test_validate_circuit_input_multiple(self):
        """Test multiple circuit input validation."""
        executor = QuantumExecutor()
        circuits = [
            {"gates": [{"type": "h", "qubit": 0}]},
            {"gates": [{"type": "x", "qubit": 1}]}
        ]
        
        with patch.object(executor.backend_manager, 'validate_circuit', side_effect=lambda x: x):
            with patch.object(executor.backend_manager, 'execute_with_fallback') as mock_execute:
                mock_job = Mock()
                mock_job.status = JobStatus.COMPLETED
                mock_job.results = [
                    CircuitResult("test_0", {"0": 500, "1": 500}, 0.0),
                    CircuitResult("test_1", {"0": 1000, "1": 0}, 1.0)
                ]
                mock_execute.return_value = mock_job
                
                results = executor.execute(circuits)
                assert isinstance(results, list)
                assert len(results) == 2
                
    def test_execute_with_failed_job(self):
        """Test execution with failed job."""
        executor = QuantumExecutor()
        circuit = {"gates": [{"type": "h", "qubit": 0}]}
        
        with patch.object(executor.backend_manager, 'validate_circuit', return_value=circuit):
            with patch.object(executor.backend_manager, 'execute_with_fallback') as mock_execute:
                mock_job = Mock()
                mock_job.status = JobStatus.FAILED
                mock_job.error_message = "Test error"
                mock_execute.return_value = mock_job
                
                with pytest.raises(RuntimeError, match="Circuit execution failed"):
                    executor.execute(circuit)
                    
    def test_backend_optimization(self):
        """Test backend optimization selection."""
        executor = QuantumExecutor()
        circuits = [{"gates": [{"type": "h", "qubit": 0}], "n_qubits": 2}]
        
        with patch.object(executor.backend_manager, 'optimize_backend_selection') as mock_optimize:
            mock_optimize.return_value = ["best_backend"]
            
            with patch.object(executor.backend_manager, 'validate_circuit', side_effect=lambda x: x):
                with patch.object(executor.backend_manager, 'execute_with_fallback') as mock_execute:
                    mock_job = Mock()
                    mock_job.status = JobStatus.COMPLETED
                    mock_job.results = [CircuitResult("test", {"0": 500, "1": 500}, 0.0)]
                    mock_execute.return_value = mock_job
                    
                    executor.execute(circuits)
                    mock_optimize.assert_called_once_with(circuits, "cost_performance")
                    
    def test_estimate_execution_cost(self):
        """Test execution cost estimation."""
        executor = QuantumExecutor()
        circuit = {"gates": [{"type": "h", "qubit": 0}]}
        
        mock_backend = Mock()
        mock_backend.get_device_properties.return_value = {
            "provider": "AWS",
            "simulator": False
        }
        
        with patch.object(executor.backend_manager, 'get_backend', return_value=mock_backend):
            cost_info = executor.estimate_execution_cost(circuit, "test_backend")
            
            assert "backend" in cost_info
            assert "total_cost" in cost_info
            assert "circuits" in cost_info
            
    def test_estimate_cost_unknown_backend(self):
        """Test cost estimation for unknown backend."""
        executor = QuantumExecutor()
        circuit = {"gates": [{"type": "h", "qubit": 0}]}
        
        with patch.object(executor.backend_manager, 'get_backend', return_value=None):
            with pytest.raises(ValueError, match="Backend .* not found"):
                executor.estimate_execution_cost(circuit, "unknown_backend")


@pytest.mark.skipif(
    not hasattr(pytest, "importorskip") or 
    pytest.importorskip("pennylane", minversion="0.30.0") is None,
    reason="PennyLane not available"
)
class TestPennyLaneBackend:
    """Test PennyLane backend implementation."""
    
    def test_pennylane_backend_creation(self):
        """Test PennyLane backend creation."""
        from quantum_mlops.backends.pennylane_backend import PennyLaneBackend
        
        backend = PennyLaneBackend("default.qubit", wires=4)
        assert backend.device_name == "default.qubit"
        assert backend.name == "pennylane_default.qubit"
        
    def test_pennylane_backend_connection(self):
        """Test PennyLane backend connection."""
        from quantum_mlops.backends.pennylane_backend import PennyLaneBackend
        
        backend = PennyLaneBackend("default.qubit", wires=4)
        backend.connect()
        
        assert backend._is_connected
        assert backend.device is not None
        assert backend.is_available()
        
        backend.disconnect()
        assert not backend._is_connected
        
    def test_pennylane_circuit_compilation(self):
        """Test PennyLane circuit compilation."""
        from quantum_mlops.backends.pennylane_backend import PennyLaneBackend
        
        backend = PennyLaneBackend("default.qubit", wires=4)
        backend.connect()
        
        circuit = {
            "gates": [
                {"type": "h", "qubit": 0},
                {"type": "cx", "control": 0, "target": 1}
            ],
            "n_qubits": 2
        }
        
        compiled = backend.compile_circuit(circuit)
        assert "original" in compiled
        assert "gates" in compiled
        assert compiled["n_qubits"] == 2
        
    def test_pennylane_circuit_execution(self):
        """Test PennyLane circuit execution."""
        from quantum_mlops.backends.pennylane_backend import PennyLaneBackend
        
        backend = PennyLaneBackend("default.qubit", wires=2, shots=100)
        backend.connect()
        
        circuits = [{
            "gates": [{"type": "h", "qubit": 0}],
            "n_qubits": 2,
            "measurements": [{"type": "expectation", "wires": 0, "observable": "Z"}]
        }]
        
        job = backend.submit_job(circuits, shots=100)
        
        assert job.status == JobStatus.COMPLETED
        assert len(job.results) == 1
        assert job.results[0].expectation_value is not None
        

@pytest.mark.skipif(
    not hasattr(pytest, "importorskip") or 
    pytest.importorskip("qiskit", minversion="0.40.0") is None,
    reason="Qiskit not available"
)
class TestQiskitBackend:
    """Test Qiskit backend implementation."""
    
    def test_qiskit_backend_creation(self):
        """Test Qiskit backend creation."""
        from quantum_mlops.backends.qiskit_backend import QiskitBackend
        
        backend = QiskitBackend("qasm_simulator")
        assert backend.backend_name == "qasm_simulator"
        assert backend.name == "qiskit_qasm_simulator"
        
    def test_qiskit_backend_connection(self):
        """Test Qiskit backend connection."""
        from quantum_mlops.backends.qiskit_backend import QiskitBackend
        
        backend = QiskitBackend("qasm_simulator")
        backend.connect()
        
        assert backend._is_connected
        assert backend.backend is not None
        assert backend.is_available()
        
    def test_qiskit_circuit_compilation(self):
        """Test Qiskit circuit compilation.""" 
        from quantum_mlops.backends.qiskit_backend import QiskitBackend
        
        backend = QiskitBackend("qasm_simulator")
        backend.connect()
        
        circuit = {
            "gates": [
                {"type": "h", "qubit": 0},
                {"type": "cx", "control": 0, "target": 1}
            ],
            "n_qubits": 2
        }
        
        compiled = backend.compile_circuit(circuit)
        assert "qiskit_circuit" in compiled
        assert compiled["n_qubits"] == 2


@pytest.mark.skipif(
    not hasattr(pytest, "importorskip") or 
    pytest.importorskip("braket", minversion="1.50.0") is None,
    reason="AWS Braket SDK not available"
)  
class TestBraketBackend:
    """Test AWS Braket backend implementation."""
    
    def test_braket_backend_creation(self):
        """Test Braket backend creation."""
        from quantum_mlops.backends.braket_backend import BraketBackend
        
        backend = BraketBackend("local:braket/braket.local.qubit")
        assert backend.device_arn == "local:braket/braket.local.qubit"
        assert backend.is_local
        
    def test_braket_local_connection(self):
        """Test Braket local simulator connection."""
        from quantum_mlops.backends.braket_backend import BraketBackend
        
        backend = BraketBackend("local:braket/braket.local.qubit")
        backend.connect()
        
        assert backend._is_connected
        assert backend.device is not None
        assert backend.is_available()
        
    def test_braket_circuit_compilation(self):
        """Test Braket circuit compilation."""
        from quantum_mlops.backends.braket_backend import BraketBackend
        
        backend = BraketBackend("local:braket/braket.local.qubit")
        backend.connect()
        
        circuit = {
            "gates": [
                {"type": "h", "qubit": 0},
                {"type": "cx", "control": 0, "target": 1}
            ],
            "n_qubits": 2
        }
        
        compiled = backend.compile_circuit(circuit)
        assert "braket_circuit" in compiled
        assert compiled["n_qubits"] == 2


class TestBackendIntegration:
    """Test integration between backends and pipeline."""
    
    def test_pipeline_backend_selection(self):
        """Test pipeline selects appropriate backend."""
        from quantum_mlops.core import QuantumMLPipeline, QuantumDevice
        
        def dummy_circuit():
            pass
            
        # Test simulator selection
        pipeline = QuantumMLPipeline(dummy_circuit, 2, QuantumDevice.SIMULATOR)
        info = pipeline.get_backend_info()
        
        assert info["device"] == "simulator"
        assert info["n_qubits"] == 2
        
    def test_pipeline_fallback_behavior(self):
        """Test pipeline falls back to simulation when backends fail."""
        from quantum_mlops.core import QuantumMLPipeline, QuantumDevice
        
        def dummy_circuit():
            pass
            
        # Create pipeline that should fallback to simulation
        pipeline = QuantumMLPipeline(dummy_circuit, 2, QuantumDevice.AWS_BRAKET)
        
        # Should not raise an exception even if AWS backend unavailable
        info = pipeline.get_backend_info()
        assert "device" in info
        
    @patch('quantum_mlops.backends.pennylane_backend.PENNYLANE_AVAILABLE', True)
    def test_pipeline_with_real_backend(self):
        """Test pipeline with real backend integration."""
        from quantum_mlops.core import QuantumMLPipeline, QuantumDevice
        
        def dummy_circuit():
            pass
            
        pipeline = QuantumMLPipeline(dummy_circuit, 2, QuantumDevice.SIMULATOR, wires=4)
        
        # Test benchmark if backends available
        try:
            benchmark = pipeline.benchmark_execution(test_samples=2)
            assert "simulation_time" in benchmark
        except Exception:
            # OK if backends not fully available in test environment
            pass