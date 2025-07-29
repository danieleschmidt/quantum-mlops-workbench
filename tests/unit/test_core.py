"""Unit tests for quantum ML core components."""

import pytest
import numpy as np

from quantum_mlops.core import QuantumMLPipeline, QuantumDevice, QuantumModel


class TestQuantumMLPipeline:
    """Test cases for QuantumMLPipeline class."""
    
    def test_pipeline_initialization(self, simple_circuit):
        """Test pipeline initialization with different devices."""
        pipeline = QuantumMLPipeline(
            circuit=simple_circuit,
            n_qubits=4,
            device=QuantumDevice.SIMULATOR
        )
        
        assert pipeline.circuit == simple_circuit
        assert pipeline.n_qubits == 4
        assert pipeline.device == QuantumDevice.SIMULATOR
        
    def test_train_method(self, quantum_pipeline, sample_data):
        """Test training method returns valid model."""
        X_train, y_train = sample_data
        
        model = quantum_pipeline.train(
            X_train, y_train,
            epochs=10,
            learning_rate=0.01
        )
        
        assert isinstance(model, QuantumModel)
        assert model.n_qubits == quantum_pipeline.n_qubits
        
    def test_evaluate_method(self, quantum_pipeline, sample_data):
        """Test model evaluation with noise analysis."""
        X_train, y_train = sample_data
        model = quantum_pipeline.train(X_train, y_train, epochs=5)
        
        X_test, y_test = sample_data[:50], sample_data[50:]
        metrics = quantum_pipeline.evaluate(
            model, X_test, y_test,
            noise_models=['depolarizing']
        )
        
        assert hasattr(metrics, 'accuracy')
        assert hasattr(metrics, 'gradient_variance')
        assert 0 <= metrics.accuracy <= 1
        assert metrics.gradient_variance >= 0


class TestQuantumModel:
    """Test cases for QuantumModel class."""
    
    def test_model_properties(self, simple_circuit):
        """Test quantum model properties."""
        model = QuantumModel(simple_circuit, n_qubits=4)
        
        assert model.n_qubits == 4
        assert model.circuit == simple_circuit
        assert isinstance(model.circuit_depth, int)
        assert model.circuit_depth > 0
        
    def test_state_vector_shape(self, simple_circuit):
        """Test quantum state vector has correct dimensions."""
        n_qubits = 3
        model = QuantumModel(simple_circuit, n_qubits)
        
        state_vector = model.state_vector
        expected_size = 2 ** n_qubits
        
        assert len(state_vector) == expected_size
        assert state_vector.dtype == complex


class TestQuantumDevice:
    """Test cases for QuantumDevice enum."""
    
    def test_device_enum_values(self):
        """Test all quantum device enum values are accessible."""
        devices = [
            QuantumDevice.SIMULATOR,
            QuantumDevice.AWS_BRAKET,
            QuantumDevice.IBM_QUANTUM,
            QuantumDevice.IONQ
        ]
        
        for device in devices:
            assert isinstance(device.value, str)
            assert len(device.value) > 0