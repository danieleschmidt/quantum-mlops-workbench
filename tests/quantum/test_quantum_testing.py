"""Tests for quantum ML testing framework."""

import pytest
import numpy as np

from quantum_mlops.testing import QuantumTestCase


class TestQuantumTestCase(QuantumTestCase):
    """Test the quantum testing framework itself."""
    
    @pytest.mark.simulation
    def test_gradient_stability(self):
        """Test that gradients remain stable under noise."""
        model = self.create_model(n_qubits=4)
        
        # Test gradient variance
        variance = self.measure_gradient_variance(
            model,
            n_samples=50,  # Reduced for testing
            noise_level=0.01
        )
        
        self.assertLess(variance, 0.1, 
                       "Gradient variance too high for reliable training")
        
    @pytest.mark.simulation
    def test_hardware_compatibility(self):
        """Ensure circuit runs on target hardware."""
        circuit = self.build_circuit()
        
        # Check gate decomposition for different backends
        self.assert_native_gates(circuit, backend='ionq')
        self.assert_native_gates(circuit, backend='ibm')
        
        # Verify connectivity constraints
        self.assert_topology_compatible(circuit, device='ibmq_toronto')
        
    @pytest.mark.simulation
    def test_noise_resilience(self):
        """Test model performance under realistic noise."""
        model = self.create_model()
        
        # Test with increasing noise levels
        for noise_prob in [0.001, 0.01, 0.05]:
            accuracy = self.evaluate_with_noise(
                model,
                noise_model='depolarizing',
                noise_prob=noise_prob
            )
            
            # Ensure graceful degradation
            self.assertGreater(accuracy, 0.6,
                             f"Model fails at {noise_prob} noise level")
            
    def test_create_model_default_qubits(self):
        """Test model creation with default qubit count."""
        model = self.create_model()
        
        self.assertEqual(model.n_qubits, self.n_qubits)
        self.assertIsNotNone(model.circuit)
        
    def test_create_model_custom_qubits(self):
        """Test model creation with custom qubit count."""
        custom_qubits = 6
        model = self.create_model(n_qubits=custom_qubits)
        
        self.assertEqual(model.n_qubits, custom_qubits)
        
    def test_build_circuit_returns_callable(self):
        """Test that build_circuit returns a callable function."""
        circuit = self.build_circuit()
        
        self.assertTrue(callable(circuit))
        
        # Test that circuit can be called with parameters
        params = np.array([0.1, 0.2, 0.3])
        result = circuit(params)
        
        self.assertIsInstance(result, float)


class TestQuantumTestFramework:
    """Additional tests for quantum testing utilities."""
    
    @pytest.mark.simulation
    def test_quantum_test_case_inheritance(self):
        """Test that QuantumTestCase can be properly inherited."""
        
        class MyQuantumTest(QuantumTestCase):
            def test_custom_quantum_behavior(self):
                model = self.create_model(n_qubits=2)
                self.assertEqual(model.n_qubits, 2)
                
        # Instantiate and run a simple test
        test_instance = MyQuantumTest()
        test_instance.setUp()
        test_instance.test_custom_quantum_behavior()
        
    @pytest.mark.simulation 
    def test_gradient_variance_scaling(self):
        """Test that gradient variance scales with noise level."""
        test_case = QuantumTestCase()
        test_case.setUp()
        
        model = test_case.create_model()
        
        # Test different noise levels
        low_noise = test_case.measure_gradient_variance(
            model, n_samples=20, noise_level=0.001
        )
        high_noise = test_case.measure_gradient_variance(
            model, n_samples=20, noise_level=0.1
        )
        
        # Higher noise should generally lead to higher variance
        assert high_noise >= low_noise * 0.8  # Allow some variance in random sampling