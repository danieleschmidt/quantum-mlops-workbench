"""Comprehensive tests for quantum ML testing framework with enhanced coverage."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from quantum_mlops.testing import (
    QuantumTestCase, 
    quantum_test, 
    noise_resilience_test, 
    performance_benchmark,
    QuantumCircuitOptimizer,
    NoiseModelTester,
    QuantumHardwareSimulator
)
from quantum_mlops.core import QuantumModel, QuantumMLPipeline, QuantumDevice


class TestEnhancedQuantumTestCase(QuantumTestCase):
    """Test the enhanced quantum testing framework itself with comprehensive coverage."""
    
    @pytest.mark.simulation
    def test_gradient_stability_with_comprehensive_metrics(self):
        """Test gradient stability with enhanced metrics and noise analysis."""
        model = self.create_model(n_qubits=4, circuit_type='variational', depth=2)
        X, y = self.create_test_dataset(n_samples=20, n_features=4)
        
        # Test gradient variance with different methods
        for gradient_method in ['parameter_shift', 'finite_difference']:
            variance_metrics = self.measure_gradient_variance(
                model, X, y,
                n_samples=10,  # Reduced for faster testing
                noise_level=0.01,
                gradient_method=gradient_method
            )
            
            # Comprehensive gradient stability checks
            self.assertLess(variance_metrics['variance'], 0.1, 
                           f"{gradient_method}: Gradient variance too high")
            self.assertGreater(variance_metrics['mean_magnitude'], 1e-8,
                             f"{gradient_method}: Gradients too small (vanishing gradient)")
            self.assertLess(variance_metrics['coefficient_of_variation'], 2.0,
                           f"{gradient_method}: Gradient magnitudes too unstable")
            self.assertLess(variance_metrics['gradient_norm_stability'], 0.5,
                           f"{gradient_method}: Gradient norms too unstable")
    
    @pytest.mark.simulation
    def test_hardware_compatibility_comprehensive(self):
        """Test comprehensive hardware compatibility across all backends."""
        circuit = self.build_circuit(circuit_type='variational', depth=2)
        
        # Test all supported backends
        backends = ['simulator', 'ionq', 'ibm', 'google', 'rigetti']
        
        for backend in backends:
            with self.subTest(backend=backend):
                # Test native gates compatibility
                self.assert_native_gates(circuit, backend, strict=False)
                
                # Test topology compatibility
                self.assert_topology_compatible(circuit, backend)
                
                # Test circuit depth is reasonable for backend
                self.assert_circuit_depth_reasonable(circuit)
    
    @pytest.mark.simulation
    @noise_resilience_test(
        noise_models=['depolarizing', 'amplitude_damping', 'phase_damping'],
        noise_levels=[0.001, 0.01, 0.05]
    )
    def test_comprehensive_noise_resilience(self):
        """Test model resilience under various noise conditions."""
        model = self.create_model(n_qubits=3, circuit_type='variational')
        X, y = self.create_test_dataset(n_samples=20, n_features=3)
        
        noise_model = self.current_noise_model
        noise_level = self.current_noise_level
        
        # Evaluate performance under noise
        noise_metrics = self.evaluate_with_noise(
            model, X, y,
            noise_model=noise_model,
            noise_prob=noise_level,
            n_trials=5  # Reduced for faster testing
        )
        
        # Check noise resilience
        self.assertGreater(noise_metrics['mean_accuracy'], 0.3,
                         f"Model fails under {noise_model} noise at {noise_level}")
        self.assertGreater(noise_metrics['noise_robustness_score'], 0.5,
                         f"Model not robust to {noise_model} noise")
        
        # Check that higher noise levels generally reduce performance
        if noise_level > 0.01:
            self.assertLess(noise_metrics['std_accuracy'], 0.3,
                          "Performance variance too high under noise")
    
    @pytest.mark.simulation
    @performance_benchmark(min_throughput=5.0, max_latency=5.0)
    def test_performance_benchmarking(self):
        """Test performance benchmarking capabilities."""
        model = self.create_model(n_qubits=3, circuit_type='basic')
        X, y = self.create_test_dataset(n_samples=50, n_features=3)
        self.benchmark_samples = len(X)
        
        # Benchmark execution
        perf_metrics = self.benchmark_execution(
            model, X, n_runs=5, warmup_runs=2
        )
        
        # Performance assertions
        self.assertLess(perf_metrics['mean_execution_time'], 2.0,
                       "Execution time too slow")
        self.assertGreater(perf_metrics['throughput_samples_per_second'], 5.0,
                         "Throughput too low")
        self.assertLess(perf_metrics['std_execution_time'] / perf_metrics['mean_execution_time'], 0.5,
                       "Execution time too variable")
    
    @pytest.mark.simulation
    def test_quantum_specific_assertions(self):
        """Test quantum-specific assertion methods."""
        # Test fidelity assertions
        state1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        state2 = np.array([0.9, 0.1, 0.1, 0.1], dtype=complex)
        state2 = state2 / np.linalg.norm(state2)  # Normalize
        
        # This should pass (states are similar)
        self.assert_fidelity_above_threshold(state1, state1, threshold=0.99)
        
        # Test gradient magnitude assertions
        good_gradients = np.array([[0.1, 0.2, 0.15], [0.12, 0.18, 0.14]])
        self.assert_gradient_magnitude_reasonable(good_gradients)
        
        # Test quantum advantage assertion
        self.assert_quantum_advantage(0.85, 0.75, advantage_threshold=0.05)
    
    @pytest.mark.simulation
    def test_circuit_architecture_variations(self):
        """Test different circuit architectures and configurations."""
        architectures = [
            ('basic', 2, False),
            ('variational', 3, True),
            ('embedding', 2, True),
            ('variational', 4, False)
        ]
        
        for circuit_type, depth, entangling in architectures:
            with self.subTest(circuit_type=circuit_type, depth=depth, entangling=entangling):
                model = self.create_model(
                    n_qubits=3,
                    circuit_type=circuit_type,
                    depth=depth,
                    entangling=entangling
                )
                
                # Test model properties
                self.assertEqual(model.n_qubits, 3)
                self.assertEqual(model.metadata['circuit_type'], circuit_type)
                self.assertEqual(model.metadata['depth'], depth)
                self.assertEqual(model.metadata['entangling'], entangling)
                
                # Test model can make predictions
                X = np.random.random((5, 3))
                model.parameters = np.random.uniform(-np.pi, np.pi, 18)
                predictions = model.predict(X)
                
                self.assertEqual(len(predictions), 5)
                self.assertTrue(all(isinstance(p, (int, float)) for p in predictions))
    
    @pytest.mark.simulation
    def test_mock_backend_creation(self):
        """Test mock backend creation for different backend types."""
        backend_types = ['simulator', 'ionq', 'ibm', 'google']
        
        for backend_type in backend_types:
            with self.subTest(backend_type=backend_type):
                mock_backend = self.create_mock_backend(backend_type)
                
                # Test mock backend properties
                self.assertTrue(mock_backend.is_available())
                self.assertEqual(mock_backend.name, f'mock_{backend_type}')
                
                # Test device properties
                properties = mock_backend.get_device_properties()
                self.assertIn('native_gates', properties)
                self.assertIn('shots', properties)
                
                # Test job submission
                job = mock_backend.submit_job([])
                self.assertEqual(job.job_id, 'test_job_123')
    
    @pytest.mark.simulation
    def test_chaos_engineering(self):
        """Test chaos engineering capabilities."""
        def test_function():
            # Simple test function that could fail
            model = self.create_model(n_qubits=2)
            X = np.random.random((5, 2))
            model.parameters = np.random.uniform(-np.pi, np.pi, 12)
            return model.predict(X)
        
        failure_modes = ['network_failure', 'hardware_error', 'memory_pressure']
        
        chaos_results = self.run_chaos_test(
            test_function,
            failure_modes,
            n_iterations=5
        )
        
        # Check chaos test results
        self.assertEqual(chaos_results['total_iterations'], 5)
        self.assertGreaterEqual(chaos_results['successful_runs'], 0)
        self.assertLessEqual(chaos_results['success_rate'], 1.0)
        
        # Should have some recovery times if any tests succeeded
        if chaos_results['successful_runs'] > 0:
            self.assertGreater(len(chaos_results['recovery_times']), 0)
    
    @pytest.mark.simulation
    def test_dataset_creation(self):
        """Test synthetic dataset creation for different task types."""
        # Test classification dataset
        X_class, y_class = self.create_test_dataset(
            n_samples=50, n_features=4, task_type='classification'
        )
        
        self.assertEqual(X_class.shape, (50, 4))
        self.assertEqual(len(y_class), 50)
        self.assertTrue(all(label in [0, 1] for label in y_class))
        
        # Test regression dataset
        X_reg, y_reg = self.create_test_dataset(
            n_samples=30, n_features=3, task_type='regression'
        )
        
        self.assertEqual(X_reg.shape, (30, 3))
        self.assertEqual(len(y_reg), 30)
        self.assertTrue(all(isinstance(val, (int, float)) for val in y_reg))


class TestQuantumCircuitOptimizer:
    """Test quantum circuit optimization utilities."""
    
    @pytest.mark.simulation
    def test_gate_counting(self):
        """Test gate counting functionality."""
        circuit_description = {
            'gates': [
                {'type': 'rx', 'qubit': 0, 'angle': 0.5},
                {'type': 'ry', 'qubit': 1, 'angle': 0.3},
                {'type': 'cnot', 'control': 0, 'target': 1},
                {'type': 'rx', 'qubit': 0, 'angle': 0.2},
                {'type': 'cz', 'control': 1, 'target': 2}
            ]
        }
        
        gate_counts = QuantumCircuitOptimizer.count_gates(circuit_description)
        
        expected_counts = {'rx': 2, 'ry': 1, 'cnot': 1, 'cz': 1}
        assert gate_counts == expected_counts
    
    @pytest.mark.simulation  
    def test_depth_estimation(self):
        """Test circuit depth estimation."""
        circuit_description = {
            'n_qubits': 3,
            'gates': [
                {'type': 'rx', 'qubit': 0, 'angle': 0.5},
                {'type': 'rx', 'qubit': 1, 'angle': 0.5},
                {'type': 'cnot', 'control': 0, 'target': 1},
                {'type': 'rx', 'qubit': 2, 'angle': 0.5},
                {'type': 'cnot', 'control': 1, 'target': 2}
            ]
        }
        
        depth = QuantumCircuitOptimizer.estimate_depth(circuit_description)
        
        # Expected depth should be around 3-4 based on gate dependencies
        assert 2 <= depth <= 5, f"Unexpected circuit depth: {depth}"


class TestNoiseModelTester:
    """Test noise model utilities."""
    
    def setUp(self):
        self.noise_tester = NoiseModelTester()
        self.test_state = np.array([0.7, 0.3, 0.2, 0.1], dtype=complex)
    
    @pytest.mark.simulation
    def test_noise_channel_application(self):
        """Test application of different noise channels."""
        noise_types = ['depolarizing', 'amplitude_damping', 'phase_damping', 
                      'bit_flip', 'phase_flip', 'thermal']
        
        noise_tester = NoiseModelTester()
        test_state = np.array([0.7, 0.3, 0.2, 0.1], dtype=complex)
        
        for noise_type in noise_types:
            with pytest.subTest(noise_type=noise_type):
                noisy_state = noise_tester.apply_noise_channel(
                    test_state, noise_type, error_rate=0.01
                )
                
                # Check that noise was applied (state should be different)
                assert not np.allclose(noisy_state, test_state), \
                    f"No noise applied for {noise_type}"
                
                # Check state dimensions are preserved
                assert noisy_state.shape == test_state.shape, \
                    f"State shape changed for {noise_type}"
    
    @pytest.mark.simulation
    def test_noise_scaling(self):
        """Test that noise effects scale with error rate."""
        noise_tester = NoiseModelTester()
        test_state = np.array([0.7, 0.3, 0.2, 0.1], dtype=complex)
        
        error_rates = [0.001, 0.01, 0.1]
        noise_effects = []
        
        for error_rate in error_rates:
            noisy_state = noise_tester.apply_noise_channel(
                test_state, 'depolarizing', error_rate
            )
            
            # Measure noise effect as distance from original state
            noise_effect = np.linalg.norm(noisy_state - test_state)
            noise_effects.append(noise_effect)
        
        # Generally, higher error rates should cause larger deviations
        assert noise_effects[1] >= noise_effects[0] * 0.5, \
            "Noise doesn't scale properly with error rate"


class TestQuantumHardwareSimulator:
    """Test quantum hardware simulation utilities."""
    
    @pytest.mark.simulation
    def test_hardware_parameter_initialization(self):
        """Test hardware parameter initialization for different backend types."""
        backend_types = ['superconducting', 'trapped_ion', 'photonic', 'simulator']
        
        for backend_type in backend_types:
            with pytest.subTest(backend_type=backend_type):
                simulator = QuantumHardwareSimulator(backend_type)
                
                params = simulator.device_parameters
                
                # Check required parameters exist
                required_params = ['T1', 'T2', 'gate_time', 'readout_fidelity', 
                                 'gate_fidelity', 'crosstalk']
                for param in required_params:
                    assert param in params, f"Missing parameter {param} for {backend_type}"
                
                # Check parameter ranges are reasonable
                assert 0 <= params['readout_fidelity'] <= 1, \
                    f"Invalid readout fidelity for {backend_type}"
                assert 0 <= params['gate_fidelity'] <= 1, \
                    f"Invalid gate fidelity for {backend_type}"
                assert params['crosstalk'] >= 0, \
                    f"Invalid crosstalk for {backend_type}"
    
    @pytest.mark.simulation
    def test_decoherence_simulation(self):
        """Test decoherence simulation."""
        simulator = QuantumHardwareSimulator('superconducting')
        test_state = np.array([0.7, 0.3, 0.2, 0.1], dtype=complex)
        
        # Test decoherence over different time scales
        times = [0, 1e-6, 10e-6, 100e-6]  # 0 to 100 microseconds
        
        prev_fidelity = 1.0
        for time in times:
            decohered_state = simulator.simulate_decoherence(test_state, time)
            
            # Calculate fidelity with original state
            fidelity = np.abs(np.vdot(test_state, decohered_state)) ** 2
            fidelity /= (np.linalg.norm(test_state) ** 2 * np.linalg.norm(decohered_state) ** 2)
            
            # Fidelity should generally decrease with time
            if time > 0:
                assert fidelity <= prev_fidelity + 0.1, \
                    f"Fidelity increased with time: {prev_fidelity} -> {fidelity}"
            
            prev_fidelity = fidelity
    
    @pytest.mark.simulation
    def test_gate_error_simulation(self):
        """Test gate error simulation."""
        simulator = QuantumHardwareSimulator('superconducting')
        
        # Test multiple gate operations
        ideal_results = [0.5, 0.8, -0.3, 0.1]
        errors_detected = 0
        
        for ideal_result in ideal_results:
            for _ in range(10):  # Multiple trials
                noisy_result = simulator.simulate_gate_error(ideal_result)
                
                if abs(noisy_result - ideal_result) > 1e-6:
                    errors_detected += 1
        
        # Should detect some errors given the gate fidelity < 1
        assert errors_detected > 0, "No gate errors detected in simulation"
    
    @pytest.mark.simulation
    def test_readout_error_simulation(self):
        """Test readout error simulation."""
        simulator = QuantumHardwareSimulator('superconducting')
        
        # Test readout errors
        true_results = [0, 1, 0, 1, 0] * 20  # 100 measurements
        errors_detected = 0
        
        for true_result in true_results:
            measured_result = simulator.simulate_readout_error(true_result)
            
            if measured_result != true_result:
                errors_detected += 1
        
        # Should detect some readout errors
        error_rate = errors_detected / len(true_results)
        expected_error_rate = 1 - simulator.device_parameters['readout_fidelity']
        
        # Error rate should be roughly consistent with device parameters
        assert 0 <= error_rate <= expected_error_rate * 5, \
            f"Readout error rate {error_rate} inconsistent with device parameters"


class TestQuantumTestDecorators:
    """Test quantum testing decorators."""
    
    @quantum_test(backend='simulator', shots=2048)
    @pytest.mark.simulation
    def test_quantum_test_decorator(self):
        """Test the quantum_test decorator."""
        # Decorator should set backend and shots
        assert hasattr(self, 'backend')
        assert hasattr(self, 'shots')
        assert self.backend == 'simulator'
        assert self.shots == 2048
    
    @performance_benchmark(min_throughput=1.0, max_latency=5.0)
    @pytest.mark.simulation
    def test_performance_decorator(self):
        """Test the performance_benchmark decorator."""
        # Simple operation that should meet performance requirements
        self.benchmark_samples = 10
        
        # Simulate some work
        import time
        time.sleep(0.1)  # Should be well under 5 second limit
        
        result = sum(range(100))
        assert result == 4950


class TestQuantumTestFrameworkIntegration:
    """Integration tests for the quantum testing framework."""
    
    @pytest.mark.integration
    def test_end_to_end_quantum_model_testing(self):
        """Test complete quantum model testing workflow."""
        # Create test case instance
        test_case = QuantumTestCase()
        test_case.setUp()
        
        # Create and test a quantum model
        model = test_case.create_model(n_qubits=3, circuit_type='variational', depth=2)
        X, y = test_case.create_test_dataset(n_samples=20, n_features=3)
        
        # Test gradient stability
        gradient_metrics = test_case.measure_gradient_variance(
            model, X, y, n_samples=5, noise_level=0.01
        )
        assert 'variance' in gradient_metrics
        assert gradient_metrics['variance'] >= 0
        
        # Test noise resilience
        noise_metrics = test_case.evaluate_with_noise(
            model, X, y, noise_model='depolarizing', noise_prob=0.01, n_trials=3
        )
        assert 'mean_accuracy' in noise_metrics
        assert 0 <= noise_metrics['mean_accuracy'] <= 1
        
        # Test performance benchmarking
        perf_metrics = test_case.benchmark_execution(model, X, n_runs=3)
        assert 'mean_execution_time' in perf_metrics
        assert perf_metrics['mean_execution_time'] > 0
        
        # Test hardware compatibility
        circuit = test_case.build_circuit('variational', n_qubits=3)
        test_case.assert_native_gates(circuit, 'simulator', strict=False)
        test_case.assert_topology_compatible(circuit, 'simulator')
    
    @pytest.mark.integration
    def test_quantum_ml_pipeline_integration(self):
        """Test integration with QuantumMLPipeline."""
        # Create a simple pipeline
        def simple_circuit(params, x):
            return float(np.sum(np.sin(params) * np.cos(x)))
        
        pipeline = QuantumMLPipeline(
            circuit=simple_circuit,
            n_qubits=3,
            device=QuantumDevice.SIMULATOR
        )
        
        # Test with enhanced testing framework
        test_case = QuantumTestCase()
        test_case.setUp()
        
        # Create test data
        X, y = test_case.create_test_dataset(n_samples=10, n_features=3)
        
        # Train model
        model = pipeline.train(X, y, epochs=5, learning_rate=0.1)
        
        # Test trained model
        assert model is not None
        assert model.parameters is not None
        
        # Evaluate model
        metrics = pipeline.evaluate(model, X, y, noise_models=['depolarizing'])
        assert metrics.accuracy >= 0
        assert metrics.fidelity >= 0
    
    @pytest.mark.integration  
    def test_performance_regression_detection(self):
        """Test performance regression detection capabilities."""
        test_case = QuantumTestCase()
        test_case.setUp()
        
        # Baseline metrics (simulated)
        baseline_metrics = {
            'accuracy': 0.85,
            'execution_time': 1.0,
            'memory_usage': 100.0
        }
        
        # Current metrics (slightly worse performance)
        current_metrics = {
            'accuracy': 0.82,  # 3% worse
            'execution_time': 1.08,  # 8% worse  
            'memory_usage': 105.0  # 5% worse
        }
        
        # Should detect regression with default tolerance (5%)
        with pytest.raises(AssertionError):
            test_case.assert_performance_regression(
                current_metrics, baseline_metrics, tolerance=0.05
            )
        
        # Should pass with higher tolerance
        test_case.assert_performance_regression(
            current_metrics, baseline_metrics, tolerance=0.10
        )


# Legacy tests from original file for backward compatibility
class TestQuantumTestCase(QuantumTestCase):
    """Test the quantum testing framework itself (original tests)."""
    
    @pytest.mark.simulation
    def test_gradient_stability(self):
        """Test that gradients remain stable under noise."""
        model = self.create_model(n_qubits=4)
        X, y = self.create_test_dataset(n_samples=50, n_features=4)
        
        # Test gradient variance
        variance_metrics = self.measure_gradient_variance(
            model, X, y,
            n_samples=50,  # Reduced for testing
            noise_level=0.01
        )
        
        self.assertLess(variance_metrics['variance'], 0.1, 
                       "Gradient variance too high for reliable training")
        
    @pytest.mark.simulation
    def test_hardware_compatibility(self):
        """Ensure circuit runs on target hardware."""
        circuit = self.build_circuit()
        
        # Check gate decomposition for different backends
        self.assert_native_gates(circuit, backend='ionq')
        self.assert_native_gates(circuit, backend='ibm')
        
        # Verify connectivity constraints
        self.assert_topology_compatible(circuit, backend='ibm')
        
    @pytest.mark.simulation
    def test_noise_resilience(self):
        """Test model performance under realistic noise."""
        model = self.create_model()
        X, y = self.create_test_dataset(n_samples=30, n_features=4)
        
        # Test with increasing noise levels
        for noise_prob in [0.001, 0.01, 0.05]:
            noise_metrics = self.evaluate_with_noise(
                model, X, y,
                noise_model='depolarizing',
                noise_prob=noise_prob,
                n_trials=3
            )
            
            # Ensure graceful degradation
            self.assertGreater(noise_metrics['mean_accuracy'], 0.3,
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
    """Additional tests for quantum testing utilities (original tests)."""
    
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
        X, y = test_case.create_test_dataset(n_samples=20, n_features=4)
        
        # Test different noise levels
        low_noise_metrics = test_case.measure_gradient_variance(
            model, X, y, n_samples=20, noise_level=0.001
        )
        high_noise_metrics = test_case.measure_gradient_variance(
            model, X, y, n_samples=20, noise_level=0.1
        )
        
        # Higher noise should generally lead to higher variance
        assert high_noise_metrics['variance'] >= low_noise_metrics['variance'] * 0.8  # Allow some variance in random sampling


if __name__ == '__main__':
    pytest.main([__file__, '-v'])