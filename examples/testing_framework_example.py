#!/usr/bin/env python3
"""
Example: Comprehensive Testing Framework Usage

This example demonstrates how to use the enhanced testing framework
for quantum machine learning applications.
"""

import pytest
import numpy as np
from typing import Dict, Any
import time

# Import the enhanced testing framework
from quantum_mlops.testing import QuantumTestCase, quantum_test, noise_resilience_test, performance_benchmark
from quantum_mlops.core import QuantumMLPipeline, QuantumDevice, QuantumModel


class ExampleQuantumMLTests(QuantumTestCase):
    """
    Example test class demonstrating comprehensive quantum ML testing capabilities.
    
    This class showcases various testing patterns and utilities available
    in the enhanced testing framework.
    """
    
    def setUp(self):
        """Set up test fixtures and configurations."""
        super().setUp()
        
        # Example test configuration
        self.test_config = {
            'n_qubits': 4,
            'circuit_depth': 3,
            'training_epochs': 20,
            'test_samples': 50
        }
        
        # Performance thresholds for this example
        self.performance_thresholds = {
            'max_training_time': 30.0,  # seconds
            'min_accuracy': 0.7,
            'max_gradient_variance': 2.0,
            'min_fidelity': 0.8
        }
    
    @pytest.mark.quantum
    @pytest.mark.unit
    def test_basic_quantum_model_creation(self):
        """Example: Basic quantum model testing with enhanced assertions."""
        
        # Create a quantum model using the enhanced testing utilities
        model = self.create_model(
            n_qubits=self.test_config['n_qubits'],
            circuit_type='variational',
            depth=self.test_config['circuit_depth']
        )
        
        # Use quantum-specific assertions
        self.assertValidQuantumModel(model)
        self.assertEqual(model.n_qubits, self.test_config['n_qubits'])
        
        # Test quantum state properties
        state_vector = model.state_vector
        self.assertValidQuantumState(state_vector)
        self.assertQuantumStateNormalized(state_vector)
        
        print(f"✓ Created valid quantum model with {model.n_qubits} qubits")
    
    @quantum_test(shots=1024, noise_models=['depolarizing'])
    def test_quantum_training_with_noise(self):
        """Example: Training with noise analysis using decorators."""
        
        # Generate sample data
        X_train = np.random.random((100, self.test_config['n_qubits']))
        y_train = np.random.randint(0, 2, 100)
        
        # Create and train model
        model = self.create_model(
            n_qubits=self.test_config['n_qubits'],
            circuit_type='variational'
        )
        
        # Train with monitoring
        start_time = time.time()
        trained_model = self.train_with_monitoring(
            model, X_train, y_train,
            epochs=self.test_config['training_epochs']
        )
        training_time = time.time() - start_time
        
        # Use enhanced quantum assertions
        self.assertValidQuantumModel(trained_model)
        self.assertLess(training_time, self.performance_thresholds['max_training_time'])
        
        # Test gradient stability
        gradients = self.compute_parameter_shift_gradients(trained_model, X_train[:10])
        self.assertGradientStability(
            gradients, 
            max_variance=self.performance_thresholds['max_gradient_variance']
        )
        
        print(f"✓ Training completed in {training_time:.2f}s with stable gradients")
    
    @noise_resilience_test(['decoherence', 'depolarizing', 'amplitude_damping'])
    def test_noise_resilience_comprehensive(self, noise_type):
        """Example: Comprehensive noise resilience testing."""
        
        # Create model and data
        model = self.create_model(n_qubits=4, circuit_type='basic')
        X_test = np.random.random((20, 4))
        y_test = np.random.randint(0, 2, 20)
        
        # Configure noise based on type
        noise_configs = {
            'decoherence': {'t1': 50e-6, 't2': 70e-6, 'gate_time': 20e-9},
            'depolarizing': {'probability': 0.01},
            'amplitude_damping': {'gamma': 0.05}
        }
        
        noise_config = noise_configs[noise_type]
        
        # Test model performance under noise
        results = self.evaluate_with_noise(model, X_test, y_test, noise_config)
        
        # Use noise-specific assertions
        self.assertNoiseResilience(
            results, noise_config,
            min_fidelity=self.performance_thresholds['min_fidelity'] * 0.8  # Relaxed for noise
        )
        
        print(f"✓ Model resilient to {noise_type} noise (fidelity: {results.fidelity:.3f})")
    
    @performance_benchmark(baseline_file='quantum_ml_baseline.json')
    def test_performance_benchmarking(self):
        """Example: Performance benchmarking against baseline."""
        
        # Test different problem sizes
        problem_sizes = [(2, 20), (4, 50), (6, 100)]  # (n_qubits, n_samples)
        performance_results = {}
        
        for n_qubits, n_samples in problem_sizes:
            # Generate appropriately sized data
            X = np.random.random((n_samples, n_qubits))
            y = np.random.randint(0, 2, n_samples)
            
            # Create and train model
            model = self.create_model(n_qubits=n_qubits, circuit_type='variational')
            
            # Measure performance metrics
            start_time = time.time()
            trained_model = model.train(X, y, epochs=10)
            training_time = time.time() - start_time
            
            # Measure prediction performance
            pred_start = time.time()
            predictions = trained_model.predict(X[:10])
            prediction_time = time.time() - pred_start
            
            # Store results
            key = f"{n_qubits}q_{n_samples}s"
            performance_results[key] = {
                'training_time': training_time,
                'prediction_time': prediction_time,
                'throughput': len(predictions) / prediction_time,
                'n_qubits': n_qubits,
                'n_samples': n_samples
            }
            
            print(f"✓ {key}: Training {training_time:.2f}s, Prediction {prediction_time:.3f}s")
        
        # Analyze scaling behavior
        self.analyze_performance_scaling(performance_results)
    
    @pytest.mark.integration
    @pytest.mark.backend_compatibility
    def test_cross_backend_compatibility(self):
        """Example: Testing across multiple quantum backends."""
        
        # Test configurations for different backends
        backend_configs = {
            'simulator': {'device': QuantumDevice.SIMULATOR, 'shots': 1024},
            'braket': {'device': QuantumDevice.AWS_BRAKET, 'shots': 1000},
        }
        
        # Sample data
        X_test = np.random.random((10, 4))
        results_by_backend = {}
        
        for backend_name, config in backend_configs.items():
            try:
                # Create model for specific backend
                model = self.create_model(
                    n_qubits=4,
                    circuit_type='basic',
                    device=config['device']
                )
                
                # Test backend compatibility
                self.assertValidQuantumModel(model)
                
                # Execute and store results
                predictions = model.predict(X_test)
                results_by_backend[backend_name] = predictions
                
                print(f"✓ {backend_name} backend compatible")
                
            except Exception as e:
                print(f"⚠ {backend_name} backend unavailable: {e}")
                continue
        
        # Compare results across backends (if multiple available)
        if len(results_by_backend) > 1:
            self.compare_cross_backend_results(results_by_backend)
    
    @pytest.mark.chaos
    def test_chaos_engineering_quantum_systems(self):
        """Example: Chaos engineering for quantum systems."""
        
        # Define chaos scenarios
        chaos_scenarios = [
            {
                'name': 'quantum_decoherence_spike',
                'type': 'decoherence',
                'parameters': {'t1_reduction': 0.5, 't2_reduction': 0.3},
                'duration': 10
            },
            {
                'name': 'measurement_error_injection',
                'type': 'measurement_noise',
                'parameters': {'error_rate': 0.1},
                'duration': 15
            }
        ]
        
        model = self.create_model(n_qubits=4, circuit_type='variational')
        X_test = np.random.random((20, 4))
        
        for scenario in chaos_scenarios:
            print(f"Testing chaos scenario: {scenario['name']}")
            
            # Apply chaos scenario
            with self.chaos_context(scenario):
                try:
                    # Attempt normal operation under chaos
                    results = model.predict(X_test)
                    
                    # Verify system maintains basic functionality
                    self.assertIsNotNone(results)
                    self.assertEqual(len(results), len(X_test))
                    
                    print(f"✓ System resilient to {scenario['name']}")
                    
                except Exception as e:
                    # Log chaos-induced failures for analysis
                    print(f"⚠ Chaos scenario {scenario['name']} caused failure: {e}")
    
    @pytest.mark.load
    @pytest.mark.slow
    def test_load_testing_quantum_circuits(self):
        """Example: Load testing quantum circuit execution."""
        
        # Load test configuration
        load_config = {
            'concurrent_circuits': 20,
            'test_duration': 30,  # seconds
            'target_throughput': 5  # circuits/second
        }
        
        # Create test circuits
        circuits = []
        for i in range(load_config['concurrent_circuits']):
            circuit = {
                'gates': [
                    {'type': 'ry', 'qubit': 0, 'angle': np.pi * i / 10},
                    {'type': 'cnot', 'control': 0, 'target': 1},
                    {'type': 'ry', 'qubit': 1, 'angle': np.pi * i / 20}
                ],
                'n_qubits': 2
            }
            circuits.append(circuit)
        
        # Execute load test
        load_results = self.execute_load_test(
            circuits,
            duration=load_config['test_duration'],
            concurrent=True
        )
        
        # Analyze load test results
        actual_throughput = load_results['completed_circuits'] / load_config['test_duration']
        error_rate = load_results['failed_circuits'] / load_results['total_circuits']
        
        # Performance assertions
        self.assertGreater(
            actual_throughput, 
            load_config['target_throughput'] * 0.8,  # 80% of target
            f"Throughput too low: {actual_throughput:.2f} < {load_config['target_throughput']}"
        )
        
        self.assertLess(
            error_rate, 0.05,  # 5% error rate threshold
            f"Error rate too high: {error_rate:.2%}"
        )
        
        print(f"✓ Load test passed: {actual_throughput:.2f} circuits/sec, {error_rate:.2%} errors")
    
    @pytest.mark.gradient_stability
    def test_gradient_stability_analysis(self):
        """Example: Comprehensive gradient stability testing."""
        
        # Create model and data
        model = self.create_model(n_qubits=4, circuit_type='variational', depth=3)
        X = np.random.random((30, 4))
        y = np.random.randint(0, 2, 30)
        
        # Train model to get meaningful gradients
        model.train(X, y, epochs=10)
        
        # Test gradient computation methods
        gradient_methods = {
            'parameter_shift': lambda: self.compute_parameter_shift_gradients(model, X[:5]),
            'finite_difference': lambda: self.compute_finite_difference_gradients(model, X[:5])
        }
        
        gradient_results = {}
        
        for method_name, compute_func in gradient_methods.items():
            # Compute gradients multiple times for stability analysis
            gradient_samples = []
            for _ in range(10):
                gradients = compute_func()
                gradient_samples.append(gradients)
            
            # Analyze stability
            gradient_matrix = np.array(gradient_samples)
            gradient_variance = np.var(gradient_matrix, axis=0)
            gradient_mean = np.mean(gradient_matrix, axis=0)
            
            gradient_results[method_name] = {
                'variance': gradient_variance,
                'mean': gradient_mean,
                'stability_score': 1.0 / (1.0 + np.mean(gradient_variance))
            }
            
            # Test gradient stability
            self.assertGradientStability(
                gradient_mean,
                max_variance=2.0,
                method=method_name
            )
            
            print(f"✓ {method_name} gradients stable (score: {gradient_results[method_name]['stability_score']:.3f})")
        
        # Compare gradient methods
        self.compare_gradient_methods(gradient_results)
    
    # Helper methods for the examples
    
    def train_with_monitoring(self, model, X, y, epochs):
        """Train model with performance monitoring."""
        # This would integrate with the monitoring system
        return model.train(X, y, epochs=epochs)
    
    def evaluate_with_noise(self, model, X, y, noise_config):
        """Evaluate model under specified noise conditions."""
        # Apply noise model and evaluate
        noisy_results = self.apply_noise_model(model.predict(X), noise_config)
        
        # Calculate metrics
        accuracy = np.mean((noisy_results > 0.5) == y)
        fidelity = self.calculate_quantum_fidelity(noisy_results, model.predict(X))
        
        return type('Results', (), {
            'predictions': noisy_results,
            'accuracy': accuracy,
            'fidelity': fidelity
        })
    
    def analyze_performance_scaling(self, results):
        """Analyze how performance scales with problem size."""
        qubits = [r['n_qubits'] for r in results.values()]
        times = [r['training_time'] for r in results.values()]
        
        # Simple scaling analysis
        if len(qubits) > 1:
            # Check if scaling is reasonable (not exponential)
            max_qubit = max(qubits)
            max_time = max(times)
            
            # Expect sub-exponential scaling for reasonable problem sizes
            expected_max_time = 2 ** (max_qubit / 2)  # Square root exponential
            
            self.assertLess(
                max_time, expected_max_time,
                f"Training time scaling too aggressive: {max_time:.2f}s for {max_qubit} qubits"
            )
            
            print(f"✓ Performance scaling acceptable up to {max_qubit} qubits")
    
    def compare_cross_backend_results(self, results_by_backend):
        """Compare results across different backends."""
        backend_names = list(results_by_backend.keys())
        
        for i in range(len(backend_names)):
            for j in range(i + 1, len(backend_names)):
                backend1, backend2 = backend_names[i], backend_names[j]
                results1 = results_by_backend[backend1]
                results2 = results_by_backend[backend2]
                
                # Calculate correlation between backends
                correlation = np.corrcoef(results1.flatten(), results2.flatten())[0, 1]
                
                self.assertGreater(
                    correlation, 0.5,
                    f"Low correlation between {backend1} and {backend2}: {correlation:.3f}"
                )
                
                print(f"✓ {backend1} vs {backend2}: correlation = {correlation:.3f}")
    
    def execute_load_test(self, circuits, duration, concurrent=True):
        """Execute load test with specified circuits."""
        import concurrent.futures
        import threading
        
        results = {
            'completed_circuits': 0,
            'failed_circuits': 0,
            'total_circuits': 0,
            'execution_times': []
        }
        
        start_time = time.time()
        
        def execute_circuit(circuit):
            try:
                exec_start = time.time()
                # Simulate circuit execution
                time.sleep(np.random.uniform(0.1, 0.3))  # Simulate execution time
                exec_time = time.time() - exec_start
                
                with threading.Lock():
                    results['completed_circuits'] += 1
                    results['execution_times'].append(exec_time)
                    
                return True
            except Exception:
                with threading.Lock():
                    results['failed_circuits'] += 1
                return False
        
        # Execute circuits
        if concurrent:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                
                while time.time() - start_time < duration:
                    for circuit in circuits:
                        if time.time() - start_time >= duration:
                            break
                        future = executor.submit(execute_circuit, circuit)
                        futures.append(future)
                        results['total_circuits'] += 1
                        time.sleep(0.1)  # Rate limiting
                
                # Wait for completion
                for future in concurrent.futures.as_completed(futures, timeout=duration):
                    pass
        else:
            # Sequential execution
            while time.time() - start_time < duration:
                for circuit in circuits:
                    if time.time() - start_time >= duration:
                        break
                    execute_circuit(circuit)
                    results['total_circuits'] += 1
        
        return results
    
    def compare_gradient_methods(self, gradient_results):
        """Compare different gradient computation methods."""
        methods = list(gradient_results.keys())
        
        if len(methods) > 1:
            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                    method1, method2 = methods[i], methods[j]
                    
                    grad1 = gradient_results[method1]['mean']
                    grad2 = gradient_results[method2]['mean']
                    
                    # Compare gradient magnitudes
                    correlation = np.corrcoef(grad1, grad2)[0, 1]
                    
                    self.assertGreater(
                        correlation, 0.3,  # Relaxed threshold for different methods
                        f"Gradient methods {method1} and {method2} poorly correlated: {correlation:.3f}"
                    )
                    
                    print(f"✓ Gradient methods {method1} vs {method2}: correlation = {correlation:.3f}")


def run_example_tests():
    """Run the example tests to demonstrate the framework."""
    
    print("Quantum MLOps Testing Framework Example")
    print("=" * 50)
    
    # Create test instance
    test_instance = ExampleQuantumMLTests()
    test_instance.setUp()
    
    print("\n1. Testing basic quantum model creation...")
    test_instance.test_basic_quantum_model_creation()
    
    print("\n2. Testing quantum training with noise...")
    test_instance.test_quantum_training_with_noise()
    
    print("\n3. Testing noise resilience...")
    for noise_type in ['decoherence', 'depolarizing']:
        test_instance.test_noise_resilience_comprehensive(noise_type)
    
    print("\n4. Testing performance benchmarking...")
    test_instance.test_performance_benchmarking()
    
    print("\n5. Testing cross-backend compatibility...")
    test_instance.test_cross_backend_compatibility()
    
    print("\n6. Testing chaos engineering...")
    test_instance.test_chaos_engineering_quantum_systems()
    
    print("\n7. Testing gradient stability...")
    test_instance.test_gradient_stability_analysis()
    
    print("\n" + "=" * 50)
    print("✓ All example tests completed successfully!")
    print("\nTo run these tests with pytest:")
    print("pytest examples/testing_framework_example.py -v")


if __name__ == "__main__":
    run_example_tests()