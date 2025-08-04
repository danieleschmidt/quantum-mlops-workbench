#!/usr/bin/env python3
"""
Demonstration of quantum backend integration for the MLOps workbench.

This script demonstrates how to use the new quantum backend system with
PennyLane, Qiskit, and AWS Braket integrations.

Usage:
    python examples/quantum_backend_demo.py
    
Requirements:
    pip install numpy pennylane qiskit amazon-braket-sdk
"""

import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import numpy as np
    from quantum_mlops.core import QuantumMLPipeline, QuantumDevice
    from quantum_mlops.backends import QuantumExecutor, BackendManager
    
    def main():
        print("=== Quantum Backend Integration Demo ===\n")
        
        # 1. Demonstrate Backend Manager
        print("1. Setting up Backend Manager...")
        manager = BackendManager()
        print(f"   - Fallback order: {manager.fallback_order}")
        print(f"   - Retry attempts: {manager.retry_attempts}")
        
        # 2. Create Quantum Executor
        print("\n2. Creating Quantum Executor...")
        executor = QuantumExecutor(manager)
        
        # 3. Check available backends
        print("\n3. Checking Backend Status...")
        try:
            status = executor.get_backend_status()
            available = executor.list_available_backends()
            print(f"   - Available backends: {available}")
            
            for name, info in status.items():
                print(f"   - {name}: {'✓' if info.get('available') else '✗'} "
                      f"({'connected' if info.get('connected') else 'disconnected'})")
        except Exception as e:
            print(f"   - Backend status check failed: {e}")
        
        # 4. Demonstrate Circuit Execution
        print("\n4. Testing Circuit Execution...")
        
        # Simple test circuit
        test_circuit = {
            "gates": [
                {"type": "h", "qubit": 0},
                {"type": "cx", "control": 0, "target": 1},
                {"type": "ry", "qubit": 0, "angle": np.pi/4}
            ],
            "n_qubits": 2,
            "measurements": [{"type": "expectation", "wires": 0, "observable": "Z"}]
        }
        
        try:
            print("   - Executing test circuit...")
            result = executor.execute(test_circuit, shots=100)
            print(f"   - Execution successful!")
            print(f"   - Expectation value: {result.expectation_value:.4f}")
            print(f"   - Counts: {dict(list(result.counts.items())[:3])}...")  # Show first 3
            
        except Exception as e:
            print(f"   - Circuit execution failed: {e}")
        
        # 5. Demonstrate ML Pipeline Integration
        print("\n5. Testing ML Pipeline Integration...")
        
        def simple_circuit():
            """Simple quantum circuit for ML."""
            return 0.5  # Placeholder
            
        try:
            # Create pipeline with simulator
            pipeline = QuantumMLPipeline(
                simple_circuit, 
                n_qubits=2, 
                device=QuantumDevice.SIMULATOR,
                shots=100
            )
            
            print("   - Pipeline created successfully")
            
            # Get backend info
            info = pipeline.get_backend_info()
            print(f"   - Device: {info['device']}")
            print(f"   - Real backend available: {info['real_backend_available']}")
            
            # Generate sample data
            X_train = np.random.random((10, 2))
            y_train = np.random.randint(0, 2, 10)
            
            print("   - Training model...")
            model = pipeline.train(X_train, y_train, epochs=3)
            
            print(f"   - Model trained! Parameters shape: {model.parameters.shape}")
            
            # Test evaluation
            X_test = np.random.random((5, 2))
            y_test = np.random.randint(0, 2, 5)
            
            metrics = pipeline.evaluate(model, X_test, y_test)
            print(f"   - Model accuracy: {metrics.accuracy:.4f}")
            print(f"   - Model loss: {metrics.loss:.4f}")
            
        except Exception as e:
            print(f"   - Pipeline integration failed: {e}")
            import traceback
            traceback.print_exc()
        
        # 6. Backend Benchmarking
        print("\n6. Backend Benchmarking...")
        try:
            benchmark_results = executor.benchmark_backends(shots=50)
            
            print("   - Benchmark Results:")
            for backend_name, results in benchmark_results.items():
                if results.get("success"):
                    print(f"     * {backend_name}: {results['execution_time']:.4f}s "
                          f"(expectation: {results.get('expectation_value', 'N/A')})")
                else:
                    print(f"     * {backend_name}: Failed - {results.get('error', 'Unknown error')}")
                    
        except Exception as e:
            print(f"   - Benchmarking failed: {e}")
        
        # 7. Cost Estimation
        print("\n7. Cost Estimation...")
        try:
            available_backends = executor.list_available_backends()
            if available_backends:
                backend_name = available_backends[0]
                cost_info = executor.estimate_execution_cost(
                    test_circuit, 
                    backend_name, 
                    shots=1000
                )
                
                print(f"   - Cost for {backend_name}:")
                print(f"     * Total cost: ${cost_info['total_cost']:.6f}")
                print(f"     * Per shot: ${cost_info['cost_per_shot']:.6f}")
                print(f"     * Estimated time: {cost_info.get('estimated_time_minutes', 0):.2f} min")
                
        except Exception as e:
            print(f"   - Cost estimation failed: {e}")
        
        print("\n=== Demo Complete ===")
        print("\nNext Steps:")
        print("- Install quantum libraries: pip install pennylane qiskit amazon-braket-sdk")
        print("- Configure IBM Quantum credentials for real hardware access")
        print("- Setup AWS credentials for Braket cloud access")
        print("- Explore the quantum_mlops.backends module for advanced features")
        
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Import error: {e}")
    print("\nRequired dependencies not available.")
    print("This demo requires: numpy, and optionally pennylane, qiskit, amazon-braket-sdk")
    print("\nTo install dependencies:")
    print("  pip install numpy")
    print("  pip install pennylane pennylane-lightning")  
    print("  pip install qiskit qiskit-machine-learning")
    print("  pip install amazon-braket-sdk")
    sys.exit(1)
    
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)