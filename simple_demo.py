#!/usr/bin/env python3
"""
Simple Quantum MLOps Demo - Generation 1: MAKE IT WORK
Demonstrates basic functionality with minimal viable features.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from quantum_mlops import QuantumMLPipeline, QuantumDevice

def create_simple_circuit(params, x):
    """Simple quantum circuit for demo."""
    # Simplified quantum circuit simulation
    n_params = len(params) if hasattr(params, '__len__') else 1
    n_features = len(x) if hasattr(x, '__len__') else 1
    
    # Simulate quantum computation with classical operations
    result = np.sum(params) * np.sum(x) / (n_params * n_features + 1e-8)
    return np.tanh(result)  # Bounded output

def main():
    """Demo the quantum MLOps pipeline."""
    print("🚀 Generation 1: MAKE IT WORK - Simple Quantum MLOps Demo")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        print("1. Initializing Quantum ML Pipeline...")
        pipeline = QuantumMLPipeline(
            circuit=create_simple_circuit,
            n_qubits=4,
            device=QuantumDevice.SIMULATOR,
            layers=2
        )
        print("   ✅ Pipeline initialized successfully")
        
        # Generate synthetic data
        print("\n2. Generating synthetic training data...")
        X_train = np.random.rand(50, 4)
        y_train = np.random.randint(0, 2, 50)
        X_test = np.random.rand(20, 4)
        y_test = np.random.randint(0, 2, 20)
        print(f"   ✅ Training data: {X_train.shape}, Test data: {X_test.shape}")
        
        # Train model
        print("\n3. Training quantum model...")
        model = pipeline.train(
            X_train, y_train,
            epochs=5,  # Quick training for demo
            learning_rate=0.01
        )
        print("   ✅ Model training completed")
        
        # Evaluate model
        print("\n4. Evaluating model performance...")
        metrics = pipeline.evaluate(model, X_test, y_test)
        print(f"   ✅ Accuracy: {metrics.accuracy:.2%}")
        print(f"   ✅ Loss: {metrics.loss:.4f}")
        
        # Simple prediction
        print("\n5. Making predictions...")
        sample = X_test[0:1]
        # Use forward pass for prediction (pipeline doesn't have predict method)
        prediction = pipeline._forward_pass(model, sample)
        print(f"   ✅ Sample prediction: {prediction[0]:.4f}")
        
        print("\n🎉 Generation 1 Demo Complete!")
        print("✅ Basic quantum ML pipeline is working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)