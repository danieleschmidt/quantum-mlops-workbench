#!/usr/bin/env python3
"""
Quantum Advantage Analysis Demo

This example demonstrates the comprehensive quantum advantage detection framework,
showcasing cutting-edge algorithms for analyzing quantum advantage across multiple
dimensions including kernel methods, variational algorithms, noise resilience,
and quantum supremacy.

This is a research-grade implementation suitable for academic publication and
production quantum ML systems.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, Any, List

# Import quantum MLOps framework
from quantum_mlops import (
    QuantumMLPipeline,
    QuantumDevice,
    QuantumAdvantageTester,
    AdvantageAnalysisEngine,
    QuantumKernelAnalyzer,
    VariationalAdvantageAnalyzer,
    NoiseResilientTester,
    QuantumSupremacyAnalyzer
)

# Import classical ML models for comparison
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def create_sample_datasets() -> Dict[str, Any]:
    """Create sample datasets for quantum advantage testing."""
    
    datasets = {}
    
    # Linear separable dataset
    X_linear, y_linear = make_classification(
        n_samples=200, n_features=4, n_redundant=0, n_informative=4,
        n_clusters_per_class=1, random_state=42
    )
    
    # Non-linear dataset (moons)
    X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)
    
    # High-dimensional dataset
    X_high_dim, y_high_dim = make_classification(
        n_samples=200, n_features=8, n_redundant=2, n_informative=6,
        n_clusters_per_class=2, random_state=42
    )
    
    # Normalize datasets
    scaler = StandardScaler()
    
    datasets['linear'] = {
        'X': scaler.fit_transform(X_linear),
        'y': y_linear,
        'name': 'Linear Separable'
    }
    
    datasets['moons'] = {
        'X': scaler.fit_transform(X_moons),
        'y': y_moons,
        'name': 'Moons (Non-linear)'
    }
    
    datasets['high_dim'] = {
        'X': scaler.fit_transform(X_high_dim),
        'y': y_high_dim,
        'name': 'High Dimensional'
    }
    
    return datasets


def create_quantum_circuit(n_qubits: int = 4) -> Any:
    """Create a sample quantum circuit for demonstration."""
    
    try:
        import pennylane as qml
        
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev)
        def circuit(params, x):
            # Data encoding
            for i in range(min(len(x), n_qubits)):
                qml.RY(x[i], wires=i)
            
            # Variational layers
            n_layers = len(params) // (n_qubits * 2)
            param_idx = 0
            
            for layer in range(n_layers):
                # Single-qubit rotations
                for i in range(n_qubits):
                    qml.RY(params[param_idx], wires=i)
                    param_idx += 1
                    qml.RZ(params[param_idx], wires=i)
                    param_idx += 1
                
                # Entangling gates
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            return qml.expval(qml.PauliZ(0))
        
        return circuit
        
    except ImportError:
        print("PennyLane not available, using mock quantum circuit")
        
        def mock_circuit(params, x):
            return np.random.rand()  # Mock quantum computation
        
        return mock_circuit


def demo_quantum_kernel_advantage():
    """Demonstrate quantum kernel advantage analysis."""
    
    print("\n🔬 QUANTUM KERNEL ADVANTAGE ANALYSIS")
    print("="*60)
    
    # Initialize quantum kernel analyzer
    kernel_analyzer = QuantumKernelAnalyzer(
        n_qubits=4,
        feature_map="iqp",  # Use IQP feature map
        shots=1000
    )
    
    # Create test dataset
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Run comprehensive kernel advantage analysis
    result = kernel_analyzer.comprehensive_advantage_analysis(X, y)
    
    # Display results
    print(f"\n📊 KERNEL ADVANTAGE RESULTS:")
    print(f"  Spectral Advantage: {result.spectral_advantage:.4f}")
    print(f"  Expressivity Advantage: {result.expressivity_advantage:.4f}")
    print(f"  Performance Advantage: {result.performance_advantage:.4f}")
    print(f"  Overall Score: {result.overall_advantage_score:.4f}")
    print(f"  Advantage Category: {result.advantage_category}")
    print(f"  Statistical Significance: {result.statistically_significant}")
    
    return result


def demo_variational_advantage():
    """Demonstrate variational quantum advantage analysis."""
    
    print("\n🧮 VARIATIONAL QUANTUM ADVANTAGE ANALYSIS")
    print("="*60)
    
    # Initialize variational analyzer
    variational_analyzer = VariationalAdvantageAnalyzer(
        n_qubits=6,
        algorithm="vqe",
        n_layers=3
    )
    
    # Define a simple cost function
    def cost_function(params):
        # Simple quadratic cost with some structure
        return np.sum(params**2) + 0.1 * np.sum(np.sin(params))
    
    print(f"Cost function: Quadratic with trigonometric perturbation")
    print(f"Parameter space: {variational_analyzer.n_params} dimensions")
    
    # Run comprehensive variational advantage analysis
    result = variational_analyzer.comprehensive_advantage_analysis(cost_function)
    
    # Display results
    print(f"\n📊 VARIATIONAL ADVANTAGE RESULTS:")
    print(f"  Landscape Advantage: {result.landscape_advantage:.4f}")
    print(f"  Expressivity Advantage: {result.expressivity_advantage:.4f}")
    print(f"  Cost Advantage: {result.cost_advantage:.4f}")
    print(f"  Gradient Variance: {result.gradient_variance:.6f}")
    print(f"  Barren Plateau Detected: {result.plateau_detected}")
    print(f"  Overall Score: {result.overall_advantage_score:.4f}")
    print(f"  Advantage Category: {result.advantage_category}")
    
    return result


def demo_noise_resilient_advantage():
    """Demonstrate noise-resilient quantum advantage analysis."""
    
    print("\n🔧 NOISE-RESILIENT ADVANTAGE ANALYSIS")
    print("="*60)
    
    # Initialize noise-resilient tester
    noise_tester = NoiseResilientTester(
        n_qubits=4,
        circuit_depth=8,
        shots=1000
    )
    
    # Create quantum circuit
    quantum_circuit = create_quantum_circuit(4)
    
    # Create classical model for comparison
    X_dummy = np.random.rand(100, 4)
    y_dummy = np.random.randint(0, 2, 100)
    classical_model = RandomForestClassifier(n_estimators=50, random_state=42)
    classical_model.fit(X_dummy, y_dummy)
    
    print(f"Testing circuit depth: 8")
    print(f"Noise models: Depolarizing, Amplitude Damping, Thermal")
    
    # Run noise-resilient analysis
    result = noise_tester.comprehensive_noise_advantage_analysis(
        quantum_circuit,
        classical_model,
        dataset=(X_dummy, y_dummy)
    )
    
    # Display results
    print(f"\n📊 NOISE RESILIENCE RESULTS:")
    print(f"  Noise Resilience Score: {result.noise_resilience_score:.4f}")
    print(f"  Advantage Lost Threshold: {result.advantage_lost_threshold:.4f}")
    print(f"  Error Mitigation Improvement: {result.mitigation_improvement:.4f}")
    print(f"  Overall Score: {result.noise_resilient_advantage_score:.4f}")
    print(f"  Advantage Category: {result.advantage_category}")
    
    return result


def demo_quantum_supremacy_analysis():
    """Demonstrate quantum supremacy analysis."""
    
    print("\n🚀 QUANTUM SUPREMACY ANALYSIS")
    print("="*60)
    
    # Initialize supremacy analyzer
    supremacy_analyzer = QuantumSupremacyAnalyzer(
        max_qubits=12,
        max_circuit_depth=15,
        shots=1000
    )
    
    problem_sizes = [4, 6, 8, 10, 12]
    print(f"Testing problem sizes: {problem_sizes}")
    
    # Run comprehensive supremacy analysis
    result = supremacy_analyzer.comprehensive_supremacy_analysis(
        problem_sizes=problem_sizes
    )
    
    # Display results
    print(f"\n📊 QUANTUM SUPREMACY RESULTS:")
    print(f"  Quantum Scaling Exponent: {result.quantum_scaling_exponent:.3f}")
    print(f"  Classical Scaling Exponent: {result.classical_scaling_exponent:.3f}")
    print(f"  Scaling Advantage: {result.scaling_advantage:.3f}")
    print(f"  Crossover Point: {result.crossover_point} qubits")
    print(f"  Sample Efficiency Advantage: {result.sample_efficiency_advantage:.3f}")
    print(f"  Supremacy Achieved: {result.supremacy_achieved}")
    print(f"  Supremacy Confidence: {result.supremacy_confidence:.3f}")
    print(f"  Supremacy Category: {result.supremacy_category}")
    
    return result


def demo_comprehensive_analysis():
    """Demonstrate comprehensive quantum advantage analysis."""
    
    print("\n🔍 COMPREHENSIVE QUANTUM ADVANTAGE ANALYSIS")
    print("="*80)
    
    # Create datasets
    datasets = create_sample_datasets()
    dataset_name = 'moons'  # Use non-linear dataset
    data = datasets[dataset_name]
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        data['X'], data['y'], test_size=0.3, random_state=42
    )
    
    print(f"Dataset: {data['name']}")
    print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    
    # Initialize quantum advantage analysis engine
    engine = AdvantageAnalysisEngine(
        n_qubits=min(8, X_train.shape[1] + 2),  # Ensure sufficient qubits
        enable_kernel_analysis=True,
        enable_variational_analysis=True,
        enable_noise_analysis=True,
        enable_supremacy_analysis=True,
        seed=42
    )
    
    # Prepare analysis configuration
    analysis_config = {
        'X': X_train,
        'y': y_train,
        'problem_type': 'classification',
        'analysis_types': ['kernel', 'variational', 'noise_resilient', 'supremacy']
    }
    
    # Add cost function for variational analysis
    def cost_function(params):
        return np.sum(params**2) + 0.1 * np.sum(np.cos(params))
    
    analysis_config['cost_function'] = cost_function
    
    # Add quantum circuit
    analysis_config['quantum_circuit'] = create_quantum_circuit(engine.n_qubits)
    
    # Add classical model
    classical_model = RandomForestClassifier(n_estimators=100, random_state=42)
    classical_model.fit(X_train, y_train)
    analysis_config['classical_model'] = classical_model
    
    print(f"\n🔄 Running comprehensive analysis...")
    start_time = time.time()
    
    # Run comprehensive analysis
    result = engine.comprehensive_analysis(analysis_config)
    
    analysis_time = time.time() - start_time
    print(f"Analysis completed in {analysis_time:.2f} seconds")
    
    # Display comprehensive results
    print(f"\n📊 COMPREHENSIVE RESULTS:")
    print(f"  Overall Advantage Score: {result.overall_advantage_score:.4f}")
    print(f"  Advantage Confidence: {result.advantage_confidence.value}")
    print(f"  Advantage Category: {result.advantage_category}")
    print(f"  Statistical Significance: {result.statistical_significance}")
    print(f"  Practical Feasibility: {result.practical_feasibility}")
    
    # Key advantages
    print(f"\n🎯 KEY ADVANTAGES:")
    for advantage in result.key_advantages:
        print(f"  ✓ {advantage}")
    
    # Limitations
    print(f"\n⚠️  LIMITATIONS:")
    for limitation in result.limitations:
        print(f"  • {limitation}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    for recommendation in result.recommendations:
        print(f"  → {recommendation}")
    
    # Performance summary
    print(f"\n📈 PERFORMANCE SUMMARY:")
    for metric, value in result.performance_summary.items():
        print(f"  {metric}: {value}")
    
    # Generate and display report
    report = engine.generate_report(result)
    print(f"\n📄 DETAILED REPORT:")
    print(report)
    
    # Export results
    try:
        engine.export_results(result, "/tmp/quantum_advantage_results.json")
        print(f"\n💾 Results exported to /tmp/quantum_advantage_results.json")
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
    
    return result


def plot_advantage_comparison(results: Dict[str, Any]):
    """Plot quantum advantage comparison across different analyses."""
    
    try:
        import matplotlib.pyplot as plt
        
        # Extract advantage scores
        analyses = []
        scores = []
        
        if 'kernel' in results:
            analyses.append('Kernel\nAdvantage')
            scores.append(results['kernel'].overall_advantage_score)
        
        if 'variational' in results:
            analyses.append('Variational\nAdvantage')
            scores.append(results['variational'].overall_advantage_score)
        
        if 'noise_resilient' in results:
            analyses.append('Noise\nResilient')
            scores.append(results['noise_resilient'].noise_resilient_advantage_score)
        
        if 'supremacy' in results:
            analyses.append('Quantum\nSupremacy')
            scores.append(results['supremacy'].supremacy_confidence)
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = plt.bar(analyses, scores, color=colors[:len(analyses)])
        
        plt.title('Quantum Advantage Analysis Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Advantage Score', fontsize=12)
        plt.xlabel('Analysis Type', fontsize=12)
        plt.ylim(0, 1.0)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add horizontal line for significance threshold
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, 
                   label='Significance Threshold')
        
        plt.legend()
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig('/tmp/quantum_advantage_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Comparison plot saved to /tmp/quantum_advantage_comparison.png")
        
        plt.show()
        
    except ImportError:
        print("\n📊 Matplotlib not available for plotting")
    except Exception as e:
        print(f"\n❌ Plotting failed: {e}")


def main():
    """Run comprehensive quantum advantage analysis demo."""
    
    print("🌌 QUANTUM ADVANTAGE ANALYSIS FRAMEWORK DEMO")
    print("=" * 80)
    print("This demo showcases cutting-edge quantum advantage detection algorithms")
    print("suitable for academic research and production quantum ML systems.\n")
    
    results = {}
    
    try:
        # Demo individual analyses
        print("\n🚀 Running individual advantage analyses...")
        
        # Kernel advantage analysis
        try:
            results['kernel'] = demo_quantum_kernel_advantage()
        except Exception as e:
            print(f"❌ Kernel analysis failed: {e}")
        
        # Variational advantage analysis
        try:
            results['variational'] = demo_variational_advantage()
        except Exception as e:
            print(f"❌ Variational analysis failed: {e}")
        
        # Noise-resilient analysis
        try:
            results['noise_resilient'] = demo_noise_resilient_advantage()
        except Exception as e:
            print(f"❌ Noise analysis failed: {e}")
        
        # Quantum supremacy analysis
        try:
            results['supremacy'] = demo_quantum_supremacy_analysis()
        except Exception as e:
            print(f"❌ Supremacy analysis failed: {e}")
        
        # Comprehensive analysis
        try:
            results['comprehensive'] = demo_comprehensive_analysis()
        except Exception as e:
            print(f"❌ Comprehensive analysis failed: {e}")
        
        # Plot comparison
        plot_advantage_comparison(results)
        
        # Summary
        print(f"\n🎉 DEMO COMPLETE!")
        print(f"Completed {len(results)} quantum advantage analyses")
        print(f"Results demonstrate the framework's capability for research-grade")
        print(f"quantum advantage detection across multiple algorithmic dimensions.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print(f"This may be due to missing dependencies (PennyLane, scikit-learn, etc.)")
        
    return results


if __name__ == "__main__":
    # Run the comprehensive demo
    demo_results = main()
    
    print("\n📚 For more information, see:")
    print("  - Documentation: /docs/quantum-advantage-detection/")
    print("  - Research paper: Coming soon to arXiv")
    print("  - Source code: /src/quantum_mlops/advantage_detection/")