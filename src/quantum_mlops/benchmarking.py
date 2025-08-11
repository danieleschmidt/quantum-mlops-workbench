"""Quantum advantage benchmarking and performance comparison tools."""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

import numpy as np

from .core import QuantumMLPipeline, QuantumDevice, QuantumModel, QuantumMetrics
from .exceptions import QuantumMLOpsException

# Import advanced advantage detection modules
from .advantage_detection import (
    AdvantageAnalysisEngine,
    ComprehensiveAdvantageResult,
    AnalysisType,
    QuantumKernelAnalyzer,
    VariationalAdvantageAnalyzer,
    NoiseResilientTester,
    QuantumSupremacyAnalyzer
)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, log_loss
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results of a single benchmark run."""
    
    model_name: str
    accuracy: float
    training_time: float
    inference_time: float
    model_size: int  # Number of parameters
    memory_usage: float  # MB
    additional_metrics: Dict[str, Any]


@dataclass
class AdvantageAnalysis:
    """Analysis of quantum advantage in different scenarios."""
    
    problem_size: int
    quantum_accuracy: float
    classical_accuracy: float
    quantum_time: float
    classical_time: float
    advantage_ratio: float
    statistical_significance: float


class QuantumAdvantageTester:
    """Test and analyze quantum advantage over classical methods."""
    
    def __init__(self, device: QuantumDevice = QuantumDevice.SIMULATOR) -> None:
        """Initialize quantum advantage tester.
        
        Args:
            device: Quantum backend device to use
        """
        self.device = device
        self.benchmark_results: List[BenchmarkResult] = []
        self.advantage_analyses: List[AdvantageAnalysis] = []
    
    def compare(
        self,
        quantum_model: QuantumMLPipeline,
        classical_models: Dict[str, Any],
        dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        metrics: List[str] = ['accuracy', 'training_time', 'inference_time'],
        n_runs: int = 5
    ) -> Dict[str, Any]:
        """Compare quantum model against classical models.
        
        Args:
            quantum_model: Quantum ML pipeline
            classical_models: Dictionary of classical models
            dataset: (X_train, X_test, y_train, y_test) tuple
            metrics: Metrics to compare
            n_runs: Number of runs for statistical significance
            
        Returns:
            Comparison results dictionary
        """
        X_train, X_test, y_train, y_test = dataset
        
        # Benchmark quantum model
        quantum_results = self._benchmark_quantum_model(
            quantum_model, X_train, X_test, y_train, y_test, n_runs
        )
        
        # Benchmark classical models
        classical_results = {}
        if SKLEARN_AVAILABLE:
            for name, model in classical_models.items():
                classical_results[name] = self._benchmark_classical_model(
                    model, X_train, X_test, y_train, y_test, n_runs
                )
        else:
            logger.warning("Scikit-learn not available for classical benchmarking")
        
        # Analyze quantum advantage
        advantage_analysis = self._analyze_quantum_advantage(
            quantum_results, classical_results, X_train.shape[0]
        )
        
        return {
            'quantum_results': quantum_results,
            'classical_results': classical_results,
            'advantage_analysis': advantage_analysis,
            'dataset_info': {
                'n_features': X_train.shape[1],
                'n_train_samples': X_train.shape[0],
                'n_test_samples': X_test.shape[0],
                'n_classes': len(np.unique(y_train))
            }
        }
    
    def comprehensive_advantage_analysis(
        self,
        quantum_model: QuantumMLPipeline,
        classical_models: Dict[str, Any],
        dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        analysis_types: List[str] = ['kernel', 'variational', 'noise_resilient', 'supremacy'],
        **kwargs: Any
    ) -> ComprehensiveAdvantageResult:
        """Run comprehensive quantum advantage analysis with multiple detection algorithms."""
        
        X_train, X_test, y_train, y_test = dataset
        n_qubits = quantum_model.n_qubits
        
        # Initialize advantage analysis engine
        engine = AdvantageAnalysisEngine(
            n_qubits=n_qubits,
            enable_kernel_analysis='kernel' in analysis_types,
            enable_variational_analysis='variational' in analysis_types,
            enable_noise_analysis='noise_resilient' in analysis_types,
            enable_supremacy_analysis='supremacy' in analysis_types,
            seed=42  # Fixed seed for reproducibility
        )
        
        # Prepare analysis configuration
        analysis_config = {
            'X': X_train,
            'y': y_train,
            'problem_type': 'machine_learning',
            'analysis_types': analysis_types
        }
        
        # Add quantum circuit if available
        if hasattr(quantum_model, 'circuit'):
            analysis_config['quantum_circuit'] = quantum_model.circuit
        
        # Add cost function for variational analysis
        if 'variational' in analysis_types:
            def cost_function(params):
                # Simple cost function using quantum model
                try:
                    # Simplified cost evaluation
                    return np.sum(params**2) * 0.1  # Placeholder cost
                except:
                    return np.sum(params**2) * 0.1
            
            analysis_config['cost_function'] = cost_function
        
        # Add classical model for noise analysis
        if 'noise_resilient' in analysis_types and classical_models:
            analysis_config['classical_model'] = list(classical_models.values())[0]
        
        # Add problem sizes for supremacy analysis
        if 'supremacy' in analysis_types:
            analysis_config['problem_sizes'] = [4, 8, 12, min(16, n_qubits)]
        
        # Run comprehensive analysis
        results = engine.comprehensive_analysis(analysis_config, **kwargs)
        
        logger.info(f"Comprehensive advantage analysis complete: {results.advantage_category}")
        
        return results
    
    def _benchmark_quantum_model(
        self,
        model: QuantumMLPipeline,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        n_runs: int
    ) -> BenchmarkResult:
        """Benchmark quantum model performance."""
        accuracies = []
        training_times = []
        inference_times = []
        
        for run in range(n_runs):
            logger.info(f"Quantum benchmark run {run + 1}/{n_runs}")
            
            # Training
            start_time = time.time()
            trained_model = model.train(
                X_train, y_train,
                epochs=50,
                learning_rate=0.01,
                track_gradients=True
            )
            training_time = time.time() - start_time
            training_times.append(training_time)
            
            # Inference
            start_time = time.time()
            metrics = model.evaluate(trained_model, X_test, y_test)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            accuracies.append(metrics.accuracy)
        
        # Estimate model size and memory usage
        model_size = len(trained_model.parameters) if trained_model.parameters is not None else 0
        memory_usage = model_size * 8 / (1024 * 1024)  # Rough estimate in MB
        
        return BenchmarkResult(
            model_name="Quantum ML",
            accuracy=np.mean(accuracies),
            training_time=np.mean(training_times),
            inference_time=np.mean(inference_times),
            model_size=model_size,
            memory_usage=memory_usage,
            additional_metrics={
                'accuracy_std': np.std(accuracies),
                'training_time_std': np.std(training_times),
                'inference_time_std': np.std(inference_times),
                'gradient_variance': getattr(metrics, 'gradient_variance', 0.0),
                'fidelity': getattr(metrics, 'fidelity', 1.0)
            }
        )
    
    def _benchmark_classical_model(
        self,
        model: Any,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        n_runs: int
    ) -> BenchmarkResult:
        """Benchmark classical model performance."""
        if not SKLEARN_AVAILABLE:
            raise QuantumMLOpsException("Scikit-learn required for classical benchmarking")
        
        accuracies = []
        training_times = []
        inference_times = []
        
        for run in range(n_runs):
            # Create fresh model instance
            if hasattr(model, 'get_params'):
                fresh_model = model.__class__(**model.get_params())
            else:
                fresh_model = model
            
            # Training
            start_time = time.time()
            fresh_model.fit(X_train, y_train)
            training_time = time.time() - start_time
            training_times.append(training_time)
            
            # Inference
            start_time = time.time()
            y_pred = fresh_model.predict(X_test)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
        
        # Estimate model size
        model_size = self._estimate_model_size(fresh_model)
        memory_usage = model_size * 8 / (1024 * 1024)  # Rough estimate in MB
        
        return BenchmarkResult(
            model_name=fresh_model.__class__.__name__,
            accuracy=np.mean(accuracies),
            training_time=np.mean(training_times),
            inference_time=np.mean(inference_times),
            model_size=model_size,
            memory_usage=memory_usage,
            additional_metrics={
                'accuracy_std': np.std(accuracies),
                'training_time_std': np.std(training_times),
                'inference_time_std': np.std(inference_times)
            }
        )
    
    def _estimate_model_size(self, model: Any) -> int:
        """Estimate number of parameters in classical model."""
        if hasattr(model, 'coef_'):
            # Linear models
            coef = model.coef_
            size = coef.size
            if hasattr(model, 'intercept_'):
                size += model.intercept_.size
            return size
        
        elif hasattr(model, 'n_features_in_') and hasattr(model, 'n_estimators'):
            # Tree-based ensembles (rough estimate)
            return model.n_features_in_ * model.n_estimators * 10
        
        elif hasattr(model, 'hidden_layer_sizes'):
            # Neural networks
            sizes = [model.n_features_in_] + list(model.hidden_layer_sizes) + [model.n_outputs_]
            total_params = sum(sizes[i] * sizes[i+1] + sizes[i+1] for i in range(len(sizes)-1))
            return total_params
        
        else:
            # Default estimate
            return getattr(model, 'n_features_in_', 100) * 10
    
    def _analyze_quantum_advantage(
        self,
        quantum_results: BenchmarkResult,
        classical_results: Dict[str, BenchmarkResult],
        problem_size: int
    ) -> Dict[str, AdvantageAnalysis]:
        """Analyze quantum advantage across different classical models."""
        analyses = {}
        
        for classical_name, classical_result in classical_results.items():
            # Calculate advantage ratios
            accuracy_advantage = quantum_results.accuracy / classical_result.accuracy
            training_time_advantage = classical_result.training_time / quantum_results.training_time
            inference_time_advantage = classical_result.inference_time / quantum_results.inference_time
            
            # Overall advantage score (weighted combination)
            advantage_ratio = (
                0.5 * accuracy_advantage +
                0.25 * training_time_advantage +
                0.25 * inference_time_advantage
            )
            
            # Statistical significance (simplified t-test approximation)
            quantum_std = quantum_results.additional_metrics.get('accuracy_std', 0.01)
            classical_std = classical_result.additional_metrics.get('accuracy_std', 0.01)
            
            pooled_std = np.sqrt((quantum_std**2 + classical_std**2) / 2)
            t_stat = abs(quantum_results.accuracy - classical_result.accuracy) / pooled_std
            
            # Rough p-value approximation (assuming normal distribution)
            statistical_significance = max(0, 1 - 2 * (1 - self._normal_cdf(t_stat)))
            
            analyses[classical_name] = AdvantageAnalysis(
                problem_size=problem_size,
                quantum_accuracy=quantum_results.accuracy,
                classical_accuracy=classical_result.accuracy,
                quantum_time=quantum_results.training_time + quantum_results.inference_time,
                classical_time=classical_result.training_time + classical_result.inference_time,
                advantage_ratio=advantage_ratio,
                statistical_significance=statistical_significance
            )
        
        return analyses
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF for statistical significance."""
        # Simple approximation using error function
        return 0.5 * (1 + np.tanh(x * np.sqrt(2 / np.pi)))
    
    def generate_scaling_analysis(
        self,
        quantum_circuit_fn: callable,
        classical_model_fn: callable,
        problem_sizes: List[int],
        n_runs: int = 3
    ) -> Dict[str, List[float]]:
        """Analyze how quantum vs classical performance scales with problem size.
        
        Args:
            quantum_circuit_fn: Function that creates quantum circuit for given size
            classical_model_fn: Function that creates classical model for given size
            problem_sizes: List of problem sizes to test
            n_runs: Number of runs per size
            
        Returns:
            Dictionary with scaling analysis results
        """
        quantum_times = []
        classical_times = []
        quantum_accuracies = []
        classical_accuracies = []
        
        for size in problem_sizes:
            logger.info(f"Testing problem size: {size}")
            
            # Generate synthetic dataset
            X_train = np.random.rand(100, size)
            y_train = np.random.randint(0, 2, 100)
            X_test = np.random.rand(20, size)
            y_test = np.random.randint(0, 2, 20)
            
            # Test quantum model
            quantum_circuit = quantum_circuit_fn(size)
            quantum_pipeline = QuantumMLPipeline(
                circuit=quantum_circuit,
                n_qubits=min(size, 10),  # Limit qubits for simulation
                device=self.device
            )
            
            quantum_result = self._benchmark_quantum_model(
                quantum_pipeline, X_train, X_test, y_train, y_test, n_runs
            )
            
            quantum_times.append(quantum_result.training_time + quantum_result.inference_time)
            quantum_accuracies.append(quantum_result.accuracy)
            
            # Test classical model
            if SKLEARN_AVAILABLE:
                classical_model = classical_model_fn(size)
                classical_result = self._benchmark_classical_model(
                    classical_model, X_train, X_test, y_train, y_test, n_runs
                )
                
                classical_times.append(classical_result.training_time + classical_result.inference_time)
                classical_accuracies.append(classical_result.accuracy)
            else:
                classical_times.append(0.1)  # Dummy value
                classical_accuracies.append(0.5)  # Dummy value
        
        return {
            'problem_sizes': problem_sizes,
            'quantum_times': quantum_times,
            'classical_times': classical_times,
            'quantum_accuracies': quantum_accuracies,
            'classical_accuracies': classical_accuracies,
            'quantum_advantage_times': [c/q if q > 0 else 1 for q, c in zip(quantum_times, classical_times)],
            'quantum_advantage_accuracy': [q/c if c > 0 else 1 for q, c in zip(quantum_accuracies, classical_accuracies)]
        }
    
    def plot_advantage_regions(self, comparison_results: Dict[str, Any]) -> None:
        """Plot quantum advantage regions across different problem scenarios.
        
        Args:
            comparison_results: Results from compare() method
        """
        try:
            import matplotlib.pyplot as plt
            
            # Extract data
            quantum_result = comparison_results['quantum_results']
            classical_results = comparison_results['classical_results']
            advantage_analysis = comparison_results['advantage_analysis']
            
            # Create comparison plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Accuracy comparison
            models = ['Quantum'] + list(classical_results.keys())
            accuracies = [quantum_result.accuracy] + [r.accuracy for r in classical_results.values()]
            accuracy_stds = [quantum_result.additional_metrics.get('accuracy_std', 0)] + \
                           [r.additional_metrics.get('accuracy_std', 0) for r in classical_results.values()]
            
            bars1 = ax1.bar(models, accuracies, yerr=accuracy_stds, capsize=5)
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Model Accuracy Comparison')
            ax1.grid(True, alpha=0.3)
            
            # Color quantum bar differently
            bars1[0].set_color('red')
            for i in range(1, len(bars1)):
                bars1[i].set_color('blue')
            
            # Training time comparison
            training_times = [quantum_result.training_time] + [r.training_time for r in classical_results.values()]
            training_stds = [quantum_result.additional_metrics.get('training_time_std', 0)] + \
                           [r.additional_metrics.get('training_time_std', 0) for r in classical_results.values()]
            
            bars2 = ax2.bar(models, training_times, yerr=training_stds, capsize=5)
            ax2.set_ylabel('Training Time (s)')
            ax2.set_title('Training Time Comparison')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
            
            bars2[0].set_color('red')
            for i in range(1, len(bars2)):
                bars2[i].set_color('blue')
            
            # Model size comparison
            model_sizes = [quantum_result.model_size] + [r.model_size for r in classical_results.values()]
            
            bars3 = ax3.bar(models, model_sizes)
            ax3.set_ylabel('Number of Parameters')
            ax3.set_title('Model Size Comparison')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
            
            bars3[0].set_color('red')
            for i in range(1, len(bars3)):
                bars3[i].set_color('blue')
            
            # Advantage ratio heatmap
            classical_names = list(advantage_analysis.keys())
            advantage_ratios = [analysis.advantage_ratio for analysis in advantage_analysis.values()]
            significance = [analysis.statistical_significance for analysis in advantage_analysis.values()]
            
            # Create scatter plot with color coding for significance
            scatter = ax4.scatter(classical_names, advantage_ratios, 
                                c=significance, cmap='RdYlGn', s=100, alpha=0.7)
            ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No advantage')
            ax4.set_ylabel('Quantum Advantage Ratio')
            ax4.set_title('Quantum Advantage vs Classical Models')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            # Add colorbar for significance
            cbar = plt.colorbar(scatter, ax=ax4)
            cbar.set_label('Statistical Significance')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
    
    def generate_benchmark_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed benchmark report.
        
        Args:
            results: Results from compare() method
            
        Returns:
            Formatted benchmark report string
        """
        report = "QUANTUM ADVANTAGE BENCHMARK REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Dataset info
        dataset_info = results['dataset_info']
        report += f"Dataset Information:\n"
        report += f"  Features: {dataset_info['n_features']}\n"
        report += f"  Training samples: {dataset_info['n_train_samples']}\n"
        report += f"  Test samples: {dataset_info['n_test_samples']}\n"
        report += f"  Classes: {dataset_info['n_classes']}\n\n"
        
        # Quantum model results
        quantum_result = results['quantum_results']
        report += f"Quantum Model Performance:\n"
        report += f"  Accuracy: {quantum_result.accuracy:.4f} ± {quantum_result.additional_metrics.get('accuracy_std', 0):.4f}\n"
        report += f"  Training time: {quantum_result.training_time:.4f}s ± {quantum_result.additional_metrics.get('training_time_std', 0):.4f}s\n"
        report += f"  Inference time: {quantum_result.inference_time:.4f}s ± {quantum_result.additional_metrics.get('inference_time_std', 0):.4f}s\n"
        report += f"  Model size: {quantum_result.model_size} parameters\n"
        report += f"  Memory usage: {quantum_result.memory_usage:.2f} MB\n"
        report += f"  Gradient variance: {quantum_result.additional_metrics.get('gradient_variance', 0):.6f}\n"
        report += f"  Fidelity: {quantum_result.additional_metrics.get('fidelity', 1):.4f}\n\n"
        
        # Classical model results
        classical_results = results['classical_results']
        for name, result in classical_results.items():
            report += f"{name} Performance:\n"
            report += f"  Accuracy: {result.accuracy:.4f} ± {result.additional_metrics.get('accuracy_std', 0):.4f}\n"
            report += f"  Training time: {result.training_time:.4f}s ± {result.additional_metrics.get('training_time_std', 0):.4f}s\n"
            report += f"  Inference time: {result.inference_time:.4f}s ± {result.additional_metrics.get('inference_time_std', 0):.4f}s\n"
            report += f"  Model size: {result.model_size} parameters\n"
            report += f"  Memory usage: {result.memory_usage:.2f} MB\n\n"
        
        # Advantage analysis
        advantage_analysis = results['advantage_analysis']
        report += "Quantum Advantage Analysis:\n"
        report += "-" * 30 + "\n"
        
        for name, analysis in advantage_analysis.items():
            advantage_label = "ADVANTAGE" if analysis.advantage_ratio > 1.1 else "NO CLEAR ADVANTAGE" if analysis.advantage_ratio > 0.9 else "DISADVANTAGE"
            
            report += f"\nvs {name}:\n"
            report += f"  Overall advantage ratio: {analysis.advantage_ratio:.3f} ({advantage_label})\n"
            report += f"  Accuracy ratio: {analysis.quantum_accuracy / analysis.classical_accuracy:.3f}\n"
            report += f"  Time ratio: {analysis.classical_time / analysis.quantum_time:.3f}\n"
            report += f"  Statistical significance: {analysis.statistical_significance:.3f}\n"
        
        # Summary
        report += "\n" + "=" * 50 + "\n"
        report += "SUMMARY:\n"
        
        best_advantage_model = max(advantage_analysis.keys(), 
                                 key=lambda k: advantage_analysis[k].advantage_ratio)
        best_advantage_ratio = advantage_analysis[best_advantage_model].advantage_ratio
        
        if best_advantage_ratio > 1.1:
            report += f"✓ Quantum advantage observed against {best_advantage_model} (ratio: {best_advantage_ratio:.3f})\n"
        elif best_advantage_ratio > 0.9:
            report += f"≈ Competitive performance with classical models\n"
        else:
            report += f"✗ No quantum advantage observed in this scenario\n"
        
        return report