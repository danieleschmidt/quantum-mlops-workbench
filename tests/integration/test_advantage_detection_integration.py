"""Integration tests for quantum advantage detection framework.

This module provides comprehensive integration tests for the advanced quantum
advantage detection algorithms, ensuring robustness across different scenarios
and edge cases.
"""

import pytest
import numpy as np
from typing import Dict, Any, List, Tuple
from unittest.mock import MagicMock, patch

from quantum_mlops import (
    AdvantageAnalysisEngine,
    QuantumKernelAnalyzer,
    VariationalAdvantageAnalyzer,
    NoiseResilientTester,
    QuantumSupremacyAnalyzer,
    ComprehensiveAdvantageResult
)

from quantum_mlops.advantage_detection.quantum_kernel_advantage import QuantumFeatureMap
from quantum_mlops.advantage_detection.variational_advantage_protocols import VariationalAlgorithm
from quantum_mlops.advantage_detection.noise_resilient_testing import NoiseModel, ErrorMitigation
from quantum_mlops.advantage_detection.multi_metric_supremacy import SupremacyMetric

# Skip tests if PennyLane is not available
pennylane = pytest.importorskip("pennylane")
sklearn = pytest.importorskip("sklearn")


class TestQuantumKernelAdvantageIntegration:
    """Integration tests for quantum kernel advantage detection."""
    
    @pytest.fixture
    def sample_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create sample dataset for testing."""
        from sklearn.datasets import make_classification
        
        X, y = make_classification(
            n_samples=50, n_features=4, n_classes=2, 
            n_redundant=0, random_state=42
        )
        return X, y
    
    @pytest.fixture
    def kernel_analyzer(self) -> QuantumKernelAnalyzer:
        """Create quantum kernel analyzer for testing."""
        return QuantumKernelAnalyzer(
            n_qubits=4,
            feature_map=QuantumFeatureMap.IQP,
            shots=100,  # Reduced for faster testing
            seed=42
        )
    
    def test_kernel_advantage_basic_functionality(
        self, 
        kernel_analyzer: QuantumKernelAnalyzer,
        sample_dataset: Tuple[np.ndarray, np.ndarray]
    ):
        """Test basic quantum kernel advantage analysis."""
        X, y = sample_dataset
        
        # Run analysis
        result = kernel_analyzer.comprehensive_advantage_analysis(X, y)
        
        # Validate result structure
        assert hasattr(result, 'overall_advantage_score')
        assert hasattr(result, 'advantage_category')
        assert hasattr(result, 'spectral_advantage')
        assert hasattr(result, 'performance_advantage')
        
        # Validate result values
        assert isinstance(result.overall_advantage_score, float)
        assert result.advantage_category in ['strong', 'moderate', 'weak', 'none']
        assert isinstance(result.statistically_significant, bool)
        
    def test_different_feature_maps(self, sample_dataset: Tuple[np.ndarray, np.ndarray]):
        """Test kernel advantage with different feature maps."""
        X, y = sample_dataset
        
        feature_maps = [
            QuantumFeatureMap.IQP,
            QuantumFeatureMap.ZFEATURE,
            QuantumFeatureMap.QAOA_INSPIRED,
            QuantumFeatureMap.HARDWARE_EFFICIENT
        ]
        
        for feature_map in feature_maps:
            analyzer = QuantumKernelAnalyzer(
                n_qubits=4, 
                feature_map=feature_map,
                shots=100,
                seed=42
            )
            
            result = analyzer.comprehensive_advantage_analysis(X, y)
            
            # Should complete without errors
            assert result is not None
            assert hasattr(result, 'overall_advantage_score')
    
    def test_kernel_matrix_computation(self, kernel_analyzer: QuantumKernelAnalyzer):
        """Test quantum kernel matrix computation."""
        X = np.random.rand(10, 4)
        
        # Compute quantum kernel matrix
        quantum_kernel = kernel_analyzer.compute_quantum_kernel_matrix(X)
        
        # Validate matrix properties
        assert quantum_kernel.shape == (10, 10)
        assert np.allclose(quantum_kernel, quantum_kernel.T)  # Should be symmetric
        assert np.all(quantum_kernel >= 0)  # Kernel values should be non-negative
        assert np.all(np.diag(quantum_kernel) <= 1.1)  # Diagonal elements should be ~1


class TestVariationalAdvantageIntegration:
    """Integration tests for variational quantum advantage analysis."""
    
    @pytest.fixture
    def cost_function(self) -> callable:
        """Create sample cost function for testing."""
        def cost_fn(params: np.ndarray) -> float:
            return np.sum(params**2) + 0.1 * np.sum(np.sin(params))
        return cost_fn
    
    @pytest.fixture
    def variational_analyzer(self) -> VariationalAdvantageAnalyzer:
        """Create variational advantage analyzer for testing."""
        return VariationalAdvantageAnalyzer(
            n_qubits=4,
            algorithm=VariationalAlgorithm.VQE,
            n_layers=2,
            shots=100,
            seed=42
        )
    
    def test_variational_advantage_basic_functionality(
        self, 
        variational_analyzer: VariationalAdvantageAnalyzer,
        cost_function: callable
    ):
        """Test basic variational advantage analysis."""
        result = variational_analyzer.comprehensive_advantage_analysis(cost_function)
        
        # Validate result structure
        assert hasattr(result, 'overall_advantage_score')
        assert hasattr(result, 'advantage_category')
        assert hasattr(result, 'landscape_advantage')
        assert hasattr(result, 'expressivity_advantage')
        assert hasattr(result, 'plateau_detected')
        
        # Validate result values
        assert isinstance(result.overall_advantage_score, float)
        assert result.advantage_category in ['strong', 'moderate', 'weak', 'none']
        assert isinstance(result.plateau_detected, bool)
    
    def test_different_variational_algorithms(self, cost_function: callable):
        """Test variational advantage with different algorithms."""
        algorithms = [
            VariationalAlgorithm.VQE,
            VariationalAlgorithm.QAOA,
            VariationalAlgorithm.VQC
        ]
        
        for algorithm in algorithms:
            analyzer = VariationalAdvantageAnalyzer(
                n_qubits=4,
                algorithm=algorithm,
                n_layers=2,
                shots=100,
                seed=42
            )
            
            result = analyzer.comprehensive_advantage_analysis(cost_function)
            
            # Should complete without errors
            assert result is not None
            assert hasattr(result, 'overall_advantage_score')
    
    def test_barren_plateau_detection(self, variational_analyzer: VariationalAdvantageAnalyzer):
        """Test barren plateau detection."""
        # Create a cost function that should exhibit barren plateaus
        def plateau_cost_function(params: np.ndarray) -> float:
            # Very flat landscape with small gradients
            return 0.01 * np.sum(params**2) + 1e-6 * np.sum(np.sin(params))
        
        result = variational_analyzer.barren_plateau_analysis(
            plateau_cost_function, n_gradient_samples=20
        )
        
        assert 'gradient_variance' in result
        assert 'plateau_detected' in result
        assert 'effective_dimension' in result
        assert isinstance(result['plateau_detected'], bool)


class TestNoiseResilientIntegration:
    """Integration tests for noise-resilient advantage testing."""
    
    @pytest.fixture
    def noise_tester(self) -> NoiseResilientTester:
        """Create noise-resilient tester for testing."""
        return NoiseResilientTester(
            n_qubits=4,
            circuit_depth=5,
            shots=100,
            seed=42
        )
    
    @pytest.fixture
    def sample_circuit(self) -> callable:
        """Create sample quantum circuit for testing."""
        def circuit():
            # Mock quantum circuit output
            return np.random.rand(4)  # 4 expectation values
        return circuit
    
    @pytest.fixture
    def classical_model(self):
        """Create classical model for comparison."""
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_dummy = np.random.rand(50, 4)
        y_dummy = np.random.randint(0, 2, 50)
        model.fit(X_dummy, y_dummy)
        return model
    
    def test_noise_resilient_basic_functionality(
        self,
        noise_tester: NoiseResilientTester,
        sample_circuit: callable,
        classical_model
    ):
        """Test basic noise-resilient advantage analysis."""
        result = noise_tester.comprehensive_noise_advantage_analysis(
            sample_circuit,
            classical_model,
            noise_levels=[0.001, 0.01, 0.05]
        )
        
        # Validate result structure
        assert hasattr(result, 'noise_resilient_advantage_score')
        assert hasattr(result, 'advantage_category')
        assert hasattr(result, 'noise_resilience_score')
        assert hasattr(result, 'advantage_lost_threshold')
        
        # Validate result values
        assert isinstance(result.noise_resilient_advantage_score, float)
        assert result.advantage_category in [
            'strong_noise_resilient', 'moderate_noise_resilient', 
            'weak_noise_resilient', 'noise_limited'
        ]
    
    def test_different_noise_models(
        self,
        noise_tester: NoiseResilientTester,
        sample_circuit: callable,
        classical_model
    ):
        """Test noise-resilient analysis with different noise models."""
        noise_models = [
            NoiseModel.DEPOLARIZING,
            NoiseModel.AMPLITUDE_DAMPING,
            NoiseModel.PHASE_DAMPING
        ]
        
        for noise_model in noise_models:
            result = noise_tester.comprehensive_noise_advantage_analysis(
                sample_circuit,
                classical_model,
                noise_models=[noise_model],
                noise_levels=[0.01, 0.05]
            )
            
            # Should complete without errors
            assert result is not None
            assert hasattr(result, 'noise_resilient_advantage_score')
    
    def test_error_mitigation_techniques(
        self,
        noise_tester: NoiseResilientTester,
        sample_circuit: callable
    ):
        """Test error mitigation techniques."""
        # Test zero-noise extrapolation
        zne_result, zne_error = noise_tester.zero_noise_extrapolation(
            sample_circuit, [0.001, 0.005, 0.01]
        )
        
        assert isinstance(zne_result, float)
        assert isinstance(zne_error, float)
        
        # Test symmetry verification
        sv_result, sv_variance = noise_tester.symmetry_verification(sample_circuit)
        
        assert isinstance(sv_result, float)
        assert isinstance(sv_variance, float)


class TestQuantumSupremacyIntegration:
    """Integration tests for quantum supremacy analysis."""
    
    @pytest.fixture
    def supremacy_analyzer(self) -> QuantumSupremacyAnalyzer:
        """Create quantum supremacy analyzer for testing."""
        return QuantumSupremacyAnalyzer(
            max_qubits=8,
            max_circuit_depth=10,
            shots=100,
            seed=42
        )
    
    def test_supremacy_basic_functionality(
        self,
        supremacy_analyzer: QuantumSupremacyAnalyzer
    ):
        """Test basic quantum supremacy analysis."""
        problem_sizes = [4, 6, 8]
        
        result = supremacy_analyzer.comprehensive_supremacy_analysis(
            problem_sizes=problem_sizes
        )
        
        # Validate result structure
        assert hasattr(result, 'supremacy_confidence')
        assert hasattr(result, 'supremacy_category')
        assert hasattr(result, 'scaling_advantage')
        assert hasattr(result, 'crossover_point')
        
        # Validate result values
        assert isinstance(result.supremacy_confidence, float)
        assert result.supremacy_category in ['strong', 'moderate', 'conditional', 'none']
        assert isinstance(result.supremacy_achieved, bool)
    
    def test_scaling_analysis(self, supremacy_analyzer: QuantumSupremacyAnalyzer):
        """Test scaling analysis."""
        problem_sizes = [4, 6, 8]
        
        scaling_result = supremacy_analyzer.scaling_analysis(
            problem_sizes,
            supremacy_analyzer.create_random_circuit
        )
        
        assert 'quantum_scaling_exponent' in scaling_result
        assert 'classical_scaling_exponent' in scaling_result
        assert 'scaling_advantage' in scaling_result
        assert 'crossover_point' in scaling_result
    
    def test_sample_complexity_analysis(
        self,
        supremacy_analyzer: QuantumSupremacyAnalyzer
    ):
        """Test sample complexity analysis."""
        result = supremacy_analyzer.sample_complexity_analysis(
            target_accuracies=[0.7, 0.8, 0.9]
        )
        
        assert 'quantum_sample_complexities' in result
        assert 'classical_sample_complexities' in result
        assert 'sample_efficiency_advantage' in result
        
        # Validate list lengths
        assert len(result['quantum_sample_complexities']) == 3
        assert len(result['classical_sample_complexities']) == 3


class TestAdvantageAnalysisEngineIntegration:
    """Integration tests for the comprehensive advantage analysis engine."""
    
    @pytest.fixture
    def sample_analysis_data(self) -> Dict[str, Any]:
        """Create sample data for comprehensive analysis."""
        from sklearn.datasets import make_classification
        
        X, y = make_classification(
            n_samples=50, n_features=4, n_classes=2,
            random_state=42
        )
        
        def cost_function(params):
            return np.sum(params**2) + 0.1 * np.sum(np.sin(params))
        
        def quantum_circuit():
            return np.random.rand(4)
        
        from sklearn.ensemble import RandomForestClassifier
        classical_model = RandomForestClassifier(n_estimators=10, random_state=42)
        classical_model.fit(X, y)
        
        return {
            'X': X,
            'y': y,
            'cost_function': cost_function,
            'quantum_circuit': quantum_circuit,
            'classical_model': classical_model,
            'problem_type': 'classification'
        }
    
    @pytest.fixture
    def analysis_engine(self) -> AdvantageAnalysisEngine:
        """Create advantage analysis engine for testing."""
        return AdvantageAnalysisEngine(
            n_qubits=4,
            enable_kernel_analysis=True,
            enable_variational_analysis=True,
            enable_noise_analysis=True,
            enable_supremacy_analysis=True,
            shots=100,
            seed=42
        )
    
    def test_comprehensive_analysis(
        self,
        analysis_engine: AdvantageAnalysisEngine,
        sample_analysis_data: Dict[str, Any]
    ):
        """Test comprehensive quantum advantage analysis."""
        # Configure analysis
        analysis_config = sample_analysis_data.copy()
        analysis_config['analysis_types'] = ['kernel', 'variational', 'noise_resilient']
        
        # Run comprehensive analysis
        result = analysis_engine.comprehensive_analysis(analysis_config)
        
        # Validate result structure
        assert isinstance(result, ComprehensiveAdvantageResult)
        assert hasattr(result, 'overall_advantage_score')
        assert hasattr(result, 'advantage_confidence')
        assert hasattr(result, 'advantage_category')
        assert hasattr(result, 'key_advantages')
        assert hasattr(result, 'limitations')
        assert hasattr(result, 'recommendations')
        
        # Validate result values
        assert isinstance(result.overall_advantage_score, float)
        assert result.advantage_confidence.value in ['high', 'medium', 'low', 'none']
        assert isinstance(result.key_advantages, list)
        assert isinstance(result.limitations, list)
        assert isinstance(result.recommendations, list)
    
    def test_partial_analysis_types(
        self,
        analysis_engine: AdvantageAnalysisEngine,
        sample_analysis_data: Dict[str, Any]
    ):
        """Test analysis with subset of analysis types."""
        # Test with only kernel analysis
        config_kernel = sample_analysis_data.copy()
        config_kernel['analysis_types'] = ['kernel']
        
        result_kernel = analysis_engine.comprehensive_analysis(config_kernel)
        assert result_kernel.kernel_advantage is not None
        assert result_kernel.variational_advantage is None
        
        # Test with only variational analysis
        config_var = sample_analysis_data.copy()
        config_var['analysis_types'] = ['variational']
        
        result_var = analysis_engine.comprehensive_analysis(config_var)
        assert result_var.variational_advantage is not None
        assert result_var.kernel_advantage is None
    
    def test_problem_characteristics_analysis(
        self,
        analysis_engine: AdvantageAnalysisEngine,
        sample_analysis_data: Dict[str, Any]
    ):
        """Test problem characteristics analysis."""
        X = sample_analysis_data['X']
        y = sample_analysis_data['y']
        
        characteristics = analysis_engine.analyze_problem_characteristics(
            X, y, 'classification'
        )
        
        # Validate characteristics
        assert 'problem_type' in characteristics
        assert 'n_samples' in characteristics
        assert 'n_features' in characteristics
        assert 'task_type' in characteristics
        assert 'n_classes' in characteristics
        
        assert characteristics['problem_type'] == 'classification'
        assert characteristics['n_samples'] == X.shape[0]
        assert characteristics['n_features'] == X.shape[1]
    
    def test_report_generation(
        self,
        analysis_engine: AdvantageAnalysisEngine,
        sample_analysis_data: Dict[str, Any]
    ):
        """Test report generation functionality."""
        # Configure minimal analysis
        analysis_config = sample_analysis_data.copy()
        analysis_config['analysis_types'] = ['kernel']
        
        # Run analysis
        result = analysis_engine.comprehensive_analysis(analysis_config)
        
        # Generate report
        report = analysis_engine.generate_report(result)
        
        # Validate report content
        assert isinstance(report, str)
        assert 'Quantum Advantage Analysis Report' in report
        assert 'Overall Assessment' in report
        assert 'Key Advantages' in report
        assert 'Recommendations' in report
    
    @pytest.mark.slow
    def test_full_comprehensive_analysis(
        self,
        analysis_engine: AdvantageAnalysisEngine,
        sample_analysis_data: Dict[str, Any]
    ):
        """Test full comprehensive analysis with all analysis types."""
        # Configure full analysis
        analysis_config = sample_analysis_data.copy()
        analysis_config['analysis_types'] = [
            'kernel', 'variational', 'noise_resilient', 'supremacy'
        ]
        analysis_config['problem_sizes'] = [4, 6]  # Smaller for faster testing
        
        # Run comprehensive analysis
        result = analysis_engine.comprehensive_analysis(analysis_config)
        
        # Validate all analysis results are present
        assert result.kernel_advantage is not None
        assert result.variational_advantage is not None
        assert result.noise_resilient_advantage is not None
        assert result.quantum_supremacy is not None
        
        # Validate overall metrics
        assert 0 <= result.overall_advantage_score <= 1
        assert len(result.key_advantages) > 0
        assert len(result.recommendations) > 0


class TestIntegrationErrorHandling:
    """Test error handling and edge cases in integration scenarios."""
    
    def test_invalid_dataset_handling(self):
        """Test handling of invalid datasets."""
        analyzer = QuantumKernelAnalyzer(n_qubits=4, seed=42)
        
        # Test with mismatched X and y dimensions
        X = np.random.rand(10, 4)
        y = np.random.randint(0, 2, 5)  # Wrong size
        
        with pytest.raises(ValueError):
            analyzer.comprehensive_advantage_analysis(X, y)
    
    def test_insufficient_qubits_handling(self):
        """Test handling when insufficient qubits are available."""
        # Try to create analyzer with 0 qubits
        with pytest.raises((ValueError, Exception)):
            QuantumKernelAnalyzer(n_qubits=0)
    
    def test_missing_dependencies_handling(self):
        """Test graceful handling of missing dependencies."""
        with patch('quantum_mlops.advantage_detection.quantum_kernel_advantage.PENNYLANE_AVAILABLE', False):
            with pytest.raises(Exception):
                QuantumKernelAnalyzer(n_qubits=4)
    
    def test_analysis_engine_partial_failure(self):
        """Test analysis engine behavior when some analyses fail."""
        engine = AdvantageAnalysisEngine(
            n_qubits=4,
            enable_kernel_analysis=True,
            enable_variational_analysis=False,  # Disable to test partial failure
            seed=42
        )
        
        # Create minimal valid config
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=20, n_features=4, random_state=42)
        
        config = {
            'X': X,
            'y': y,
            'problem_type': 'classification',
            'analysis_types': ['kernel']
        }
        
        # Should complete successfully with partial analysis
        result = engine.comprehensive_analysis(config)
        assert result.kernel_advantage is not None
        assert result.variational_advantage is None


# Test fixtures and utilities
@pytest.fixture
def mock_pennylane():
    """Mock PennyLane for testing without full quantum simulation."""
    with patch('pennylane.device') as mock_device:
        mock_device.return_value = MagicMock()
        yield mock_device


@pytest.mark.integration
class TestAdvantageDetectionWorkflows:
    """Test complete workflows for quantum advantage detection."""
    
    def test_ml_classification_workflow(self):
        """Test complete ML classification advantage detection workflow."""
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        
        # Create dataset
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Classical models
        classical_models = {
            'random_forest': RandomForestClassifier(n_estimators=10, random_state=42)
        }
        
        # Initialize analysis engine
        engine = AdvantageAnalysisEngine(
            n_qubits=4,
            enable_kernel_analysis=True,
            enable_variational_analysis=False,  # Skip for faster testing
            enable_noise_analysis=False,
            enable_supremacy_analysis=False,
            seed=42
        )
        
        # Run analysis
        config = {
            'X': X_train,
            'y': y_train,
            'problem_type': 'classification',
            'analysis_types': ['kernel']
        }
        
        result = engine.comprehensive_analysis(config)
        
        # Validate workflow completion
        assert result is not None
        assert result.overall_advantage_score >= 0
        assert len(result.recommendations) > 0
    
    def test_optimization_workflow(self):
        """Test complete optimization problem workflow."""
        # Create optimization problem
        def optimization_cost(params):
            return np.sum(params**2) + 0.1 * np.sum(np.sin(params))
        
        # Initialize variational analyzer
        analyzer = VariationalAdvantageAnalyzer(
            n_qubits=4,
            algorithm=VariationalAlgorithm.VQE,
            n_layers=2,
            seed=42
        )
        
        # Run analysis
        result = analyzer.comprehensive_advantage_analysis(optimization_cost)
        
        # Validate workflow completion
        assert result is not None
        assert result.overall_advantage_score >= 0
        assert result.advantage_category in ['strong', 'moderate', 'weak', 'none']