"""Multi-Metric Quantum Supremacy Testing.

This module implements comprehensive quantum supremacy analysis across multiple dimensions
including scaling analysis, sample complexity, solution quality, and resource efficiency.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import linregress, kstest
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

from ..logging_config import get_logger
from ..exceptions import QuantumMLOpsException

logger = get_logger(__name__)


class SupremacyMetric(Enum):
    """Quantum supremacy metrics to analyze."""
    
    EXECUTION_TIME = "execution_time"
    SOLUTION_QUALITY = "solution_quality" 
    SCALING_EXPONENT = "scaling_exponent"
    SAMPLE_COMPLEXITY = "sample_complexity"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    APPROXIMATION_RATIO = "approximation_ratio"
    CIRCUIT_EXPRESSIVITY = "circuit_expressivity"


class ScalingRegime(Enum):
    """Different scaling regimes for analysis."""
    
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"
    SUPER_EXPONENTIAL = "super_exponential"
    SUBLINEAR = "sublinear"


@dataclass
class SupremacyResult:
    """Comprehensive quantum supremacy analysis results."""
    
    # Scaling analysis
    problem_sizes: List[int]
    quantum_scaling_exponent: float
    classical_scaling_exponent: float
    scaling_advantage: float
    scaling_regime: ScalingRegime
    crossover_point: int
    
    # Performance metrics
    quantum_execution_times: List[float]
    classical_execution_times: List[float]
    quantum_solution_qualities: List[float]
    classical_solution_qualities: List[float]
    
    # Sample complexity analysis
    quantum_sample_complexities: List[int]
    classical_sample_complexities: List[int]
    sample_efficiency_advantage: float
    
    # Resource efficiency
    quantum_resource_usage: List[float]
    classical_resource_usage: List[float]
    resource_efficiency_ratio: float
    
    # Approximation quality
    approximation_ratios: List[float]
    theoretical_limits: List[float]
    approximation_advantage: float
    
    # Statistical validation
    supremacy_p_value: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    statistical_power: float
    
    # Supremacy classification
    supremacy_achieved: bool
    supremacy_confidence: float
    supremacy_category: str  # "strong", "moderate", "conditional", "none"


class QuantumSupremacyAnalyzer:
    """Advanced quantum supremacy testing framework."""
    
    def __init__(
        self,
        max_qubits: int = 20,
        max_circuit_depth: int = 20,
        shots: int = 1000,
        seed: Optional[int] = None
    ) -> None:
        """Initialize quantum supremacy analyzer.
        
        Args:
            max_qubits: Maximum number of qubits to test
            max_circuit_depth: Maximum circuit depth to test
            shots: Number of measurement shots
            seed: Random seed for reproducibility
        """
        self.max_qubits = max_qubits
        self.max_circuit_depth = max_circuit_depth
        self.shots = shots
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
            
        if not PENNYLANE_AVAILABLE:
            raise QuantumMLOpsException(
                "PennyLane is required for supremacy analysis"
            )
        
        logger.info(
            f"Initialized QuantumSupremacyAnalyzer with max {max_qubits} qubits, "
            f"max depth {max_circuit_depth}"
        )
    
    def create_random_circuit(
        self,
        n_qubits: int,
        depth: int,
        gate_set: List[str] = None
    ) -> Callable:
        """Create random quantum circuit for supremacy testing."""
        
        if gate_set is None:
            gate_set = ["RX", "RY", "RZ", "CNOT", "CZ", "H"]
        
        dev = qml.device("default.qubit", wires=n_qubits, shots=self.shots)
        
        @qml.qnode(dev)
        def random_circuit():
            """Random quantum circuit."""
            
            for d in range(depth):
                # Random single-qubit gates
                for i in range(n_qubits):
                    gate = np.random.choice(["RX", "RY", "RZ", "H"])
                    param = np.random.uniform(-np.pi, np.pi)
                    
                    if gate == "RX":
                        qml.RX(param, wires=i)
                    elif gate == "RY":
                        qml.RY(param, wires=i)
                    elif gate == "RZ":
                        qml.RZ(param, wires=i)
                    elif gate == "H":
                        qml.Hadamard(wires=i)
                
                # Random two-qubit gates
                if n_qubits > 1:
                    for i in range(0, n_qubits - 1, 2):
                        if i + 1 < n_qubits:
                            gate = np.random.choice(["CNOT", "CZ"])
                            if gate == "CNOT":
                                qml.CNOT(wires=[i, i + 1])
                            elif gate == "CZ":
                                qml.CZ(wires=[i, i + 1])
            
            # Return probability distribution
            return qml.probs(wires=range(min(n_qubits, 10)))  # Limit output size
        
        return random_circuit
    
    def simulate_classical_algorithm(
        self,
        problem_size: int,
        algorithm_type: str = "random_forest"
    ) -> Tuple[float, float]:
        """Simulate classical algorithm for comparison."""
        
        # Generate synthetic problem data
        np.random.seed(self.seed)
        X = np.random.rand(problem_size, min(problem_size, 20))
        y = np.random.randint(0, 2, problem_size)
        
        start_time = time.time()
        
        if algorithm_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=min(100, problem_size),
                random_state=self.seed
            )
        elif algorithm_type == "neural_network":
            model = MLPRegressor(
                hidden_layer_sizes=(min(100, problem_size),),
                max_iter=min(1000, problem_size * 10),
                random_state=self.seed
            )
        elif algorithm_type == "gradient_boosting":
            model = GradientBoostingRegressor(
                n_estimators=min(100, problem_size),
                random_state=self.seed
            )
        else:
            # Fallback to simple model
            model = LogisticRegression(random_state=self.seed)
        
        try:
            model.fit(X, y)
            predictions = model.predict(X)
            execution_time = time.time() - start_time
            
            # Solution quality (1 - MSE normalized)
            mse = np.mean((y - predictions) ** 2)
            quality = max(0, 1 - mse / np.var(y))
            
        except Exception as e:
            logger.warning(f"Classical simulation error: {e}")
            execution_time = problem_size * 0.01  # Estimate
            quality = 0.5  # Mediocre quality
        
        return execution_time, quality
    
    def scaling_analysis(
        self,
        problem_sizes: List[int],
        quantum_circuit_generator: Callable,
        classical_algorithm: str = "random_forest"
    ) -> Dict[str, Any]:
        """Perform scaling analysis across problem sizes."""
        
        logger.info("Performing scaling analysis")
        
        quantum_times = []
        classical_times = []
        quantum_qualities = []
        classical_qualities = []
        
        for size in problem_sizes:
            try:
                # Quantum performance
                n_qubits = min(size, self.max_qubits)
                depth = min(size, self.max_circuit_depth)
                
                quantum_circuit = quantum_circuit_generator(n_qubits, depth)
                
                start_time = time.time()
                quantum_result = quantum_circuit()
                quantum_time = time.time() - start_time
                
                # Quantum solution quality (probability distribution entropy)
                quantum_quality = -np.sum(quantum_result * np.log2(quantum_result + 1e-12))
                quantum_quality = quantum_quality / n_qubits  # Normalize
                
                quantum_times.append(quantum_time)
                quantum_qualities.append(quantum_quality)
                
            except Exception as e:
                logger.warning(f"Quantum scaling error at size {size}: {e}")
                quantum_times.append(float('inf'))
                quantum_qualities.append(0.0)
            
            # Classical performance
            try:
                classical_time, classical_quality = self.simulate_classical_algorithm(
                    size, classical_algorithm
                )
                classical_times.append(classical_time)
                classical_qualities.append(classical_quality)
                
            except Exception as e:
                logger.warning(f"Classical scaling error at size {size}: {e}")
                classical_times.append(size * 0.01)
                classical_qualities.append(0.5)
        
        # Fit scaling exponents
        valid_indices = [i for i, (qt, ct) in enumerate(zip(quantum_times, classical_times))
                        if qt != float('inf') and ct > 0]
        
        if len(valid_indices) > 2:
            valid_sizes = [problem_sizes[i] for i in valid_indices]
            valid_quantum_times = [quantum_times[i] for i in valid_indices]
            valid_classical_times = [classical_times[i] for i in valid_indices]
            
            # Log-log regression for scaling exponents
            log_sizes = np.log10(valid_sizes)
            log_quantum_times = np.log10(valid_quantum_times)
            log_classical_times = np.log10(valid_classical_times)
            
            quantum_slope, _, _, _, _ = linregress(log_sizes, log_quantum_times)
            classical_slope, _, _, _, _ = linregress(log_sizes, log_classical_times)
            
        else:
            quantum_slope = 2.0  # Assume quadratic
            classical_slope = 1.5  # Assume sub-quadratic
        
        # Determine scaling regime
        if quantum_slope > 3.0:
            scaling_regime = ScalingRegime.SUPER_EXPONENTIAL
        elif quantum_slope > 2.0:
            scaling_regime = ScalingRegime.EXPONENTIAL  
        elif quantum_slope > 1.0:
            scaling_regime = ScalingRegime.POLYNOMIAL
        else:
            scaling_regime = ScalingRegime.SUBLINEAR
        
        # Find crossover point
        crossover_point = self._find_crossover_point(
            problem_sizes, quantum_times, classical_times
        )
        
        return {
            "quantum_times": quantum_times,
            "classical_times": classical_times,
            "quantum_qualities": quantum_qualities,
            "classical_qualities": classical_qualities,
            "quantum_scaling_exponent": quantum_slope,
            "classical_scaling_exponent": classical_slope,
            "scaling_advantage": classical_slope - quantum_slope,
            "scaling_regime": scaling_regime,
            "crossover_point": crossover_point
        }
    
    def _find_crossover_point(
        self,
        sizes: List[int],
        quantum_times: List[float],
        classical_times: List[float]
    ) -> int:
        """Find crossover point where quantum becomes advantageous."""
        
        for i, (size, qt, ct) in enumerate(zip(sizes, quantum_times, classical_times)):
            if qt != float('inf') and qt < ct:
                return size
        
        return sizes[-1] if sizes else 0
    
    def sample_complexity_analysis(
        self,
        target_accuracies: List[float] = None,
        problem_size: int = 100
    ) -> Dict[str, Any]:
        """Analyze sample complexity for target accuracies."""
        
        if target_accuracies is None:
            target_accuracies = [0.5, 0.7, 0.8, 0.9, 0.95]
        
        logger.info("Analyzing sample complexity")
        
        quantum_complexities = []
        classical_complexities = []
        
        # Generate test circuit
        n_qubits = min(problem_size, self.max_qubits)
        test_circuit = self.create_random_circuit(n_qubits, 10)
        
        for target_acc in target_accuracies:
            # Quantum sample complexity (shots needed for accuracy)
            quantum_samples = self._estimate_quantum_samples(test_circuit, target_acc)
            quantum_complexities.append(quantum_samples)
            
            # Classical sample complexity
            classical_samples = self._estimate_classical_samples(problem_size, target_acc)
            classical_complexities.append(classical_samples)
        
        # Sample efficiency advantage
        if classical_complexities and quantum_complexities:
            efficiency_ratios = [c/q if q > 0 else 1.0 
                               for c, q in zip(classical_complexities, quantum_complexities)]
            sample_efficiency = np.mean(efficiency_ratios)
        else:
            sample_efficiency = 1.0
        
        return {
            "quantum_sample_complexities": quantum_complexities,
            "classical_sample_complexities": classical_complexities,
            "sample_efficiency_advantage": sample_efficiency,
            "target_accuracies": target_accuracies
        }
    
    def _estimate_quantum_samples(
        self,
        circuit: Callable,
        target_accuracy: float
    ) -> int:
        """Estimate quantum samples needed for target accuracy."""
        
        # Theoretical estimate: O(1/ε²) for ε accuracy
        base_samples = int(1.0 / ((1.0 - target_accuracy) ** 2))
        
        # Add quantum-specific factors
        quantum_overhead = max(1, self.max_qubits // 4)  # Quantum measurement overhead
        
        return base_samples * quantum_overhead
    
    def _estimate_classical_samples(
        self,
        problem_size: int,
        target_accuracy: float
    ) -> int:
        """Estimate classical samples needed for target accuracy."""
        
        # Classical learning theory estimate
        vc_dimension = min(problem_size, 100)  # Estimate VC dimension
        
        # Sample complexity: O(d/ε * log(1/δ)) where d=VC dimension
        epsilon = 1.0 - target_accuracy
        delta = 0.1  # Confidence parameter
        
        samples = int((vc_dimension / epsilon) * np.log(1.0 / delta))
        
        return max(samples, 10)  # Minimum samples
    
    def resource_efficiency_analysis(
        self,
        problem_sizes: List[int]
    ) -> Dict[str, Any]:
        """Analyze resource efficiency scaling."""
        
        logger.info("Analyzing resource efficiency")
        
        quantum_resources = []
        classical_resources = []
        
        for size in problem_sizes:
            # Quantum resource usage (qubits * depth * shots)
            n_qubits = min(size, self.max_qubits)
            depth = min(size, self.max_circuit_depth)
            quantum_resource = n_qubits * depth * self.shots
            quantum_resources.append(quantum_resource)
            
            # Classical resource usage (flops estimate)
            classical_resource = size ** 2 * np.log(size)  # Typical ML algorithm
            classical_resources.append(classical_resource)
        
        # Resource efficiency ratio
        if quantum_resources and classical_resources:
            efficiency_ratios = [c/q if q > 0 else float('inf') 
                               for c, q in zip(classical_resources, quantum_resources)]
            mean_efficiency = np.mean([r for r in efficiency_ratios if r != float('inf')])
        else:
            mean_efficiency = 1.0
        
        return {
            "quantum_resource_usage": quantum_resources,
            "classical_resource_usage": classical_resources,
            "resource_efficiency_ratio": mean_efficiency
        }
    
    def approximation_ratio_analysis(
        self,
        problem_sizes: List[int],
        theoretical_optima: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Analyze approximation quality vs theoretical limits."""
        
        logger.info("Analyzing approximation ratios")
        
        if theoretical_optima is None:
            theoretical_optima = [1.0] * len(problem_sizes)  # Perfect solutions
        
        approximation_ratios = []
        
        for size, optimal in zip(problem_sizes, theoretical_optima):
            # Create test circuit for approximation
            n_qubits = min(size, self.max_qubits)
            circuit = self.create_random_circuit(n_qubits, 5)
            
            try:
                result = circuit()
                # Approximation quality (entropy-based measure)
                quantum_solution = -np.sum(result * np.log2(result + 1e-12))
                approximation_ratio = quantum_solution / optimal if optimal > 0 else 0
                approximation_ratios.append(min(approximation_ratio, 1.0))
                
            except Exception as e:
                logger.warning(f"Approximation analysis error: {e}")
                approximation_ratios.append(0.5)  # Default mediocre ratio
        
        # Approximation advantage (how close to theoretical limits)
        mean_ratio = np.mean(approximation_ratios) if approximation_ratios else 0.5
        
        return {
            "approximation_ratios": approximation_ratios,
            "theoretical_limits": theoretical_optima,
            "approximation_advantage": mean_ratio
        }
    
    def statistical_validation(
        self,
        quantum_performance: List[float],
        classical_performance: List[float]
    ) -> Dict[str, Any]:
        """Perform statistical validation of supremacy claims."""
        
        logger.info("Performing statistical validation")
        
        # Statistical test for performance difference
        from scipy.stats import ttest_ind, wilcoxon
        
        try:
            # T-test for mean difference
            t_stat, p_value = ttest_ind(quantum_performance, classical_performance)
            
            # Wilcoxon test for non-parametric comparison
            w_stat, w_p_value = wilcoxon(
                quantum_performance, classical_performance,
                alternative='greater'
            )
            
        except Exception as e:
            logger.warning(f"Statistical test error: {e}")
            p_value = 0.5
            w_p_value = 0.5
        
        # Confidence intervals
        quantum_mean = np.mean(quantum_performance)
        classical_mean = np.mean(classical_performance)
        
        quantum_std = np.std(quantum_performance) / np.sqrt(len(quantum_performance))
        classical_std = np.std(classical_performance) / np.sqrt(len(classical_performance))
        
        confidence_intervals = {
            "quantum": (quantum_mean - 1.96 * quantum_std, 
                       quantum_mean + 1.96 * quantum_std),
            "classical": (classical_mean - 1.96 * classical_std,
                         classical_mean + 1.96 * classical_std)
        }
        
        # Statistical power estimate
        effect_size = abs(quantum_mean - classical_mean) / np.sqrt(
            (np.var(quantum_performance) + np.var(classical_performance)) / 2
        )
        
        # Rough power estimate based on effect size
        if effect_size > 0.8:
            statistical_power = 0.9
        elif effect_size > 0.5:
            statistical_power = 0.7
        elif effect_size > 0.2:
            statistical_power = 0.5
        else:
            statistical_power = 0.3
        
        return {
            "supremacy_p_value": min(p_value, w_p_value),
            "confidence_intervals": confidence_intervals,
            "statistical_power": statistical_power,
            "effect_size": effect_size
        }
    
    def comprehensive_supremacy_analysis(
        self,
        problem_sizes: List[int] = None,
        quantum_circuit_generator: Optional[Callable] = None
    ) -> SupremacyResult:
        """Perform comprehensive quantum supremacy analysis."""
        
        logger.info("Starting comprehensive quantum supremacy analysis")
        
        if problem_sizes is None:
            problem_sizes = [4, 8, 12, 16, 20]
        
        if quantum_circuit_generator is None:
            quantum_circuit_generator = self.create_random_circuit
        
        # Scaling analysis
        scaling_results = self.scaling_analysis(
            problem_sizes, quantum_circuit_generator
        )
        
        # Sample complexity analysis
        sample_results = self.sample_complexity_analysis()
        
        # Resource efficiency analysis
        resource_results = self.resource_efficiency_analysis(problem_sizes)
        
        # Approximation ratio analysis
        approximation_results = self.approximation_ratio_analysis(problem_sizes)
        
        # Statistical validation
        statistical_results = self.statistical_validation(
            scaling_results["quantum_qualities"],
            scaling_results["classical_qualities"]
        )
        
        # Supremacy assessment
        supremacy_achieved = (
            scaling_results["scaling_advantage"] > 0.5 and
            statistical_results["supremacy_p_value"] < 0.05 and
            scaling_results["crossover_point"] <= max(problem_sizes)
        )
        
        # Supremacy confidence
        confidence_factors = [
            min(1.0, scaling_results["scaling_advantage"]),
            1.0 - statistical_results["supremacy_p_value"],
            min(1.0, sample_results["sample_efficiency_advantage"] / 2.0),
            min(1.0, approximation_results["approximation_advantage"])
        ]
        supremacy_confidence = np.mean(confidence_factors)
        
        # Supremacy category
        if supremacy_confidence > 0.8:
            supremacy_category = "strong"
        elif supremacy_confidence > 0.6:
            supremacy_category = "moderate"
        elif supremacy_confidence > 0.4:
            supremacy_category = "conditional"
        else:
            supremacy_category = "none"
        
        result = SupremacyResult(
            problem_sizes=problem_sizes,
            quantum_scaling_exponent=scaling_results["quantum_scaling_exponent"],
            classical_scaling_exponent=scaling_results["classical_scaling_exponent"],
            scaling_advantage=scaling_results["scaling_advantage"],
            scaling_regime=scaling_results["scaling_regime"],
            crossover_point=scaling_results["crossover_point"],
            quantum_execution_times=scaling_results["quantum_times"],
            classical_execution_times=scaling_results["classical_times"],
            quantum_solution_qualities=scaling_results["quantum_qualities"],
            classical_solution_qualities=scaling_results["classical_qualities"],
            quantum_sample_complexities=sample_results["quantum_sample_complexities"],
            classical_sample_complexities=sample_results["classical_sample_complexities"],
            sample_efficiency_advantage=sample_results["sample_efficiency_advantage"],
            quantum_resource_usage=resource_results["quantum_resource_usage"],
            classical_resource_usage=resource_results["classical_resource_usage"],
            resource_efficiency_ratio=resource_results["resource_efficiency_ratio"],
            approximation_ratios=approximation_results["approximation_ratios"],
            theoretical_limits=approximation_results["theoretical_limits"],
            approximation_advantage=approximation_results["approximation_advantage"],
            supremacy_p_value=statistical_results["supremacy_p_value"],
            confidence_intervals=statistical_results["confidence_intervals"],
            statistical_power=statistical_results["statistical_power"],
            supremacy_achieved=supremacy_achieved,
            supremacy_confidence=supremacy_confidence,
            supremacy_category=supremacy_category
        )
        
        logger.info(f"Supremacy analysis complete. Category: {supremacy_category}")
        
        return result


# Import time for timing measurements
import time