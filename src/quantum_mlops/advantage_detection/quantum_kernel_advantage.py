"""Quantum Kernel Advantage Detection Algorithm.

This module implements novel algorithms for detecting quantum advantage in kernel methods,
including spectral analysis, expressivity measurements, and feature map advantage detection.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.linalg import eigvals, norm
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

from ..logging_config import get_logger
from ..exceptions import QuantumMLOpsException

logger = get_logger(__name__)


class QuantumFeatureMap(Enum):
    """Supported quantum feature map types."""
    
    IQP = "iqp"  # Instantaneous Quantum Polynomial
    ZFEATURE = "zfeature"  # Z-feature map
    QAOA_INSPIRED = "qaoa_inspired"  # QAOA-inspired feature map
    HARDWARE_EFFICIENT = "hardware_efficient"  # Hardware-efficient ansatz


@dataclass
class KernelAdvantageResult:
    """Results from quantum kernel advantage analysis."""
    
    # Spectral advantage metrics
    quantum_spectral_gap: float
    classical_spectral_gap: float  
    spectral_advantage: float
    
    # Expressivity metrics
    quantum_expressivity: float
    classical_expressivity: float
    expressivity_advantage: float
    
    # Performance metrics
    quantum_accuracy: float
    classical_accuracy: float
    performance_advantage: float
    
    # Statistical significance
    p_value: float
    confidence_interval: Tuple[float, float]
    statistically_significant: bool
    
    # Resource analysis
    quantum_circuit_depth: int
    quantum_gate_count: int
    classical_feature_dimension: int
    
    # Advantage summary
    overall_advantage_score: float
    advantage_category: str  # "strong", "moderate", "weak", "none"


class QuantumKernelAnalyzer:
    """Advanced quantum kernel advantage detection system."""
    
    def __init__(
        self,
        n_qubits: int,
        feature_map: QuantumFeatureMap = QuantumFeatureMap.IQP,
        shots: int = 1000,
        seed: Optional[int] = None
    ) -> None:
        """Initialize quantum kernel analyzer.
        
        Args:
            n_qubits: Number of qubits for quantum feature map
            feature_map: Type of quantum feature map to use
            shots: Number of shots for quantum measurements
            seed: Random seed for reproducibility
        """
        self.n_qubits = n_qubits
        self.feature_map = feature_map
        self.shots = shots
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
            
        if not PENNYLANE_AVAILABLE:
            raise QuantumMLOpsException(
                "PennyLane is required for quantum kernel analysis"
            )
            
        # Initialize quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self._create_quantum_feature_map()
        
        logger.info(
            f"Initialized QuantumKernelAnalyzer with {n_qubits} qubits, "
            f"{feature_map.value} feature map"
        )
    
    def _create_quantum_feature_map(self) -> None:
        """Create quantum feature map circuit."""
        
        @qml.qnode(self.dev)
        def quantum_feature_map(x1: np.ndarray, x2: np.ndarray) -> float:
            """Quantum feature map circuit for kernel computation."""
            
            # Encode first data point
            self._encode_data(x1, "first")
            
            # Apply adjoint of second data point encoding
            qml.adjoint(lambda: self._encode_data(x2, "second"))()
            
            # Return overlap measurement
            return qml.probs(wires=range(self.n_qubits))[0]
        
        self.quantum_kernel_circuit = quantum_feature_map
    
    def _encode_data(self, x: np.ndarray, label: str) -> None:
        """Encode classical data into quantum state."""
        
        if self.feature_map == QuantumFeatureMap.IQP:
            self._iqp_encoding(x)
        elif self.feature_map == QuantumFeatureMap.ZFEATURE:
            self._zfeature_encoding(x)
        elif self.feature_map == QuantumFeatureMap.QAOA_INSPIRED:
            self._qaoa_inspired_encoding(x)
        elif self.feature_map == QuantumFeatureMap.HARDWARE_EFFICIENT:
            self._hardware_efficient_encoding(x)
    
    def _iqp_encoding(self, x: np.ndarray) -> None:
        """Instantaneous Quantum Polynomial (IQP) feature map."""
        
        # Apply Hadamard gates
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
        
        # Apply single-qubit rotations
        for i in range(self.n_qubits):
            if i < len(x):
                qml.RZ(x[i], wires=i)
        
        # Apply entangling gates (IQP structure)
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                if i < len(x) and j < len(x):
                    qml.CZ(wires=[i, j])
                    qml.RZ(x[i] * x[j], wires=j)
                    qml.CZ(wires=[i, j])
    
    def _zfeature_encoding(self, x: np.ndarray) -> None:
        """Z-feature map encoding."""
        
        for i in range(min(len(x), self.n_qubits)):
            qml.Hadamard(wires=i)
            qml.RZ(2 * x[i], wires=i)
        
        # Add entanglement layer
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    
    def _qaoa_inspired_encoding(self, x: np.ndarray) -> None:
        """QAOA-inspired feature map."""
        
        # Problem Hamiltonian-inspired layer
        for i in range(min(len(x), self.n_qubits)):
            qml.RZ(x[i], wires=i)
        
        # Mixer Hamiltonian-inspired layer
        for i in range(self.n_qubits):
            qml.RX(np.pi/2, wires=i)
        
        # Entangling layer
        for i in range(self.n_qubits - 1):
            qml.CZ(wires=[i, i + 1])
    
    def _hardware_efficient_encoding(self, x: np.ndarray) -> None:
        """Hardware-efficient feature map."""
        
        # Layer 1: Single-qubit rotations
        for i in range(min(len(x), self.n_qubits)):
            qml.RY(x[i], wires=i)
        
        # Layer 2: Entangling gates
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        
        # Layer 3: More single-qubit rotations  
        for i in range(min(len(x), self.n_qubits)):
            if i < len(x):
                qml.RZ(x[i]**2, wires=i)
    
    def compute_quantum_kernel_matrix(
        self, 
        X: np.ndarray
    ) -> np.ndarray:
        """Compute quantum kernel matrix for dataset."""
        
        n_samples = X.shape[0]
        kernel_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i, n_samples):
                kernel_value = self.quantum_kernel_circuit(X[i], X[j])
                kernel_matrix[i, j] = kernel_value
                kernel_matrix[j, i] = kernel_value  # Symmetric matrix
        
        return kernel_matrix
    
    def compute_classical_kernel_matrix(
        self,
        X: np.ndarray,
        kernel: str = "rbf",
        **kernel_params: Any
    ) -> np.ndarray:
        """Compute classical kernel matrix for comparison."""
        
        return pairwise_kernels(X, metric=kernel, **kernel_params)
    
    def spectral_analysis(
        self,
        quantum_kernel: np.ndarray,
        classical_kernel: np.ndarray
    ) -> Dict[str, float]:
        """Perform spectral analysis of kernel matrices."""
        
        # Compute eigenvalues
        quantum_eigenvals = eigvals(quantum_kernel)
        classical_eigenvals = eigvals(classical_kernel)
        
        # Real parts only (kernels should be Hermitian)
        quantum_eigenvals = np.real(quantum_eigenvals)
        classical_eigenvals = np.real(classical_eigenvals)
        
        # Sort in descending order
        quantum_eigenvals = np.sort(quantum_eigenvals)[::-1]
        classical_eigenvals = np.sort(classical_eigenvals)[::-1]
        
        # Compute spectral gaps
        quantum_spectral_gap = quantum_eigenvals[0] - np.median(quantum_eigenvals)
        classical_spectral_gap = classical_eigenvals[0] - np.median(classical_eigenvals)
        
        # Compute effective rank (participation ratio)
        quantum_effective_rank = (np.sum(quantum_eigenvals)**2 / 
                                 np.sum(quantum_eigenvals**2))
        classical_effective_rank = (np.sum(classical_eigenvals)**2 / 
                                   np.sum(classical_eigenvals**2))
        
        return {
            "quantum_spectral_gap": quantum_spectral_gap,
            "classical_spectral_gap": classical_spectral_gap,
            "spectral_advantage": quantum_spectral_gap - classical_spectral_gap,
            "quantum_effective_rank": quantum_effective_rank,
            "classical_effective_rank": classical_effective_rank,
            "rank_advantage": quantum_effective_rank - classical_effective_rank
        }
    
    def expressivity_analysis(
        self,
        X: np.ndarray,
        n_random_samples: int = 1000
    ) -> Dict[str, float]:
        """Measure expressivity of quantum vs classical kernels."""
        
        # Generate random parameter samples
        random_params = np.random.uniform(-np.pi, np.pi, (n_random_samples, X.shape[1]))
        
        # Compute kernel values for random parameters
        quantum_values = []
        classical_values = []
        
        for params in random_params:
            # Quantum kernel expressivity
            quantum_sample = self.quantum_kernel_circuit(
                X[0], params[:len(X[0])]
            )
            quantum_values.append(quantum_sample)
            
            # Classical kernel expressivity (RBF with varying gamma)
            gamma = np.exp(params[0])  # Use first parameter for gamma
            classical_sample = np.exp(-gamma * norm(X[0] - params[:len(X[0])])**2)
            classical_values.append(classical_sample)
        
        # Compute expressivity metrics
        quantum_variance = np.var(quantum_values)
        classical_variance = np.var(classical_values)
        
        quantum_range = np.max(quantum_values) - np.min(quantum_values)
        classical_range = np.max(classical_values) - np.min(classical_values)
        
        return {
            "quantum_expressivity": quantum_variance,
            "classical_expressivity": classical_variance,
            "expressivity_advantage": quantum_variance - classical_variance,
            "quantum_range": quantum_range,
            "classical_range": classical_range
        }
    
    def performance_comparison(
        self,
        X: np.ndarray,
        y: np.ndarray,
        quantum_kernel: np.ndarray,
        classical_kernel: np.ndarray,
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """Compare classification performance using quantum vs classical kernels."""
        
        # Create precomputed kernel SVMs
        quantum_svm = SVC(kernel="precomputed")
        classical_svm = SVC(kernel="precomputed")
        
        # Cross-validation scores
        quantum_scores = cross_val_score(
            quantum_svm, quantum_kernel, y, cv=cv_folds, scoring="accuracy"
        )
        classical_scores = cross_val_score(
            classical_svm, classical_kernel, y, cv=cv_folds, scoring="accuracy"
        )
        
        # Statistical test
        from scipy.stats import ttest_rel
        t_stat, p_value = ttest_rel(quantum_scores, classical_scores)
        
        # Confidence interval for difference
        diff_scores = quantum_scores - classical_scores
        mean_diff = np.mean(diff_scores)
        std_diff = np.std(diff_scores, ddof=1)
        se_diff = std_diff / np.sqrt(len(diff_scores))
        
        # 95% confidence interval
        t_critical = 2.776  # For 4 degrees of freedom (5-fold CV)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        return {
            "quantum_accuracy": np.mean(quantum_scores),
            "classical_accuracy": np.mean(classical_scores),
            "performance_advantage": mean_diff,
            "quantum_std": np.std(quantum_scores),
            "classical_std": np.std(classical_scores),
            "p_value": p_value,
            "confidence_interval": (ci_lower, ci_upper),
            "statistically_significant": p_value < 0.05
        }
    
    def comprehensive_advantage_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classical_kernel: str = "rbf",
        **classical_kernel_params: Any
    ) -> KernelAdvantageResult:
        """Perform comprehensive quantum kernel advantage analysis."""
        
        logger.info("Starting comprehensive quantum kernel advantage analysis")
        
        # Compute kernel matrices
        quantum_kernel = self.compute_quantum_kernel_matrix(X)
        classical_kernel_matrix = self.compute_classical_kernel_matrix(
            X, kernel=classical_kernel, **classical_kernel_params
        )
        
        # Spectral analysis
        spectral_results = self.spectral_analysis(
            quantum_kernel, classical_kernel_matrix
        )
        
        # Expressivity analysis
        expressivity_results = self.expressivity_analysis(X)
        
        # Performance comparison
        performance_results = self.performance_comparison(
            X, y, quantum_kernel, classical_kernel_matrix
        )
        
        # Circuit analysis
        circuit_depth = self._estimate_circuit_depth()
        gate_count = self._estimate_gate_count()
        
        # Overall advantage score (weighted combination)
        overall_score = (
            0.3 * max(0, spectral_results["spectral_advantage"]) +
            0.3 * max(0, expressivity_results["expressivity_advantage"]) +
            0.4 * max(0, performance_results["performance_advantage"])
        )
        
        # Categorize advantage
        if overall_score > 0.1:
            advantage_category = "strong"
        elif overall_score > 0.05:
            advantage_category = "moderate"
        elif overall_score > 0.01:
            advantage_category = "weak"
        else:
            advantage_category = "none"
        
        result = KernelAdvantageResult(
            quantum_spectral_gap=spectral_results["quantum_spectral_gap"],
            classical_spectral_gap=spectral_results["classical_spectral_gap"],
            spectral_advantage=spectral_results["spectral_advantage"],
            quantum_expressivity=expressivity_results["quantum_expressivity"],
            classical_expressivity=expressivity_results["classical_expressivity"],
            expressivity_advantage=expressivity_results["expressivity_advantage"],
            quantum_accuracy=performance_results["quantum_accuracy"],
            classical_accuracy=performance_results["classical_accuracy"],
            performance_advantage=performance_results["performance_advantage"],
            p_value=performance_results["p_value"],
            confidence_interval=performance_results["confidence_interval"],
            statistically_significant=performance_results["statistically_significant"],
            quantum_circuit_depth=circuit_depth,
            quantum_gate_count=gate_count,
            classical_feature_dimension=X.shape[1],
            overall_advantage_score=overall_score,
            advantage_category=advantage_category
        )
        
        logger.info(f"Analysis complete. Overall advantage: {advantage_category}")
        
        return result
    
    def _estimate_circuit_depth(self) -> int:
        """Estimate quantum circuit depth."""
        
        # Feature map dependent depth estimation
        if self.feature_map == QuantumFeatureMap.IQP:
            return 3  # H + RZ + CZ layers
        elif self.feature_map == QuantumFeatureMap.ZFEATURE:
            return 3  # H + RZ + CNOT layers
        elif self.feature_map == QuantumFeatureMap.QAOA_INSPIRED:
            return 3  # RZ + RX + CZ layers
        elif self.feature_map == QuantumFeatureMap.HARDWARE_EFFICIENT:
            return 4  # RY + CNOT + RZ layers
        
        return 3  # Default estimate
    
    def _estimate_gate_count(self) -> int:
        """Estimate total gate count."""
        
        single_qubit_gates = self.n_qubits * 2  # Estimate 2 single-qubit gates per qubit
        two_qubit_gates = max(0, self.n_qubits - 1)  # Linear connectivity estimate
        
        return single_qubit_gates + two_qubit_gates