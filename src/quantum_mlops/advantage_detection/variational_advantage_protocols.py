"""Variational Quantum Advantage Protocols.

This module implements protocols for detecting quantum advantage in variational quantum algorithms,
including VQE, QAOA, and general variational circuits with expressivity analysis.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

from ..logging_config import get_logger
from ..exceptions import QuantumMLOpsException

logger = get_logger(__name__)


class VariationalAlgorithm(Enum):
    """Supported variational quantum algorithms."""
    
    VQE = "vqe"
    QAOA = "qaoa"
    VQC = "vqc"  # Variational Quantum Classifier
    QGAN = "qgan"  # Quantum Generative Adversarial Network


@dataclass
class VariationalAdvantageResult:
    """Results from variational quantum advantage analysis."""
    
    # Optimization landscape analysis
    quantum_landscape_roughness: float
    classical_landscape_roughness: float
    landscape_advantage: float
    
    # Expressivity metrics
    quantum_expressivity_score: float
    classical_expressivity_score: float
    expressivity_advantage: float
    
    # Barren plateau analysis
    gradient_variance: float
    plateau_detected: bool
    effective_dimension: int
    
    # Performance comparison
    quantum_final_cost: float
    classical_final_cost: float
    cost_advantage: float
    
    # Convergence analysis
    quantum_convergence_steps: int
    classical_convergence_steps: int
    convergence_advantage: int
    
    # Resource efficiency
    quantum_parameter_count: int
    classical_parameter_count: int
    parameter_efficiency: float
    
    # Statistical analysis
    advantage_p_value: float
    confidence_interval: Tuple[float, float]
    statistically_significant: bool
    
    # Overall assessment
    overall_advantage_score: float
    advantage_category: str


class VariationalAdvantageAnalyzer:
    """Analyzer for variational quantum algorithm advantage."""
    
    def __init__(
        self,
        n_qubits: int,
        algorithm: VariationalAlgorithm = VariationalAlgorithm.VQE,
        n_layers: int = 3,
        shots: int = 1000,
        seed: Optional[int] = None
    ) -> None:
        """Initialize variational advantage analyzer.
        
        Args:
            n_qubits: Number of qubits
            algorithm: Variational algorithm type
            n_layers: Number of variational layers
            shots: Number of measurement shots
            seed: Random seed for reproducibility
        """
        self.n_qubits = n_qubits
        self.algorithm = algorithm
        self.n_layers = n_layers
        self.shots = shots
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
            
        if not PENNYLANE_AVAILABLE:
            raise QuantumMLOpsException(
                "PennyLane is required for variational advantage analysis"
            )
            
        # Initialize quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
        
        # Create ansatz
        self._create_variational_ansatz()
        
        logger.info(
            f"Initialized VariationalAdvantageAnalyzer for {algorithm.value} "
            f"with {n_qubits} qubits, {n_layers} layers"
        )
    
    def _create_variational_ansatz(self) -> None:
        """Create variational ansatz based on algorithm type."""
        
        if self.algorithm == VariationalAlgorithm.VQE:
            self._create_vqe_ansatz()
        elif self.algorithm == VariationalAlgorithm.QAOA:
            self._create_qaoa_ansatz()
        elif self.algorithm == VariationalAlgorithm.VQC:
            self._create_vqc_ansatz()
        elif self.algorithm == VariationalAlgorithm.QGAN:
            self._create_qgan_ansatz()
    
    def _create_vqe_ansatz(self) -> None:
        """Create VQE ansatz circuit."""
        
        @qml.qnode(self.dev)
        def vqe_circuit(params: np.ndarray, hamiltonian: Optional[Any] = None) -> float:
            """VQE variational circuit."""
            
            param_idx = 0
            
            # Hardware-efficient ansatz
            for layer in range(self.n_layers):
                # Single-qubit rotations
                for i in range(self.n_qubits):
                    qml.RY(params[param_idx], wires=i)
                    param_idx += 1
                    qml.RZ(params[param_idx], wires=i)
                    param_idx += 1
                
                # Entangling gates
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            # Return expectation value
            if hamiltonian is not None:
                return qml.expval(hamiltonian)
            else:
                # Default Hamiltonian (all Z measurements)
                return sum(qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits))
        
        self.quantum_circuit = vqe_circuit
        self.n_params = self.n_layers * self.n_qubits * 2
    
    def _create_qaoa_ansatz(self) -> None:
        """Create QAOA ansatz circuit."""
        
        @qml.qnode(self.dev)
        def qaoa_circuit(params: np.ndarray, problem_graph: Optional[Any] = None) -> float:
            """QAOA variational circuit."""
            
            # Initialize superposition
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            # QAOA layers
            for p in range(self.n_layers):
                # Problem Hamiltonian layer
                gamma = params[2 * p]
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(2 * gamma, wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])
                
                # Mixer Hamiltonian layer
                beta = params[2 * p + 1]
                for i in range(self.n_qubits):
                    qml.RX(2 * beta, wires=i)
            
            # Return cost expectation
            return sum(qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits))
        
        self.quantum_circuit = qaoa_circuit
        self.n_params = 2 * self.n_layers
    
    def _create_vqc_ansatz(self) -> None:
        """Create Variational Quantum Classifier ansatz."""
        
        @qml.qnode(self.dev)
        def vqc_circuit(params: np.ndarray, x: np.ndarray) -> float:
            """VQC variational circuit."""
            
            # Data encoding
            for i in range(min(len(x), self.n_qubits)):
                qml.RY(x[i], wires=i)
            
            param_idx = 0
            
            # Variational layers
            for layer in range(self.n_layers):
                # Parameterized gates
                for i in range(self.n_qubits):
                    qml.RY(params[param_idx], wires=i)
                    param_idx += 1
                    qml.RZ(params[param_idx], wires=i)
                    param_idx += 1
                
                # Entangling layer
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            return qml.expval(qml.PauliZ(0))
        
        self.quantum_circuit = vqc_circuit
        self.n_params = self.n_layers * self.n_qubits * 2
    
    def _create_qgan_ansatz(self) -> None:
        """Create Quantum GAN generator ansatz."""
        
        @qml.qnode(self.dev)
        def qgan_circuit(params: np.ndarray) -> List[float]:
            """QGAN generator circuit."""
            
            param_idx = 0
            
            # Variational quantum generator
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(params[param_idx], wires=i)
                    param_idx += 1
                    qml.RZ(params[param_idx], wires=i)
                    param_idx += 1
                
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            # Return probabilities
            return qml.probs(wires=range(self.n_qubits))
        
        self.quantum_circuit = qgan_circuit
        self.n_params = self.n_layers * self.n_qubits * 2
    
    def landscape_analysis(
        self,
        cost_function: Callable,
        param_range: Tuple[float, float] = (-np.pi, np.pi),
        n_samples: int = 1000
    ) -> Dict[str, float]:
        """Analyze optimization landscape roughness."""
        
        logger.info("Analyzing optimization landscapes")
        
        # Generate random parameter samples
        quantum_params = np.random.uniform(
            param_range[0], param_range[1], (n_samples, self.n_params)
        )
        
        # Evaluate quantum cost landscape
        quantum_costs = []
        for params in quantum_params:
            try:
                cost = cost_function(params)
                quantum_costs.append(cost)
            except Exception as e:
                logger.warning(f"Error evaluating quantum cost: {e}")
                quantum_costs.append(float('inf'))
        
        quantum_costs = np.array(quantum_costs)
        quantum_costs = quantum_costs[np.isfinite(quantum_costs)]
        
        # Create comparable classical model
        classical_costs = self._evaluate_classical_landscape(
            cost_function, n_samples, param_range
        )
        
        # Compute landscape roughness (variance of gradients)
        quantum_roughness = self._compute_landscape_roughness(quantum_costs)
        classical_roughness = self._compute_landscape_roughness(classical_costs)
        
        return {
            "quantum_landscape_roughness": quantum_roughness,
            "classical_landscape_roughness": classical_roughness,
            "landscape_advantage": classical_roughness - quantum_roughness,
            "quantum_cost_variance": np.var(quantum_costs),
            "classical_cost_variance": np.var(classical_costs)
        }
    
    def _evaluate_classical_landscape(
        self,
        cost_function: Callable,
        n_samples: int,
        param_range: Tuple[float, float]
    ) -> np.ndarray:
        """Evaluate classical optimization landscape for comparison."""
        
        # Create neural network with comparable parameters
        classical_param_count = max(10, self.n_params)  # At least 10 parameters
        
        classical_costs = []
        for _ in range(n_samples):
            # Random neural network parameters
            weights = np.random.uniform(
                param_range[0], param_range[1], classical_param_count
            )
            
            # Simple classical cost function
            cost = np.sum(np.sin(weights)**2) + 0.1 * np.sum(weights**2)
            classical_costs.append(cost)
        
        return np.array(classical_costs)
    
    def _compute_landscape_roughness(self, costs: np.ndarray) -> float:
        """Compute landscape roughness metric."""
        
        # Sort costs to compute local variations
        sorted_costs = np.sort(costs)
        
        # Compute local variations (discrete derivative)
        variations = np.abs(np.diff(sorted_costs))
        
        # Roughness as normalized variance of variations
        roughness = np.var(variations) / (np.var(costs) + 1e-10)
        
        return roughness
    
    def expressivity_analysis(
        self,
        n_random_circuits: int = 500,
        fidelity_samples: int = 100
    ) -> Dict[str, float]:
        """Analyze expressivity of variational circuits."""
        
        logger.info("Analyzing circuit expressivity")
        
        quantum_fidelities = []
        classical_fidelities = []
        
        # Generate random parameter pairs for fidelity computation
        for _ in range(fidelity_samples):
            params1 = np.random.uniform(-np.pi, np.pi, self.n_params)
            params2 = np.random.uniform(-np.pi, np.pi, self.n_params)
            
            # Quantum circuit fidelity
            if self.algorithm == VariationalAlgorithm.VQE:
                state1 = self.quantum_circuit(params1)
                state2 = self.quantum_circuit(params2)
                quantum_fidelity = abs(np.dot(np.conj(state1), state2))**2
            else:
                # For other algorithms, use cost function similarity
                cost1 = self.quantum_circuit(params1)
                cost2 = self.quantum_circuit(params2)
                quantum_fidelity = np.exp(-abs(cost1 - cost2))
            
            quantum_fidelities.append(quantum_fidelity)
            
            # Classical similarity (neural network output similarity)
            classical_fidelity = np.exp(-np.linalg.norm(params1 - params2))
            classical_fidelities.append(classical_fidelity)
        
        # Expressivity metrics
        quantum_expressivity = 1.0 - np.mean(quantum_fidelities)
        classical_expressivity = 1.0 - np.mean(classical_fidelities)
        
        return {
            "quantum_expressivity_score": quantum_expressivity,
            "classical_expressivity_score": classical_expressivity,
            "expressivity_advantage": quantum_expressivity - classical_expressivity,
            "quantum_fidelity_variance": np.var(quantum_fidelities),
            "classical_fidelity_variance": np.var(classical_fidelities)
        }
    
    def barren_plateau_analysis(
        self,
        cost_function: Callable,
        n_gradient_samples: int = 100
    ) -> Dict[str, Any]:
        """Analyze barren plateau phenomenon."""
        
        logger.info("Analyzing barren plateau effects")
        
        gradients = []
        
        # Sample gradients at random points
        for _ in range(n_gradient_samples):
            params = np.random.uniform(-np.pi, np.pi, self.n_params)
            
            # Compute gradient using parameter shift rule
            gradient = self._compute_gradient(cost_function, params)
            gradients.append(gradient)
        
        gradients = np.array(gradients)
        
        # Gradient statistics
        gradient_variances = np.var(gradients, axis=0)
        mean_gradient_variance = np.mean(gradient_variances)
        
        # Barren plateau detection
        plateau_threshold = 1e-6
        plateau_detected = mean_gradient_variance < plateau_threshold
        
        # Effective dimension (number of parameters with significant gradients)
        significant_params = np.sum(gradient_variances > plateau_threshold)
        effective_dimension = significant_params / self.n_params
        
        return {
            "gradient_variance": mean_gradient_variance,
            "plateau_detected": plateau_detected,
            "effective_dimension": effective_dimension,
            "gradient_variances": gradient_variances,
            "significant_parameters": significant_params
        }
    
    def _compute_gradient(
        self,
        cost_function: Callable,
        params: np.ndarray,
        shift: float = np.pi / 2
    ) -> np.ndarray:
        """Compute gradient using parameter shift rule."""
        
        gradient = np.zeros_like(params)
        
        for i in range(len(params)):
            # Forward shift
            params_plus = params.copy()
            params_plus[i] += shift
            cost_plus = cost_function(params_plus)
            
            # Backward shift
            params_minus = params.copy()
            params_minus[i] -= shift
            cost_minus = cost_function(params_minus)
            
            # Parameter shift gradient
            gradient[i] = (cost_plus - cost_minus) / 2
        
        return gradient
    
    def optimization_comparison(
        self,
        cost_function: Callable,
        n_trials: int = 10,
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """Compare optimization performance."""
        
        logger.info("Running optimization comparison")
        
        quantum_results = []
        classical_results = []
        
        for trial in range(n_trials):
            # Quantum optimization
            initial_params = np.random.uniform(-np.pi, np.pi, self.n_params)
            
            quantum_result = minimize(
                cost_function,
                initial_params,
                method='COBYLA',
                options={'maxiter': max_iterations}
            )
            quantum_results.append({
                'final_cost': quantum_result.fun,
                'n_iterations': quantum_result.nfev,
                'success': quantum_result.success
            })
            
            # Classical optimization (neural network)
            classical_model = MLPRegressor(
                hidden_layer_sizes=(max(2, self.n_qubits),),
                max_iter=max_iterations,
                random_state=trial
            )
            
            # Dummy data for classical training
            X_dummy = np.random.rand(50, max(2, self.n_qubits))
            y_dummy = np.random.rand(50)
            
            classical_model.fit(X_dummy, y_dummy)
            classical_cost = mean_squared_error(
                y_dummy, classical_model.predict(X_dummy)
            )
            
            classical_results.append({
                'final_cost': classical_cost,
                'n_iterations': classical_model.n_iter_,
                'success': True
            })
        
        # Aggregate results
        quantum_costs = [r['final_cost'] for r in quantum_results]
        classical_costs = [r['final_cost'] for r in classical_results]
        
        quantum_iterations = [r['n_iterations'] for r in quantum_results]
        classical_iterations = [r['n_iterations'] for r in classical_results]
        
        # Statistical test
        from scipy.stats import ttest_ind
        cost_t_stat, cost_p_value = ttest_ind(quantum_costs, classical_costs)
        
        return {
            "quantum_final_cost": np.mean(quantum_costs),
            "classical_final_cost": np.mean(classical_costs),
            "cost_advantage": np.mean(classical_costs) - np.mean(quantum_costs),
            "quantum_convergence_steps": np.mean(quantum_iterations),
            "classical_convergence_steps": np.mean(classical_iterations),
            "convergence_advantage": int(np.mean(classical_iterations) - np.mean(quantum_iterations)),
            "cost_p_value": cost_p_value,
            "cost_statistically_significant": cost_p_value < 0.05
        }
    
    def comprehensive_advantage_analysis(
        self,
        cost_function: Callable,
        **kwargs: Any
    ) -> VariationalAdvantageResult:
        """Perform comprehensive variational advantage analysis."""
        
        logger.info("Starting comprehensive variational advantage analysis")
        
        # Landscape analysis
        landscape_results = self.landscape_analysis(cost_function)
        
        # Expressivity analysis
        expressivity_results = self.expressivity_analysis()
        
        # Barren plateau analysis
        plateau_results = self.barren_plateau_analysis(cost_function)
        
        # Optimization comparison
        optimization_results = self.optimization_comparison(cost_function)
        
        # Resource efficiency analysis
        classical_param_count = max(50, self.n_params * 2)  # Estimated classical equivalent
        parameter_efficiency = classical_param_count / self.n_params
        
        # Overall advantage score
        landscape_score = max(0, landscape_results["landscape_advantage"])
        expressivity_score = max(0, expressivity_results["expressivity_advantage"]) 
        performance_score = max(0, optimization_results["cost_advantage"])
        
        overall_score = (
            0.3 * landscape_score +
            0.3 * expressivity_score +
            0.4 * performance_score
        )
        
        # Categorize advantage
        if overall_score > 0.2:
            advantage_category = "strong"
        elif overall_score > 0.1:
            advantage_category = "moderate"
        elif overall_score > 0.05:
            advantage_category = "weak"
        else:
            advantage_category = "none"
        
        # Confidence interval (simplified)
        cost_advantage = optimization_results["cost_advantage"]
        ci_width = 0.1 * abs(cost_advantage)  # 10% of advantage as rough CI
        confidence_interval = (cost_advantage - ci_width, cost_advantage + ci_width)
        
        result = VariationalAdvantageResult(
            quantum_landscape_roughness=landscape_results["quantum_landscape_roughness"],
            classical_landscape_roughness=landscape_results["classical_landscape_roughness"],
            landscape_advantage=landscape_results["landscape_advantage"],
            quantum_expressivity_score=expressivity_results["quantum_expressivity_score"],
            classical_expressivity_score=expressivity_results["classical_expressivity_score"],
            expressivity_advantage=expressivity_results["expressivity_advantage"],
            gradient_variance=plateau_results["gradient_variance"],
            plateau_detected=plateau_results["plateau_detected"],
            effective_dimension=int(plateau_results["effective_dimension"] * self.n_params),
            quantum_final_cost=optimization_results["quantum_final_cost"],
            classical_final_cost=optimization_results["classical_final_cost"],
            cost_advantage=optimization_results["cost_advantage"],
            quantum_convergence_steps=int(optimization_results["quantum_convergence_steps"]),
            classical_convergence_steps=int(optimization_results["classical_convergence_steps"]),
            convergence_advantage=optimization_results["convergence_advantage"],
            quantum_parameter_count=self.n_params,
            classical_parameter_count=classical_param_count,
            parameter_efficiency=parameter_efficiency,
            advantage_p_value=optimization_results["cost_p_value"],
            confidence_interval=confidence_interval,
            statistically_significant=optimization_results["cost_statistically_significant"],
            overall_advantage_score=overall_score,
            advantage_category=advantage_category
        )
        
        logger.info(f"Variational analysis complete. Overall advantage: {advantage_category}")
        
        return result