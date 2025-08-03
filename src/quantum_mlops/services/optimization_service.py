"""Quantum ML optimization and hyperparameter tuning service."""

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
from ..core import QuantumMLPipeline, QuantumModel, QuantumMetrics


logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    all_trials: List[Dict[str, Any]]
    optimization_time: float
    n_trials: int


class QuantumOptimizer:
    """Base class for quantum-aware optimizers."""
    
    def __init__(self, objective_function: Callable):
        self.objective_function = objective_function
        self.trial_history: List[Dict[str, Any]] = []
    
    def optimize(
        self,
        search_space: Dict[str, Any],
        n_trials: int = 50,
        **kwargs
    ) -> OptimizationResult:
        """Run optimization process."""
        raise NotImplementedError


class BayesianQuantumOptimizer(QuantumOptimizer):
    """Bayesian optimization for quantum ML hyperparameters."""
    
    def optimize(
        self,
        search_space: Dict[str, Any],
        n_trials: int = 50,
        acquisition_function: str = "expected_improvement",
        **kwargs
    ) -> OptimizationResult:
        """Run Bayesian optimization."""
        start_time = time.time()
        best_score = float('-inf')
        best_params = None
        
        # Initialize with random trials
        n_random = min(5, n_trials // 4)
        
        for trial in range(n_trials):
            if trial < n_random:
                # Random exploration
                params = self._sample_random_params(search_space)
            else:
                # Bayesian acquisition
                params = self._acquire_next_params(search_space, acquisition_function)
            
            # Evaluate objective
            score = self._evaluate_params(params, trial)
            
            if score > best_score:
                best_score = score
                best_params = params
            
            logger.info(f"Trial {trial+1}/{n_trials}: Score={score:.4f}, Best={best_score:.4f}")
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_trials=self.trial_history.copy(),
            optimization_time=optimization_time,
            n_trials=n_trials
        )
    
    def _sample_random_params(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample random parameters from search space."""
        params = {}
        
        for param_name, param_config in search_space.items():
            if isinstance(param_config, list):
                # Categorical parameter
                params[param_name] = np.random.choice(param_config)
            elif isinstance(param_config, tuple) and len(param_config) == 2:
                # Continuous parameter (min, max)
                low, high = param_config
                params[param_name] = np.random.uniform(low, high)
            elif isinstance(param_config, dict):
                # Complex parameter specification
                param_type = param_config.get("type", "uniform")
                
                if param_type == "uniform":
                    params[param_name] = np.random.uniform(
                        param_config["low"], param_config["high"]
                    )
                elif param_type == "log_uniform":
                    params[param_name] = np.exp(np.random.uniform(
                        np.log(param_config["low"]), np.log(param_config["high"])
                    ))
                elif param_type == "choice":
                    params[param_name] = np.random.choice(param_config["choices"])
                elif param_type == "int":
                    params[param_name] = np.random.randint(
                        param_config["low"], param_config["high"] + 1
                    )
        
        return params
    
    def _acquire_next_params(
        self, 
        search_space: Dict[str, Any], 
        acquisition_function: str
    ) -> Dict[str, Any]:
        """Acquire next parameters using Bayesian optimization."""
        # Simplified acquisition function (in production, would use proper GP)
        if len(self.trial_history) < 2:
            return self._sample_random_params(search_space)
        
        # Use information from previous trials
        best_trials = sorted(
            self.trial_history, 
            key=lambda x: x["score"], 
            reverse=True
        )[:3]
        
        # Generate candidate based on best trials with perturbation
        base_params = best_trials[0]["params"].copy()
        
        for param_name, param_config in search_space.items():
            # Add Gaussian noise around best parameter
            if isinstance(param_config, tuple) and len(param_config) == 2:
                low, high = param_config
                noise_scale = (high - low) * 0.1
                base_params[param_name] += np.random.normal(0, noise_scale)
                base_params[param_name] = np.clip(base_params[param_name], low, high)
            elif isinstance(param_config, list):
                # Sometimes mutate categorical parameters
                if np.random.random() < 0.3:
                    base_params[param_name] = np.random.choice(param_config)
        
        return base_params
    
    def _evaluate_params(self, params: Dict[str, Any], trial_idx: int) -> float:
        """Evaluate parameter configuration."""
        start_time = time.time()
        
        try:
            score = self.objective_function(params)
            status = "completed"
            error = None
        except Exception as e:
            score = float('-inf')
            status = "failed"
            error = str(e)
            logger.warning(f"Trial {trial_idx} failed: {error}")
        
        evaluation_time = time.time() - start_time
        
        # Record trial
        trial_record = {
            "trial_id": trial_idx,
            "params": params.copy(),
            "score": score,
            "status": status,
            "error": error,
            "evaluation_time": evaluation_time,
            "timestamp": time.time()
        }
        
        self.trial_history.append(trial_record)
        
        return score


class GridSearchOptimizer(QuantumOptimizer):
    """Grid search for quantum ML hyperparameters."""
    
    def optimize(
        self,
        search_space: Dict[str, Any],
        n_trials: int = 50,
        **kwargs
    ) -> OptimizationResult:
        """Run grid search optimization."""
        start_time = time.time()
        
        # Generate parameter grid
        param_grid = self._generate_parameter_grid(search_space, n_trials)
        
        best_score = float('-inf')
        best_params = None
        
        for trial_idx, params in enumerate(param_grid):
            score = self._evaluate_params(params, trial_idx)
            
            if score > best_score:
                best_score = score
                best_params = params
            
            logger.info(f"Trial {trial_idx+1}/{len(param_grid)}: Score={score:.4f}")
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_trials=self.trial_history.copy(),
            optimization_time=optimization_time,
            n_trials=len(param_grid)
        )
    
    def _generate_parameter_grid(
        self, 
        search_space: Dict[str, Any], 
        max_combinations: int
    ) -> List[Dict[str, Any]]:
        """Generate parameter grid."""
        # Simplified grid generation
        param_names = list(search_space.keys())
        param_values = []
        
        for param_name, param_config in search_space.items():
            if isinstance(param_config, list):
                param_values.append(param_config)
            elif isinstance(param_config, tuple) and len(param_config) == 2:
                low, high = param_config
                n_values = min(5, int(max_combinations ** (1/len(search_space))))
                values = np.linspace(low, high, n_values).tolist()
                param_values.append(values)
            else:
                # Default to single value
                param_values.append([param_config])
        
        # Generate all combinations
        import itertools
        combinations = list(itertools.product(*param_values))
        
        # Limit combinations
        if len(combinations) > max_combinations:
            combinations = combinations[::len(combinations)//max_combinations][:max_combinations]
        
        # Convert to parameter dictionaries
        param_grid = []
        for combination in combinations:
            params = dict(zip(param_names, combination))
            param_grid.append(params)
        
        return param_grid


class OptimizationService:
    """Service for quantum ML optimization tasks."""
    
    def __init__(self):
        self.optimization_history: Dict[str, OptimizationResult] = {}
    
    def optimize_hyperparameters(
        self,
        pipeline: QuantumMLPipeline,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        search_space: Dict[str, Any],
        optimizer_type: str = "bayesian",
        n_trials: int = 50,
        optimization_metric: str = "accuracy"
    ) -> OptimizationResult:
        """Optimize hyperparameters for quantum ML pipeline."""
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        def objective_function(params: Dict[str, Any]) -> float:
            """Objective function for hyperparameter optimization."""
            try:
                # Update pipeline configuration
                for param_name, param_value in params.items():
                    if hasattr(pipeline, param_name):
                        setattr(pipeline, param_name, param_value)
                    else:
                        pipeline.config[param_name] = param_value
                
                # Train model with current parameters
                model = pipeline.train(
                    X_train, y_train,
                    epochs=params.get("epochs", 50),
                    learning_rate=params.get("learning_rate", 0.01)
                )
                
                # Evaluate on validation set
                metrics = pipeline.evaluate(model, X_val, y_val)
                
                # Return optimization metric
                if optimization_metric == "accuracy":
                    return metrics.accuracy
                elif optimization_metric == "loss":
                    return -metrics.loss  # Minimize loss
                elif optimization_metric == "fidelity":
                    return metrics.fidelity
                else:
                    return metrics.accuracy
                
            except Exception as e:
                logger.warning(f"Objective function failed: {e}")
                return float('-inf')
        
        # Select optimizer
        if optimizer_type == "bayesian":
            optimizer = BayesianQuantumOptimizer(objective_function)
        elif optimizer_type == "grid_search":
            optimizer = GridSearchOptimizer(objective_function)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Run optimization
        result = optimizer.optimize(search_space, n_trials)
        
        # Store result
        opt_id = f"opt_{int(time.time())}"
        self.optimization_history[opt_id] = result
        
        logger.info(f"Optimization completed: {result.best_score:.4f} in {result.optimization_time:.2f}s")
        
        return result
    
    def optimize_circuit_parameters(
        self,
        model: QuantumModel,
        train_data: Tuple[np.ndarray, np.ndarray],
        optimizer_method: str = "adam",
        learning_rate: float = 0.01,
        max_iterations: int = 1000
    ) -> QuantumModel:
        """Optimize quantum circuit parameters."""
        
        X_train, y_train = train_data
        
        if model.parameters is None:
            raise ValueError("Model must have initialized parameters")
        
        # Parameter optimization loop
        params = model.parameters.copy()
        best_loss = float('inf')
        patience_counter = 0
        patience = 50
        
        # Adam optimizer state
        if optimizer_method == "adam":
            m = np.zeros_like(params)  # First moment
            v = np.zeros_like(params)  # Second moment
            beta1, beta2 = 0.9, 0.999
            epsilon = 1e-8
        
        for iteration in range(max_iterations):
            # Compute gradients using parameter shift rule
            gradients = self._compute_parameter_gradients(model, params, X_train, y_train)
            
            # Update parameters based on optimizer
            if optimizer_method == "sgd":
                params -= learning_rate * gradients
            elif optimizer_method == "adam":
                m = beta1 * m + (1 - beta1) * gradients
                v = beta2 * v + (1 - beta2) * (gradients ** 2)
                
                m_corrected = m / (1 - beta1 ** (iteration + 1))
                v_corrected = v / (1 - beta2 ** (iteration + 1))
                
                params -= learning_rate * m_corrected / (np.sqrt(v_corrected) + epsilon)
            
            # Update model parameters
            model.parameters = params.copy()
            
            # Compute current loss
            predictions = model.predict(X_train)
            current_loss = np.mean((predictions - y_train) ** 2)
            
            # Early stopping
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at iteration {iteration}")
                    break
            
            if iteration % 100 == 0:
                logger.info(f"Iteration {iteration}: Loss={current_loss:.6f}")
        
        logger.info(f"Parameter optimization completed. Final loss: {best_loss:.6f}")
        
        return model
    
    def _compute_parameter_gradients(
        self,
        model: QuantumModel,
        params: np.ndarray,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """Compute parameter gradients using parameter shift rule."""
        gradients = np.zeros_like(params)
        shift = np.pi / 2
        
        for i in range(len(params)):
            # Forward shift
            params_forward = params.copy()
            params_forward[i] += shift
            model.parameters = params_forward
            pred_forward = model.predict(X)
            loss_forward = np.mean((pred_forward - y) ** 2)
            
            # Backward shift
            params_backward = params.copy()
            params_backward[i] -= shift
            model.parameters = params_backward
            pred_backward = model.predict(X)
            loss_backward = np.mean((pred_backward - y) ** 2)
            
            # Gradient estimate
            gradients[i] = (loss_forward - loss_backward) / 2
        
        # Restore original parameters
        model.parameters = params
        
        return gradients
    
    def quantum_advantage_analysis(
        self,
        quantum_model: QuantumModel,
        classical_model: Any,
        test_data: Tuple[np.ndarray, np.ndarray],
        metrics: List[str] = ["accuracy", "training_time", "inference_time"]
    ) -> Dict[str, Any]:
        """Analyze quantum advantage over classical approaches."""
        
        X_test, y_test = test_data
        results = {"quantum": {}, "classical": {}, "advantage": {}}
        
        # Quantum model evaluation
        start_time = time.time()
        quantum_predictions = quantum_model.predict(X_test)
        quantum_inference_time = time.time() - start_time
        
        results["quantum"]["accuracy"] = np.mean(
            (quantum_predictions > 0.5) == (y_test > 0.5)
        )
        results["quantum"]["inference_time"] = quantum_inference_time
        results["quantum"]["n_qubits"] = quantum_model.n_qubits
        results["quantum"]["circuit_depth"] = quantum_model.circuit_depth
        
        # Classical model evaluation (if provided)
        if hasattr(classical_model, 'predict'):
            start_time = time.time()
            classical_predictions = classical_model.predict(X_test)
            classical_inference_time = time.time() - start_time
            
            results["classical"]["accuracy"] = np.mean(
                (classical_predictions > 0.5) == (y_test > 0.5)
            )
            results["classical"]["inference_time"] = classical_inference_time
            
            # Compute advantages
            results["advantage"]["accuracy_ratio"] = (
                results["quantum"]["accuracy"] / results["classical"]["accuracy"]
            )
            results["advantage"]["speed_ratio"] = (
                classical_inference_time / quantum_inference_time
            )
            results["advantage"]["has_quantum_advantage"] = (
                results["advantage"]["accuracy_ratio"] > 1.0 or
                results["advantage"]["speed_ratio"] > 1.0
            )
        
        return results
    
    def suggest_circuit_optimization(self, model: QuantumModel) -> Dict[str, Any]:
        """Suggest optimizations for quantum circuit."""
        suggestions = {
            "gate_optimizations": [],
            "depth_reductions": [],
            "parameter_suggestions": [],
            "hardware_considerations": []
        }
        
        # Circuit depth analysis
        if model.circuit_depth > 50:
            suggestions["depth_reductions"].append(
                "Consider circuit compression techniques to reduce depth"
            )
        
        # Parameter count analysis
        if model.parameters is not None:
            n_params = len(model.parameters)
            if n_params > model.n_qubits * 10:
                suggestions["parameter_suggestions"].append(
                    "High parameter count may indicate overparameterization"
                )
        
        # Qubit count considerations
        if model.n_qubits > 20:
            suggestions["hardware_considerations"].append(
                "Consider noise-aware optimization for NISQ devices"
            )
        
        # Gate optimization suggestions
        suggestions["gate_optimizations"].extend([
            "Use native gate decomposition for target hardware",
            "Apply gate fusion where possible",
            "Consider approximate synthesis for deep circuits"
        ])
        
        return suggestions