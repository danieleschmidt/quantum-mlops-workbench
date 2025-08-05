"""Hyperparameter optimization for quantum machine learning models."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import time
import logging
import json
from pathlib import Path

import numpy as np

from .core import QuantumMLPipeline, QuantumDevice, QuantumModel
from .exceptions import QuantumMLOpsException

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Results of hyperparameter optimization."""
    
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    optimization_time: float
    trial_history: List[Dict[str, Any]]
    hardware_budget_used: int
    convergence_achieved: bool


class HyperparameterOptimizer(ABC):
    """Base class for hyperparameter optimizers."""
    
    @abstractmethod
    def optimize(
        self,
        objective_function: Callable,
        search_space: Dict[str, Any],
        n_trials: int,
        **kwargs: Any
    ) -> OptimizationResult:
        """Optimize hyperparameters.
        
        Args:
            objective_function: Function to minimize
            search_space: Search space definition
            n_trials: Number of optimization trials
            **kwargs: Additional optimizer arguments
            
        Returns:
            Optimization results
        """
        pass


class RandomSearchOptimizer(HyperparameterOptimizer):
    """Random search hyperparameter optimizer."""
    
    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize random search optimizer.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def optimize(
        self,
        objective_function: Callable,
        search_space: Dict[str, Any],
        n_trials: int,
        **kwargs: Any
    ) -> OptimizationResult:
        """Perform random search optimization."""
        start_time = time.time()
        
        best_params = None
        best_value = float('inf')
        trial_history = []
        hardware_budget_used = 0
        
        for trial in range(n_trials):
            # Sample random parameters
            params = self._sample_random_params(search_space)
            
            try:
                # Evaluate objective function
                value = objective_function(params)
                hardware_budget_used += kwargs.get('shots_per_trial', 1000)
                
                # Update best result
                if value < best_value:
                    best_value = value
                    best_params = params.copy()
                
                trial_history.append({
                    'trial': trial,
                    'params': params,
                    'value': value,
                    'is_best': value == best_value
                })
                
                logger.info(f"Trial {trial}: value={value:.6f}, best={best_value:.6f}")
                
            except Exception as e:
                logger.warning(f"Trial {trial} failed: {e}")
                trial_history.append({
                    'trial': trial,
                    'params': params,
                    'value': float('inf'),
                    'error': str(e),
                    'is_best': False
                })
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=best_params or {},
            best_value=best_value,
            n_trials=n_trials,
            optimization_time=optimization_time,
            trial_history=trial_history,
            hardware_budget_used=hardware_budget_used,
            convergence_achieved=self._check_convergence(trial_history)
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
                if param_config.get('type') == 'int':
                    # Integer parameter
                    low = param_config['low']
                    high = param_config['high']
                    params[param_name] = np.random.randint(low, high + 1)
                elif param_config.get('type') == 'log':
                    # Log-uniform parameter
                    low = param_config['low']
                    high = param_config['high']
                    params[param_name] = np.exp(np.random.uniform(np.log(low), np.log(high)))
            else:
                # Fixed parameter
                params[param_name] = param_config
        
        return params
    
    def _check_convergence(self, trial_history: List[Dict[str, Any]]) -> bool:
        """Check if optimization has converged."""
        if len(trial_history) < 10:
            return False
        
        # Check if best value hasn't improved in last 10 trials
        recent_best_values = [trial['value'] for trial in trial_history[-10:] if trial.get('is_best', False)]
        return len(recent_best_values) == 0


class OptunaOptimizer(HyperparameterOptimizer):
    """Optuna-based hyperparameter optimizer."""
    
    def __init__(
        self,
        sampler_name: str = "TPE",
        pruner_name: str = "MedianPruner",
        seed: Optional[int] = None
    ) -> None:
        """Initialize Optuna optimizer.
        
        Args:
            sampler_name: Optuna sampler ("TPE", "RandomSampler", "CmaEsSampler")
            pruner_name: Optuna pruner ("MedianPruner", "SuccessiveHalvingPruner")
            seed: Random seed for reproducibility
        """
        if not OPTUNA_AVAILABLE:
            raise QuantumMLOpsException("Optuna not available. Install with: pip install optuna")
        
        self.sampler_name = sampler_name
        self.pruner_name = pruner_name
        self.seed = seed
    
    def optimize(
        self,
        objective_function: Callable,
        search_space: Dict[str, Any],
        n_trials: int,
        **kwargs: Any
    ) -> OptimizationResult:
        """Perform Optuna-based optimization."""
        start_time = time.time()
        
        # Create Optuna study
        sampler = self._create_sampler()
        pruner = self._create_pruner()
        
        study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            pruner=pruner
        )
        
        # Track hardware budget
        hardware_budget_used = 0
        
        def optuna_objective(trial: optuna.Trial) -> float:
            nonlocal hardware_budget_used
            
            # Sample parameters from search space
            params = self._sample_optuna_params(trial, search_space)
            
            # Evaluate objective function
            value = objective_function(params)
            hardware_budget_used += kwargs.get('shots_per_trial', 1000)
            
            return value
        
        # Run optimization
        study.optimize(optuna_objective, n_trials=n_trials)
        
        optimization_time = time.time() - start_time
        
        # Extract trial history
        trial_history = []
        for trial in study.trials:
            trial_history.append({
                'trial': trial.number,
                'params': trial.params,
                'value': trial.value if trial.value is not None else float('inf'),
                'state': trial.state.name,
                'is_best': trial == study.best_trial
            })
        
        return OptimizationResult(
            best_params=study.best_params,
            best_value=study.best_value,
            n_trials=len(study.trials),
            optimization_time=optimization_time,
            trial_history=trial_history,
            hardware_budget_used=hardware_budget_used,
            convergence_achieved=self._check_optuna_convergence(study)
        )
    
    def _create_sampler(self):
        """Create Optuna sampler."""
        if not OPTUNA_AVAILABLE or optuna is None:
            return None
        
        if self.sampler_name == "TPE":
            return optuna.samplers.TPESampler(seed=self.seed)
        elif self.sampler_name == "RandomSampler":
            return optuna.samplers.RandomSampler(seed=self.seed)
        elif self.sampler_name == "CmaEsSampler":
            return optuna.samplers.CmaEsSampler(seed=self.seed)
        else:
            return optuna.samplers.TPESampler(seed=self.seed)
    
    def _create_pruner(self):
        """Create Optuna pruner."""
        if not OPTUNA_AVAILABLE or optuna is None:
            return None
            
        if self.pruner_name == "MedianPruner":
            return optuna.pruners.MedianPruner()
        elif self.pruner_name == "SuccessiveHalvingPruner":
            return optuna.pruners.SuccessiveHalvingPruner()
        else:
            return optuna.pruners.MedianPruner()
    
    def _sample_optuna_params(self, trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample parameters using Optuna trial."""
        params = {}
        
        for param_name, param_config in search_space.items():
            if isinstance(param_config, list):
                # Categorical parameter
                params[param_name] = trial.suggest_categorical(param_name, param_config)
            elif isinstance(param_config, tuple) and len(param_config) == 2:
                # Continuous parameter
                low, high = param_config
                params[param_name] = trial.suggest_float(param_name, low, high)
            elif isinstance(param_config, dict):
                if param_config.get('type') == 'int':
                    # Integer parameter
                    params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
                elif param_config.get('type') == 'log':
                    # Log-uniform parameter
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)
            else:
                # Fixed parameter
                params[param_name] = param_config
        
        return params
    
    def _check_optuna_convergence(self, study) -> bool:
        """Check if Optuna optimization has converged."""
        if len(study.trials) < 20:
            return False
        
        # Check if best value hasn't improved significantly in recent trials
        recent_trials = study.trials[-10:]
        recent_values = [trial.value for trial in recent_trials if trial.value is not None]
        
        if not recent_values:
            return False
        
        best_recent = min(recent_values)
        return abs(study.best_value - best_recent) < 1e-6


class QuantumHyperOpt:
    """Quantum-aware hyperparameter optimization for quantum ML models."""
    
    def __init__(
        self,
        search_space: Dict[str, Any],
        optimization_backend: str = "random",
        hardware_budget: int = 10000,
        seed: Optional[int] = None
    ) -> None:
        """Initialize quantum hyperparameter optimizer.
        
        Args:
            search_space: Hyperparameter search space
            optimization_backend: Optimization backend ("random", "optuna")
            hardware_budget: Maximum quantum hardware shots
            seed: Random seed for reproducibility
        """
        self.search_space = search_space
        self.hardware_budget = hardware_budget
        self.seed = seed
        
        # Create optimizer
        if optimization_backend == "optuna":
            self.optimizer = OptunaOptimizer(seed=seed)
        else:
            self.optimizer = RandomSearchOptimizer(seed=seed)
        
        self.optimization_results: Optional[OptimizationResult] = None
    
    def optimize(
        self,
        train_fn: Callable,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        n_trials: int = 50,
        shots_per_trial: int = 1000,
        device: QuantumDevice = QuantumDevice.SIMULATOR,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Optimize hyperparameters for quantum ML model.
        
        Args:
            train_fn: Training function that takes (X, y, **params) and returns trained model
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (uses train if None)
            y_val: Validation labels (uses train if None)
            n_trials: Number of optimization trials
            shots_per_trial: Quantum shots per trial
            device: Quantum backend device
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing best parameters and optimization results
        """
        # Use training data for validation if no validation set provided
        if X_val is None or y_val is None:
            X_val, y_val = X_train, y_train
        
        # Track budget usage
        budget_used = 0
        
        def objective_function(params: Dict[str, Any]) -> float:
            nonlocal budget_used
            
            if budget_used >= self.hardware_budget:
                raise QuantumMLOpsException("Hardware budget exceeded")
            
            try:
                # Create pipeline with sampled parameters
                pipeline = QuantumMLPipeline(
                    circuit=self._create_parameterized_circuit(params),
                    n_qubits=params.get('n_qubits', 4),
                    device=device,
                    **{k: v for k, v in params.items() if k not in ['n_qubits', 'learning_rate', 'epochs']}
                )
                
                # Train model
                model = pipeline.train(
                    X_train,
                    y_train,
                    epochs=params.get('epochs', 50),
                    learning_rate=params.get('learning_rate', 0.01),
                    track_gradients=True
                )
                
                # Evaluate on validation set
                metrics = pipeline.evaluate(model, X_val, y_val)
                
                # Update budget
                budget_used += shots_per_trial
                
                # Return negative accuracy (since we minimize)
                return -metrics.accuracy
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float('inf')
        
        # Run optimization
        self.optimization_results = self.optimizer.optimize(
            objective_function=objective_function,
            search_space=self.search_space,
            n_trials=n_trials,
            shots_per_trial=shots_per_trial
        )
        
        # Train final model with best parameters
        best_params = self.optimization_results.best_params
        final_pipeline = QuantumMLPipeline(
            circuit=self._create_parameterized_circuit(best_params),
            n_qubits=best_params.get('n_qubits', 4),
            device=device,
            **{k: v for k, v in best_params.items() if k not in ['n_qubits', 'learning_rate', 'epochs']}
        )
        
        final_model = final_pipeline.train(
            X_train,
            y_train,
            epochs=best_params.get('epochs', 50),
            learning_rate=best_params.get('learning_rate', 0.01),
            track_gradients=True
        )
        
        return {
            'best_params': best_params,
            'best_accuracy': -self.optimization_results.best_value,
            'final_model': final_model,
            'final_pipeline': final_pipeline,
            'optimization_results': self.optimization_results,
            'hardware_budget_used': budget_used
        }
    
    def _create_parameterized_circuit(self, params: Dict[str, Any]) -> Callable:
        """Create parameterized quantum circuit based on hyperparameters."""
        n_layers = params.get('n_layers', 3)
        entanglement = params.get('entanglement', 'linear')
        
        def quantum_circuit(circuit_params: np.ndarray, x: np.ndarray) -> float:
            """Parameterized quantum circuit."""
            # This is a placeholder - in practice would use PennyLane/Qiskit
            # Apply data encoding
            encoded_data = np.sum(x * circuit_params[:len(x)])
            
            # Apply parameterized layers
            for layer in range(n_layers):
                layer_params = circuit_params[layer * len(x):(layer + 1) * len(x)]
                encoded_data += np.sum(layer_params) * (layer + 1) * 0.1
            
            # Apply entanglement (simplified)
            if entanglement == 'full':
                encoded_data *= 1.1
            elif entanglement == 'circular':
                encoded_data *= 1.05
            
            return np.tanh(encoded_data)
        
        return quantum_circuit
    
    def save_results(self, filepath: str) -> None:
        """Save optimization results to file.
        
        Args:
            filepath: Path to save results
        """
        if not self.optimization_results:
            raise ValueError("No optimization results to save")
        
        results_dict = {
            'best_params': self.optimization_results.best_params,
            'best_value': self.optimization_results.best_value,
            'n_trials': self.optimization_results.n_trials,
            'optimization_time': self.optimization_results.optimization_time,
            'hardware_budget_used': self.optimization_results.hardware_budget_used,
            'convergence_achieved': self.optimization_results.convergence_achieved,
            'trial_history': self.optimization_results.trial_history,
            'search_space': self.search_space
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
    
    def load_results(self, filepath: str) -> None:
        """Load optimization results from file.
        
        Args:
            filepath: Path to load results from
        """
        with open(filepath, 'r') as f:
            results_dict = json.load(f)
        
        self.optimization_results = OptimizationResult(
            best_params=results_dict['best_params'],
            best_value=results_dict['best_value'],
            n_trials=results_dict['n_trials'],
            optimization_time=results_dict['optimization_time'],
            trial_history=results_dict['trial_history'],
            hardware_budget_used=results_dict['hardware_budget_used'],
            convergence_achieved=results_dict['convergence_achieved']
        )
        
        self.search_space = results_dict.get('search_space', {})
    
    def plot_optimization_history(self) -> None:
        """Plot optimization history."""
        if not self.optimization_results:
            raise ValueError("No optimization results to plot")
        
        try:
            import matplotlib.pyplot as plt
            
            trials = [trial['trial'] for trial in self.optimization_results.trial_history]
            values = [trial['value'] for trial in self.optimization_results.trial_history]
            
            # Plot trial values
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.plot(trials, values, 'b-', alpha=0.7, label='Trial values')
            
            # Plot best values
            best_values = []
            current_best = float('inf')
            for value in values:
                if value < current_best:
                    current_best = value
                best_values.append(current_best)
            
            plt.plot(trials, best_values, 'r-', linewidth=2, label='Best value')
            plt.xlabel('Trial')
            plt.ylabel('Objective Value')
            plt.title('Optimization Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot parameter importance (if available)
            plt.subplot(1, 2, 2)
            param_names = list(self.optimization_results.best_params.keys())
            param_values = list(self.optimization_results.best_params.values())
            
            plt.barh(param_names, [abs(float(v)) if isinstance(v, (int, float)) else 1 for v in param_values])
            plt.xlabel('Parameter Value (abs)')
            plt.title('Best Parameters')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results.
        
        Returns:
            Dictionary containing optimization summary
        """
        if not self.optimization_results:
            return {}
        
        return {
            'best_params': self.optimization_results.best_params,
            'best_accuracy': -self.optimization_results.best_value,
            'n_trials': self.optimization_results.n_trials,
            'optimization_time': f"{self.optimization_results.optimization_time:.2f}s",
            'hardware_budget_used': self.optimization_results.hardware_budget_used,
            'budget_utilization': f"{self.optimization_results.hardware_budget_used / self.hardware_budget * 100:.1f}%",
            'convergence_achieved': self.optimization_results.convergence_achieved,
            'improvement_rate': len([t for t in self.optimization_results.trial_history if t.get('is_best', False)]) / self.optimization_results.n_trials
        }