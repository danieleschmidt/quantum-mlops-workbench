"""Quantum ML experiment management and tracking service."""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
from ..core import QuantumMLPipeline, QuantumModel, QuantumMetrics


logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Tracker for quantum ML experiments."""
    
    def __init__(self, experiment_dir: str = "./experiments"):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True)
        self.current_experiment: Optional[Dict[str, Any]] = None
        self.current_run: Optional[Dict[str, Any]] = None
    
    def create_experiment(
        self,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> str:
        """Create a new experiment."""
        experiment_id = f"exp_{int(time.time())}_{name.replace(' ', '_')}"
        experiment_path = self.experiment_dir / experiment_id
        experiment_path.mkdir(exist_ok=True)
        
        experiment_metadata = {
            "experiment_id": experiment_id,
            "name": name,
            "description": description,
            "tags": tags or [],
            "created_at": datetime.utcnow().isoformat(),
            "runs": [],
            "status": "active",
            "path": str(experiment_path)
        }
        
        # Save experiment metadata
        with open(experiment_path / "experiment.json", 'w') as f:
            json.dump(experiment_metadata, f, indent=2)
        
        self.current_experiment = experiment_metadata
        logger.info(f"Created experiment: {experiment_id}")
        
        return experiment_id
    
    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Load existing experiment."""
        experiment_path = self.experiment_dir / experiment_id
        
        if not experiment_path.exists():
            raise ValueError(f"Experiment {experiment_id} not found")
        
        with open(experiment_path / "experiment.json", 'r') as f:
            experiment_metadata = json.load(f)
        
        self.current_experiment = experiment_metadata
        return experiment_metadata
    
    def start_run(
        self,
        run_name: str = "",
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a new run within current experiment."""
        if self.current_experiment is None:
            raise ValueError("No active experiment. Create or load an experiment first.")
        
        run_id = f"run_{int(time.time())}_{len(self.current_experiment['runs'])}"
        run_name = run_name or f"Run {len(self.current_experiment['runs']) + 1}"
        
        run_metadata = {
            "run_id": run_id,
            "name": run_name,
            "experiment_id": self.current_experiment["experiment_id"],
            "config": config or {},
            "started_at": datetime.utcnow().isoformat(),
            "status": "running",
            "metrics": {},
            "parameters": {},
            "artifacts": [],
            "logs": []
        }
        
        # Create run directory
        run_path = Path(self.current_experiment["path"]) / run_id
        run_path.mkdir(exist_ok=True)
        
        # Save run metadata
        with open(run_path / "run.json", 'w') as f:
            json.dump(run_metadata, f, indent=2)
        
        self.current_run = run_metadata
        self.current_experiment["runs"].append(run_id)
        
        # Update experiment metadata
        exp_path = Path(self.current_experiment["path"])
        with open(exp_path / "experiment.json", 'w') as f:
            json.dump(self.current_experiment, f, indent=2)
        
        logger.info(f"Started run: {run_id}")
        return run_id
    
    def log_parameter(self, key: str, value: Any) -> None:
        """Log a parameter for the current run."""
        if self.current_run is None:
            raise ValueError("No active run. Start a run first.")
        
        self.current_run["parameters"][key] = value
        self._save_current_run()
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric for the current run."""
        if self.current_run is None:
            raise ValueError("No active run. Start a run first.")
        
        if key not in self.current_run["metrics"]:
            self.current_run["metrics"][key] = []
        
        metric_entry = {
            "value": value,
            "timestamp": time.time(),
            "step": step
        }
        
        self.current_run["metrics"][key].append(metric_entry)
        self._save_current_run()
    
    def log_quantum_metrics(self, metrics: QuantumMetrics) -> None:
        """Log quantum-specific metrics."""
        self.log_metric("accuracy", metrics.accuracy)
        self.log_metric("loss", metrics.loss)
        self.log_metric("gradient_variance", metrics.gradient_variance)
        self.log_metric("fidelity", metrics.fidelity)
        
        # Log noise analysis if available
        if metrics.noise_analysis:
            for noise_model, results in metrics.noise_analysis.items():
                self.log_metric(f"noise_{noise_model}_accuracy", results["accuracy"])
                self.log_metric(f"noise_{noise_model}_degradation", results["degradation"])
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "file") -> None:
        """Log an artifact for the current run."""
        if self.current_run is None:
            raise ValueError("No active run. Start a run first.")
        
        artifact_entry = {
            "path": artifact_path,
            "type": artifact_type,
            "logged_at": datetime.utcnow().isoformat()
        }
        
        self.current_run["artifacts"].append(artifact_entry)
        self._save_current_run()
    
    def log_model(self, model: QuantumModel, model_name: str = "model") -> str:
        """Log a quantum model as an artifact."""
        if self.current_run is None:
            raise ValueError("No active run. Start a run first.")
        
        run_path = Path(self.current_experiment["path"]) / self.current_run["run_id"]
        model_path = run_path / f"{model_name}.json"
        
        # Save model
        model.save_model(str(model_path))
        
        # Log as artifact
        self.log_artifact(str(model_path), "quantum_model")
        
        return str(model_path)
    
    def end_run(self, status: str = "completed") -> None:
        """End the current run."""
        if self.current_run is None:
            raise ValueError("No active run to end.")
        
        self.current_run["status"] = status
        self.current_run["ended_at"] = datetime.utcnow().isoformat()
        
        if "started_at" in self.current_run:
            start_time = datetime.fromisoformat(self.current_run["started_at"].replace('Z', '+00:00'))
            end_time = datetime.utcnow()
            duration = (end_time - start_time.replace(tzinfo=None)).total_seconds()
            self.current_run["duration"] = duration
        
        self._save_current_run()
        
        logger.info(f"Ended run: {self.current_run['run_id']} with status: {status}")
        self.current_run = None
    
    def _save_current_run(self) -> None:
        """Save current run metadata."""
        if self.current_run is None:
            return
        
        run_path = Path(self.current_experiment["path"]) / self.current_run["run_id"]
        with open(run_path / "run.json", 'w') as f:
            json.dump(self.current_run, f, indent=2)
    
    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """Get summary of experiment runs."""
        experiment = self.load_experiment(experiment_id)
        
        summary = {
            "experiment_id": experiment_id,
            "name": experiment["name"],
            "total_runs": len(experiment["runs"]),
            "runs": []
        }
        
        for run_id in experiment["runs"]:
            run_path = Path(experiment["path"]) / run_id / "run.json"
            
            if run_path.exists():
                with open(run_path, 'r') as f:
                    run_data = json.load(f)
                
                run_summary = {
                    "run_id": run_id,
                    "name": run_data["name"],
                    "status": run_data["status"],
                    "duration": run_data.get("duration", 0),
                    "best_accuracy": self._get_best_metric(run_data["metrics"], "accuracy"),
                    "final_loss": self._get_latest_metric(run_data["metrics"], "loss")
                }
                
                summary["runs"].append(run_summary)
        
        return summary
    
    def _get_best_metric(self, metrics: Dict[str, List], metric_name: str) -> Optional[float]:
        """Get best value for a metric."""
        if metric_name not in metrics or not metrics[metric_name]:
            return None
        
        values = [entry["value"] for entry in metrics[metric_name]]
        return max(values) if values else None
    
    def _get_latest_metric(self, metrics: Dict[str, List], metric_name: str) -> Optional[float]:
        """Get latest value for a metric."""
        if metric_name not in metrics or not metrics[metric_name]:
            return None
        
        return metrics[metric_name][-1]["value"]


class ExperimentService:
    """Service for managing quantum ML experiments."""
    
    def __init__(self, experiment_dir: str = "./experiments"):
        self.tracker = ExperimentTracker(experiment_dir)
        self.mlflow_enabled = self._check_mlflow_availability()
    
    def _check_mlflow_availability(self) -> bool:
        """Check if MLflow is available and configured."""
        try:
            import mlflow
            return True
        except ImportError:
            logger.warning("MLflow not available. Using local tracking only.")
            return False
    
    def run_experiment(
        self,
        experiment_name: str,
        pipeline: QuantumMLPipeline,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        config: Optional[Dict[str, Any]] = None,
        description: str = ""
    ) -> Dict[str, Any]:
        """Run a complete quantum ML experiment."""
        
        # Create experiment
        experiment_id = self.tracker.create_experiment(experiment_name, description)
        
        # Start run
        run_id = self.tracker.start_run("Main Run", config)
        
        try:
            # Log configuration
            if config:
                for key, value in config.items():
                    self.tracker.log_parameter(key, value)
            
            # Log pipeline configuration
            self.tracker.log_parameter("n_qubits", pipeline.n_qubits)
            self.tracker.log_parameter("device", pipeline.device.value)
            self.tracker.log_parameter("backend_config", pipeline.backend_config)
            
            X_train, y_train = train_data
            X_val, y_val = val_data
            
            # Training phase
            logger.info("Starting training phase...")
            training_config = config.get("training", {}) if config else {}
            
            model = pipeline.train(
                X_train, y_train,
                epochs=training_config.get("epochs", 100),
                learning_rate=training_config.get("learning_rate", 0.01),
                track_gradients=True
            )
            
            # Log training metrics
            if hasattr(model, 'training_history') and model.training_history:
                loss_history = model.training_history.get('loss_history', [])
                for epoch, loss in enumerate(loss_history):
                    self.tracker.log_metric("training_loss", loss, step=epoch)
                
                if 'gradient_variances' in model.training_history:
                    grad_vars = model.training_history['gradient_variances']
                    for epoch, grad_var in enumerate(grad_vars):
                        self.tracker.log_metric("gradient_variance", grad_var, step=epoch)
            
            # Validation phase
            logger.info("Starting validation phase...")
            val_metrics = pipeline.evaluate(model, X_val, y_val)
            self.tracker.log_quantum_metrics(val_metrics)
            
            # Test phase (if test data provided)
            test_metrics = None
            if test_data is not None:
                logger.info("Starting test phase...")
                X_test, y_test = test_data
                test_metrics = pipeline.evaluate(
                    model, X_test, y_test,
                    noise_models=config.get("noise_models", []) if config else []
                )
                
                # Log test metrics with prefix
                self.tracker.log_metric("test_accuracy", test_metrics.accuracy)
                self.tracker.log_metric("test_loss", test_metrics.loss)
                self.tracker.log_metric("test_fidelity", test_metrics.fidelity)
            
            # Save model
            model_path = self.tracker.log_model(model, "trained_model")
            
            # Log model metadata
            self.tracker.log_parameter("model_path", model_path)
            self.tracker.log_parameter("circuit_depth", model.circuit_depth)
            if model.parameters is not None:
                self.tracker.log_parameter("parameter_count", len(model.parameters))
            
            # End run successfully
            self.tracker.end_run("completed")
            
            # Prepare results
            results = {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "model": model,
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics,
                "model_path": model_path,
                "status": "completed"
            }
            
            logger.info(f"Experiment completed successfully: {experiment_id}")
            
            return results
            
        except Exception as e:
            # End run with failure
            self.tracker.end_run("failed")
            logger.error(f"Experiment failed: {str(e)}")
            
            return {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "status": "failed",
                "error": str(e)
            }
    
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metric: str = "accuracy"
    ) -> Dict[str, Any]:
        """Compare results across multiple experiments."""
        
        comparison_data = {
            "experiments": [],
            "best_experiment": None,
            "metric_used": metric
        }
        
        best_score = float('-inf')
        best_experiment = None
        
        for exp_id in experiment_ids:
            try:
                summary = self.tracker.get_experiment_summary(exp_id)
                
                # Get best run from experiment
                best_run = None
                best_run_score = float('-inf')
                
                for run in summary["runs"]:
                    if metric in ["accuracy", "fidelity"] and run.get(f"best_{metric}"):
                        score = run[f"best_{metric}"]
                    elif metric == "loss" and run.get("final_loss"):
                        score = -run["final_loss"]  # Lower loss is better
                    else:
                        continue
                    
                    if score > best_run_score:
                        best_run_score = score
                        best_run = run
                
                exp_data = {
                    "experiment_id": exp_id,
                    "name": summary["name"],
                    "total_runs": summary["total_runs"],
                    "best_run": best_run,
                    "best_score": best_run_score
                }
                
                comparison_data["experiments"].append(exp_data)
                
                if best_run_score > best_score:
                    best_score = best_run_score
                    best_experiment = exp_data
            
            except Exception as e:
                logger.warning(f"Could not load experiment {exp_id}: {e}")
        
        comparison_data["best_experiment"] = best_experiment
        
        return comparison_data
    
    def generate_experiment_report(self, experiment_id: str) -> str:
        """Generate a detailed experiment report."""
        
        summary = self.tracker.get_experiment_summary(experiment_id)
        
        report_lines = [
            f"# Quantum ML Experiment Report",
            f"",
            f"**Experiment ID:** {experiment_id}",
            f"**Name:** {summary['name']}",
            f"**Total Runs:** {summary['total_runs']}",
            f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"",
            f"## Run Summary",
            f""
        ]
        
        # Add run details
        for i, run in enumerate(summary['runs']):
            report_lines.extend([
                f"### Run {i+1}: {run['name']}",
                f"",
                f"- **Status:** {run['status']}",
                f"- **Duration:** {run.get('duration', 0):.2f} seconds",
                f"- **Best Accuracy:** {run.get('best_accuracy', 'N/A'):.4f}" if run.get('best_accuracy') else "- **Best Accuracy:** N/A",
                f"- **Final Loss:** {run.get('final_loss', 'N/A'):.4f}" if run.get('final_loss') else "- **Final Loss:** N/A",
                f""
            ])
        
        # Best performing run
        best_run = max(
            summary['runs'],
            key=lambda x: x.get('best_accuracy', 0),
            default=None
        )
        
        if best_run:
            report_lines.extend([
                f"## Best Performing Run",
                f"",
                f"**Run:** {best_run['name']}",
                f"**Accuracy:** {best_run.get('best_accuracy', 'N/A'):.4f}" if best_run.get('best_accuracy') else "**Accuracy:** N/A",
                f""
            ])
        
        return "\n".join(report_lines)
    
    def cleanup_experiments(self, days_old: int = 30) -> int:
        """Clean up old experiment data."""
        
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        cleaned_count = 0
        
        for exp_dir in self.tracker.experiment_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            exp_metadata_file = exp_dir / "experiment.json"
            if not exp_metadata_file.exists():
                continue
            
            try:
                with open(exp_metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                created_at = datetime.fromisoformat(metadata['created_at'].replace('Z', '+00:00'))
                if created_at.timestamp() < cutoff_time:
                    # Remove experiment directory
                    import shutil
                    shutil.rmtree(exp_dir)
                    cleaned_count += 1
                    logger.info(f"Cleaned up old experiment: {metadata['experiment_id']}")
            
            except Exception as e:
                logger.warning(f"Error cleaning up experiment {exp_dir.name}: {e}")
        
        return cleaned_count