"""Sample data seeds for quantum MLOps database."""

import json
import time
from datetime import datetime, timedelta
from typing import List

from ..connection import get_db_session
from ..models import ExperimentModel, RunModel, ModelRegistryEntry, JobModel


def seed_sample_experiments() -> List[str]:
    """Create sample experiments for testing and development."""
    
    experiments_data = [
        {
            "experiment_id": "exp_vqe_h2_molecule",
            "name": "VQE H2 Molecule Ground State",
            "description": "Variational Quantum Eigensolver for finding H2 molecule ground state energy using UCCSD ansatz",
            "tags": ["vqe", "chemistry", "uccsd", "h2"],
            "metadata": {
                "molecule": "H2",
                "bond_length": 0.735,
                "basis_set": "sto-3g",
                "ansatz": "uccsd"
            }
        },
        {
            "experiment_id": "exp_qaoa_maxcut",
            "name": "QAOA MaxCut Optimization",
            "description": "Quantum Approximate Optimization Algorithm for solving MaxCut problem on random graphs",
            "tags": ["qaoa", "optimization", "maxcut", "graph"],
            "metadata": {
                "graph_nodes": 8,
                "graph_edges": 12,
                "p_layers": 3,
                "optimizer": "cobyla"
            }
        },
        {
            "experiment_id": "exp_qml_classifier",
            "name": "Quantum Machine Learning Classifier",
            "description": "Quantum classifier for binary classification using variational quantum circuits",
            "tags": ["qml", "classification", "variational", "binary"],
            "metadata": {
                "dataset": "iris_binary",
                "features": 4,
                "samples": 100,
                "circuit_type": "variational"
            }
        }
    ]
    
    created_experiment_ids = []
    
    with get_db_session() as session:
        for exp_data in experiments_data:
            # Check if experiment already exists
            existing = session.query(ExperimentModel).filter(
                ExperimentModel.experiment_id == exp_data["experiment_id"]
            ).first()
            
            if not existing:
                experiment = ExperimentModel(**exp_data)
                session.add(experiment)
                created_experiment_ids.append(exp_data["experiment_id"])
        
        session.commit()
    
    return created_experiment_ids


def seed_sample_runs() -> List[str]:
    """Create sample runs for the experiments."""
    
    runs_data = [
        {
            "run_id": "run_vqe_h2_001",
            "experiment_id": "exp_vqe_h2_molecule",
            "name": "VQE H2 Run 1 - COBYLA",
            "status": "completed",
            "config": {
                "optimizer": "COBYLA",
                "max_iter": 1000,
                "tol": 1e-6,
                "shots": 1024
            },
            "parameters": {
                "optimizer": "COBYLA",
                "max_iterations": 1000,
                "convergence_threshold": 1e-6,
                "n_qubits": 4,
                "n_layers": 2
            },
            "metrics": {
                "final_energy": [{"value": -1.857, "step": 100, "timestamp": time.time()}],
                "convergence_steps": [{"value": 87, "step": 100, "timestamp": time.time()}],
                "fidelity": [{"value": 0.995, "step": 100, "timestamp": time.time()}]
            },
            "artifacts": [
                {
                    "path": "/models/vqe_h2_run_001.pkl",
                    "type": "quantum_model",
                    "logged_at": datetime.utcnow().isoformat()
                }
            ],
            "duration": 450.5
        },
        {
            "run_id": "run_qaoa_maxcut_001",
            "experiment_id": "exp_qaoa_maxcut",
            "name": "QAOA MaxCut Run 1 - p=3",
            "status": "completed",
            "config": {
                "p_layers": 3,
                "optimizer": "COBYLA",
                "graph_type": "random",
                "n_nodes": 8
            },
            "parameters": {
                "p_layers": 3,
                "optimizer": "COBYLA",
                "n_qubits": 8,
                "graph_nodes": 8,
                "graph_edges": 12
            },
            "metrics": {
                "cut_value": [{"value": 8.2, "step": 50, "timestamp": time.time()}],
                "approximation_ratio": [{"value": 0.85, "step": 50, "timestamp": time.time()}],
                "iterations": [{"value": 45, "step": 50, "timestamp": time.time()}]
            },
            "artifacts": [
                {
                    "path": "/models/qaoa_maxcut_run_001.pkl",
                    "type": "quantum_model",
                    "logged_at": datetime.utcnow().isoformat()
                }
            ],
            "duration": 320.8
        },
        {
            "run_id": "run_qml_classifier_001",
            "experiment_id": "exp_qml_classifier",
            "name": "QML Classifier Run 1 - 4 qubits",
            "status": "completed",
            "config": {
                "n_qubits": 4,
                "n_layers": 3,
                "optimizer": "adam",
                "learning_rate": 0.01,
                "epochs": 100
            },
            "parameters": {
                "n_qubits": 4,
                "n_layers": 3,
                "optimizer": "adam",
                "learning_rate": 0.01,
                "batch_size": 32,
                "epochs": 100
            },
            "metrics": {
                "accuracy": [{"value": 0.92, "step": 100, "timestamp": time.time()}],
                "loss": [{"value": 0.15, "step": 100, "timestamp": time.time()}],
                "val_accuracy": [{"value": 0.88, "step": 100, "timestamp": time.time()}],
                "gradient_variance": [{"value": 0.025, "step": 100, "timestamp": time.time()}]
            },
            "artifacts": [
                {
                    "path": "/models/qml_classifier_run_001.pkl",
                    "type": "quantum_model",
                    "logged_at": datetime.utcnow().isoformat()
                }
            ],
            "duration": 1250.3
        }
    ]
    
    created_run_ids = []
    
    with get_db_session() as session:
        for run_data in runs_data:
            # Check if run already exists
            existing = session.query(RunModel).filter(
                RunModel.run_id == run_data["run_id"]
            ).first()
            
            if not existing:
                # Set timestamps
                run_data["started_at"] = datetime.utcnow() - timedelta(minutes=30)
                run_data["ended_at"] = run_data["started_at"] + timedelta(seconds=run_data["duration"])
                
                run = RunModel(**run_data)
                session.add(run)
                created_run_ids.append(run_data["run_id"])
        
        session.commit()
    
    return created_run_ids


def seed_sample_models() -> List[str]:
    """Create sample model registry entries."""
    
    models_data = [
        {
            "model_id": "model_vqe_h2_v1_0",
            "name": "VQE_H2_Molecule",
            "version": "1.0",
            "description": "VQE model for H2 molecule ground state calculation using UCCSD ansatz",
            "tags": ["vqe", "chemistry", "h2", "production"],
            "file_path": "/models/vqe_h2_v1_0.pkl",
            "n_qubits": 4,
            "circuit_depth": 12,
            "parameter_count": 8,
            "training_metrics": {
                "final_energy": -1.857,
                "convergence_steps": 87,
                "fidelity": 0.995,
                "training_time": 450.5
            },
            "validation_metrics": {
                "energy_accuracy": 0.999,
                "chemical_accuracy": True,
                "noise_resilience": 0.92
            },
            "metadata": {
                "molecule": "H2",
                "bond_length": 0.735,
                "basis_set": "sto-3g",
                "optimizer": "COBYLA",
                "backend": "simulator"
            }
        },
        {
            "model_id": "model_qaoa_maxcut_v1_0",
            "name": "QAOA_MaxCut",
            "version": "1.0",
            "description": "QAOA model for MaxCut optimization on 8-node graphs",
            "tags": ["qaoa", "optimization", "maxcut", "production"],
            "file_path": "/models/qaoa_maxcut_v1_0.pkl",
            "n_qubits": 8,
            "circuit_depth": 18,
            "parameter_count": 6,
            "training_metrics": {
                "cut_value": 8.2,
                "approximation_ratio": 0.85,
                "iterations": 45,
                "training_time": 320.8
            },
            "validation_metrics": {
                "avg_approximation_ratio": 0.83,
                "success_rate": 0.95,
                "scalability_score": 0.78
            },
            "metadata": {
                "graph_nodes": 8,
                "graph_edges": 12,
                "p_layers": 3,
                "optimizer": "COBYLA",
                "backend": "simulator"
            }
        },
        {
            "model_id": "model_qml_classifier_v1_0",
            "name": "QML_Binary_Classifier",
            "version": "1.0",
            "description": "Quantum machine learning binary classifier using variational circuits",
            "tags": ["qml", "classification", "binary", "production"],
            "file_path": "/models/qml_classifier_v1_0.pkl",
            "n_qubits": 4,
            "circuit_depth": 15,
            "parameter_count": 24,
            "training_metrics": {
                "accuracy": 0.92,
                "loss": 0.15,
                "gradient_variance": 0.025,
                "training_time": 1250.3
            },
            "validation_metrics": {
                "val_accuracy": 0.88,
                "val_loss": 0.18,
                "f1_score": 0.89,
                "roc_auc": 0.94
            },
            "metadata": {
                "dataset": "iris_binary",
                "features": 4,
                "circuit_type": "variational",
                "optimizer": "adam",
                "backend": "simulator"
            }
        }
    ]
    
    created_model_ids = []
    
    with get_db_session() as session:
        for model_data in models_data:
            # Check if model already exists
            existing = session.query(ModelRegistryEntry).filter(
                ModelRegistryEntry.model_id == model_data["model_id"]
            ).first()
            
            if not existing:
                model = ModelRegistryEntry(**model_data)
                session.add(model)
                created_model_ids.append(model_data["model_id"])
        
        session.commit()
    
    return created_model_ids


def seed_sample_jobs() -> List[str]:
    """Create sample quantum job entries."""
    
    jobs_data = [
        {
            "job_id": "job_vqe_h2_sim_001",
            "experiment_id": "exp_vqe_h2_molecule",
            "run_id": "run_vqe_h2_001",
            "backend": "simulator",
            "device": "local_simulator",
            "status": "completed",
            "priority": 0,
            "n_circuits": 50,
            "n_qubits": 4,
            "circuit_depth": 12,
            "shots": 1024,
            "estimated_cost": 0.0,
            "actual_cost": 0.0,
            "estimated_runtime": 400.0,
            "actual_runtime": 450.5,
            "queue_time": 0.0,
            "parameters": {
                "ansatz": "uccsd",
                "optimizer": "COBYLA",
                "max_iter": 1000
            },
            "results": {
                "final_energy": -1.857,
                "convergence": True,
                "iterations": 87,
                "success": True
            }
        },
        {
            "job_id": "job_qaoa_braket_001",
            "experiment_id": "exp_qaoa_maxcut",
            "run_id": "run_qaoa_maxcut_001",
            "backend": "aws_braket",
            "device": "Aria-1",
            "status": "completed",
            "priority": 1,
            "n_circuits": 20,
            "n_qubits": 8,
            "circuit_depth": 18,
            "shots": 1000,
            "estimated_cost": 7.5,
            "actual_cost": 8.2,
            "estimated_runtime": 300.0,
            "actual_runtime": 320.8,
            "queue_time": 45.0,
            "parameters": {
                "p_layers": 3,
                "graph_nodes": 8,
                "optimizer": "COBYLA"
            },
            "results": {
                "cut_value": 8.2,
                "approximation_ratio": 0.85,
                "success": True,
                "measurements": {"00000000": 245, "11110000": 755}
            }
        },
        {
            "job_id": "job_qml_ibm_001",
            "experiment_id": "exp_qml_classifier",
            "run_id": "run_qml_classifier_001",
            "backend": "ibm_quantum",
            "device": "ibmq_toronto",
            "status": "completed",
            "priority": 0,
            "n_circuits": 100,
            "n_qubits": 4,
            "circuit_depth": 15,
            "shots": 1024,
            "estimated_cost": 0.0,
            "actual_cost": 0.0,
            "estimated_runtime": 1200.0,
            "actual_runtime": 1250.3,
            "queue_time": 180.0,
            "parameters": {
                "n_layers": 3,
                "learning_rate": 0.01,
                "epochs": 100
            },
            "results": {
                "accuracy": 0.92,
                "loss": 0.15,
                "success": True,
                "final_parameters": [0.1, -0.3, 0.7, -0.2, 0.5, -0.8, 0.4, -0.1]
            }
        }
    ]
    
    created_job_ids = []
    
    with get_db_session() as session:
        for job_data in jobs_data:
            # Check if job already exists
            existing = session.query(JobModel).filter(
                JobModel.job_id == job_data["job_id"]
            ).first()
            
            if not existing:
                # Set timestamps
                now = datetime.utcnow()
                job_data["submitted_at"] = now - timedelta(minutes=60)
                job_data["started_at"] = job_data["submitted_at"] + timedelta(seconds=job_data["queue_time"])
                job_data["completed_at"] = job_data["started_at"] + timedelta(seconds=job_data["actual_runtime"])
                
                job = JobModel(**job_data)
                session.add(job)
                created_job_ids.append(job_data["job_id"])
        
        session.commit()
    
    return created_job_ids


def seed_all_sample_data() -> dict:
    """Seed all sample data and return summary."""
    
    print("Seeding sample data for quantum MLOps workbench...")
    
    # Seed in order due to foreign key dependencies
    experiments = seed_sample_experiments()
    runs = seed_sample_runs()
    models = seed_sample_models()
    jobs = seed_sample_jobs()
    
    summary = {
        "experiments_created": len(experiments),
        "runs_created": len(runs),
        "models_created": len(models),
        "jobs_created": len(jobs),
        "total_records": len(experiments) + len(runs) + len(models) + len(jobs)
    }
    
    print(f"Sample data seeding completed:")
    print(f"  - Experiments: {summary['experiments_created']}")
    print(f"  - Runs: {summary['runs_created']}")
    print(f"  - Models: {summary['models_created']}")
    print(f"  - Jobs: {summary['jobs_created']}")
    print(f"  - Total records: {summary['total_records']}")
    
    return summary


if __name__ == "__main__":
    seed_all_sample_data()