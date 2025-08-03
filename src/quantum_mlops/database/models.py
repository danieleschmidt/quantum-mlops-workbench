"""SQLAlchemy models for quantum MLOps workbench."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, 
    JSON, String, Text, UniqueConstraint, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func


Base = declarative_base()


class TimestampMixin:
    """Mixin for timestamp fields."""
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)


class ExperimentModel(Base, TimestampMixin):
    """Database model for quantum ML experiments."""
    
    __tablename__ = "experiments"
    
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    tags = Column(JSON, default=list)
    status = Column(String(50), default="active", nullable=False)
    metadata = Column(JSON, default=dict)
    
    # Relationships
    runs = relationship("RunModel", back_populates="experiment", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('ix_experiments_name_status', 'name', 'status'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "status": self.status,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class RunModel(Base, TimestampMixin):
    """Database model for experiment runs."""
    
    __tablename__ = "runs"
    
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(255), unique=True, nullable=False, index=True)
    experiment_id = Column(String(255), ForeignKey("experiments.experiment_id"), nullable=False)
    name = Column(String(255), nullable=False)
    status = Column(String(50), default="running", nullable=False)
    config = Column(JSON, default=dict)
    parameters = Column(JSON, default=dict)
    metrics = Column(JSON, default=dict)
    artifacts = Column(JSON, default=list)
    logs = Column(JSON, default=list)
    started_at = Column(DateTime, server_default=func.now())
    ended_at = Column(DateTime)
    duration = Column(Float)  # seconds
    
    # Relationships
    experiment = relationship("ExperimentModel", back_populates="runs")
    
    # Indexes
    __table_args__ = (
        Index('ix_runs_experiment_status', 'experiment_id', 'status'),
        Index('ix_runs_started_at', 'started_at'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "status": self.status,
            "config": self.config,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration": self.duration,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ModelRegistryEntry(Base, TimestampMixin):
    """Database model for quantum ML model registry."""
    
    __tablename__ = "model_registry"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    description = Column(Text)
    tags = Column(JSON, default=list)
    file_path = Column(String(500), nullable=False)
    n_qubits = Column(Integer, nullable=False)
    circuit_depth = Column(Integer)
    parameter_count = Column(Integer)
    training_metrics = Column(JSON, default=dict)
    training_history = Column(JSON, default=dict)
    validation_metrics = Column(JSON, default=dict)
    metadata = Column(JSON, default=dict)
    status = Column(String(50), default="active", nullable=False)
    
    # Model lineage
    parent_model_id = Column(String(255), ForeignKey("model_registry.model_id"))
    parent_model = relationship("ModelRegistryEntry", remote_side=[model_id])
    
    # Indexes
    __table_args__ = (
        UniqueConstraint('name', 'version', name='uq_model_name_version'),
        Index('ix_models_name_status', 'name', 'status'),
        Index('ix_models_n_qubits', 'n_qubits'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "tags": self.tags,
            "file_path": self.file_path,
            "n_qubits": self.n_qubits,
            "circuit_depth": self.circuit_depth,
            "parameter_count": self.parameter_count,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "metadata": self.metadata,
            "status": self.status,
            "parent_model_id": self.parent_model_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class JobModel(Base, TimestampMixin):
    """Database model for quantum job executions."""
    
    __tablename__ = "jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(255), unique=True, nullable=False, index=True)
    experiment_id = Column(String(255), ForeignKey("experiments.experiment_id"))
    run_id = Column(String(255), ForeignKey("runs.run_id"))
    backend = Column(String(100), nullable=False)
    device = Column(String(100))
    status = Column(String(50), default="queued", nullable=False)
    priority = Column(Integer, default=0)
    
    # Circuit information
    n_circuits = Column(Integer, nullable=False)
    n_qubits = Column(Integer, nullable=False)
    circuit_depth = Column(Integer)
    shots = Column(Integer, default=1024)
    
    # Resource usage
    estimated_cost = Column(Float, default=0.0)
    actual_cost = Column(Float)
    estimated_runtime = Column(Float)  # seconds
    actual_runtime = Column(Float)
    queue_time = Column(Float)
    
    # Job data
    circuit_data = Column(JSON)
    parameters = Column(JSON, default=dict)
    results = Column(JSON)
    error_message = Column(Text)
    
    # Timestamps
    submitted_at = Column(DateTime, server_default=func.now())
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Relationships
    experiment = relationship("ExperimentModel")
    run = relationship("RunModel")
    
    # Indexes
    __table_args__ = (
        Index('ix_jobs_backend_status', 'backend', 'status'),
        Index('ix_jobs_submitted_at', 'submitted_at'),
        Index('ix_jobs_priority', 'priority'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "backend": self.backend,
            "device": self.device,
            "status": self.status,
            "priority": self.priority,
            "n_circuits": self.n_circuits,
            "n_qubits": self.n_qubits,
            "circuit_depth": self.circuit_depth,
            "shots": self.shots,
            "estimated_cost": self.estimated_cost,
            "actual_cost": self.actual_cost,
            "estimated_runtime": self.estimated_runtime,
            "actual_runtime": self.actual_runtime,
            "queue_time": self.queue_time,
            "parameters": self.parameters,
            "results": self.results,
            "error_message": self.error_message,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class MetricModel(Base, TimestampMixin):
    """Database model for storing metrics time series."""
    
    __tablename__ = "metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(255), ForeignKey("runs.run_id"), nullable=False)
    metric_name = Column(String(255), nullable=False)
    metric_value = Column(Float, nullable=False)
    step = Column(Integer)
    timestamp = Column(DateTime, server_default=func.now())
    metadata = Column(JSON, default=dict)
    
    # Relationships
    run = relationship("RunModel")
    
    # Indexes
    __table_args__ = (
        Index('ix_metrics_run_name', 'run_id', 'metric_name'),
        Index('ix_metrics_timestamp', 'timestamp'),
        Index('ix_metrics_step', 'step'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "step": self.step,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class DeploymentModel(Base, TimestampMixin):
    """Database model for model deployments."""
    
    __tablename__ = "deployments"
    
    id = Column(Integer, primary_key=True, index=True)
    deployment_id = Column(String(255), unique=True, nullable=False, index=True)
    model_id = Column(String(255), ForeignKey("model_registry.model_id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    status = Column(String(50), default="active", nullable=False)
    endpoint_url = Column(String(500))
    config = Column(JSON, default=dict)
    
    # Performance metrics
    prediction_count = Column(Integer, default=0)
    avg_latency = Column(Float)
    success_rate = Column(Float)
    
    # Resource allocation
    cpu_limit = Column(Float)
    memory_limit = Column(Integer)  # MB
    gpu_required = Column(Boolean, default=False)
    
    # Timestamps
    deployed_at = Column(DateTime, server_default=func.now())
    last_used_at = Column(DateTime)
    
    # Relationships
    model = relationship("ModelRegistryEntry")
    
    # Indexes
    __table_args__ = (
        Index('ix_deployments_model_status', 'model_id', 'status'),
        Index('ix_deployments_deployed_at', 'deployed_at'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "deployment_id": self.deployment_id,
            "model_id": self.model_id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "endpoint_url": self.endpoint_url,
            "config": self.config,
            "prediction_count": self.prediction_count,
            "avg_latency": self.avg_latency,
            "success_rate": self.success_rate,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }