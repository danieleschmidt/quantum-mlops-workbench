"""Repository pattern implementations for quantum MLOps data access."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, desc, func, or_
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from .models import (
    ExperimentModel, RunModel, ModelRegistryEntry, JobModel, 
    MetricModel, DeploymentModel
)


class BaseRepository:
    """Base repository with common database operations."""
    
    def __init__(self, session: Session, model_class):
        self.session = session
        self.model_class = model_class
    
    def create(self, **kwargs) -> Any:
        """Create a new record."""
        try:
            instance = self.model_class(**kwargs)
            self.session.add(instance)
            self.session.flush()
            return instance
        except IntegrityError as e:
            self.session.rollback()
            raise ValueError(f"Integrity constraint violation: {e}")
    
    def get_by_id(self, record_id: int) -> Optional[Any]:
        """Get record by primary key."""
        return self.session.query(self.model_class).filter(
            self.model_class.id == record_id
        ).first()
    
    def get_all(self, limit: int = 100, offset: int = 0) -> List[Any]:
        """Get all records with pagination."""
        return self.session.query(self.model_class).offset(offset).limit(limit).all()
    
    def update(self, record_id: int, **kwargs) -> Optional[Any]:
        """Update a record by ID."""
        instance = self.get_by_id(record_id)
        if instance:
            for key, value in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            instance.updated_at = datetime.utcnow()
            self.session.flush()
        return instance
    
    def delete(self, record_id: int) -> bool:
        """Delete a record by ID."""
        instance = self.get_by_id(record_id)
        if instance:
            self.session.delete(instance)
            self.session.flush()
            return True
        return False
    
    def count(self) -> int:
        """Count total records."""
        return self.session.query(self.model_class).count()


class ExperimentRepository(BaseRepository):
    """Repository for experiment data operations."""
    
    def __init__(self, session: Session):
        super().__init__(session, ExperimentModel)
    
    def get_by_experiment_id(self, experiment_id: str) -> Optional[ExperimentModel]:
        """Get experiment by experiment_id."""
        return self.session.query(ExperimentModel).filter(
            ExperimentModel.experiment_id == experiment_id
        ).first()
    
    def get_by_name(self, name: str) -> Optional[ExperimentModel]:
        """Get experiment by name."""
        return self.session.query(ExperimentModel).filter(
            ExperimentModel.name == name
        ).first()
    
    def search(
        self,
        name_filter: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ExperimentModel]:
        """Search experiments with filters."""
        query = self.session.query(ExperimentModel)
        
        if name_filter:
            query = query.filter(ExperimentModel.name.ilike(f"%{name_filter}%"))
        
        if status:
            query = query.filter(ExperimentModel.status == status)
        
        if tags:
            # PostgreSQL JSON contains check
            for tag in tags:
                query = query.filter(ExperimentModel.tags.contains([tag]))
        
        return query.order_by(desc(ExperimentModel.created_at)).offset(offset).limit(limit).all()
    
    def get_with_runs(self, experiment_id: str) -> Optional[ExperimentModel]:
        """Get experiment with all associated runs."""
        return self.session.query(ExperimentModel).filter(
            ExperimentModel.experiment_id == experiment_id
        ).first()
    
    def get_recent(self, days: int = 7, limit: int = 10) -> List[ExperimentModel]:
        """Get recent experiments."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        return self.session.query(ExperimentModel).filter(
            ExperimentModel.created_at >= cutoff_date
        ).order_by(desc(ExperimentModel.created_at)).limit(limit).all()


class RunRepository(BaseRepository):
    """Repository for run data operations."""
    
    def __init__(self, session: Session):
        super().__init__(session, RunModel)
    
    def get_by_run_id(self, run_id: str) -> Optional[RunModel]:
        """Get run by run_id."""
        return self.session.query(RunModel).filter(
            RunModel.run_id == run_id
        ).first()
    
    def get_by_experiment(
        self,
        experiment_id: str,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[RunModel]:
        """Get runs for an experiment."""
        query = self.session.query(RunModel).filter(
            RunModel.experiment_id == experiment_id
        )
        
        if status:
            query = query.filter(RunModel.status == status)
        
        return query.order_by(desc(RunModel.started_at)).limit(limit).all()
    
    def get_best_run(
        self,
        experiment_id: str,
        metric_name: str = "accuracy",
        maximize: bool = True
    ) -> Optional[RunModel]:
        """Get best performing run for an experiment."""
        runs = self.get_by_experiment(experiment_id)
        
        best_run = None
        best_score = float('-inf') if maximize else float('inf')
        
        for run in runs:
            if metric_name in run.metrics:
                metric_values = run.metrics[metric_name]
                if isinstance(metric_values, list) and metric_values:
                    # Get latest metric value
                    score = metric_values[-1].get('value', 0)
                elif isinstance(metric_values, (int, float)):
                    score = metric_values
                else:
                    continue
                
                if maximize and score > best_score or not maximize and score < best_score:
                    best_score = score
                    best_run = run
        
        return best_run
    
    def update_metrics(self, run_id: str, metrics: Dict[str, Any]) -> Optional[RunModel]:
        """Update run metrics."""
        run = self.get_by_run_id(run_id)
        if run:
            if run.metrics is None:
                run.metrics = {}
            run.metrics.update(metrics)
            run.updated_at = datetime.utcnow()
            self.session.flush()
        return run
    
    def add_artifact(self, run_id: str, artifact: Dict[str, Any]) -> Optional[RunModel]:
        """Add artifact to run."""
        run = self.get_by_run_id(run_id)
        if run:
            if run.artifacts is None:
                run.artifacts = []
            run.artifacts.append(artifact)
            run.updated_at = datetime.utcnow()
            self.session.flush()
        return run
    
    def get_running_runs(self) -> List[RunModel]:
        """Get all currently running runs."""
        return self.session.query(RunModel).filter(
            RunModel.status == "running"
        ).all()


class ModelRepository(BaseRepository):
    """Repository for model registry operations."""
    
    def __init__(self, session: Session):
        super().__init__(session, ModelRegistryEntry)
    
    def get_by_model_id(self, model_id: str) -> Optional[ModelRegistryEntry]:
        """Get model by model_id."""
        return self.session.query(ModelRegistryEntry).filter(
            ModelRegistryEntry.model_id == model_id
        ).first()
    
    def get_by_name_version(self, name: str, version: str) -> Optional[ModelRegistryEntry]:
        """Get model by name and version."""
        return self.session.query(ModelRegistryEntry).filter(
            and_(
                ModelRegistryEntry.name == name,
                ModelRegistryEntry.version == version
            )
        ).first()
    
    def get_versions(self, name: str) -> List[ModelRegistryEntry]:
        """Get all versions of a model."""
        return self.session.query(ModelRegistryEntry).filter(
            ModelRegistryEntry.name == name
        ).order_by(desc(ModelRegistryEntry.created_at)).all()
    
    def get_latest_version(self, name: str) -> Optional[ModelRegistryEntry]:
        """Get latest version of a model."""
        versions = self.get_versions(name)
        return versions[0] if versions else None
    
    def search(
        self,
        name_filter: Optional[str] = None,
        tags: Optional[List[str]] = None,
        n_qubits: Optional[int] = None,
        status: str = "active",
        limit: int = 100,
        offset: int = 0
    ) -> List[ModelRegistryEntry]:
        """Search models with filters."""
        query = self.session.query(ModelRegistryEntry)
        
        if name_filter:
            query = query.filter(ModelRegistryEntry.name.ilike(f"%{name_filter}%"))
        
        if status:
            query = query.filter(ModelRegistryEntry.status == status)
        
        if n_qubits:
            query = query.filter(ModelRegistryEntry.n_qubits == n_qubits)
        
        if tags:
            for tag in tags:
                query = query.filter(ModelRegistryEntry.tags.contains([tag]))
        
        return query.order_by(desc(ModelRegistryEntry.created_at)).offset(offset).limit(limit).all()
    
    def get_best_models(
        self,
        metric_name: str = "accuracy",
        limit: int = 10
    ) -> List[ModelRegistryEntry]:
        """Get best performing models by metric."""
        # This is a simplified implementation
        # In practice, would need more complex logic to extract metric values
        return self.session.query(ModelRegistryEntry).filter(
            ModelRegistryEntry.validation_metrics.isnot(None)
        ).order_by(desc(ModelRegistryEntry.created_at)).limit(limit).all()
    
    def update_status(self, model_id: str, status: str) -> Optional[ModelRegistryEntry]:
        """Update model status."""
        model = self.get_by_model_id(model_id)
        if model:
            model.status = status
            model.updated_at = datetime.utcnow()
            self.session.flush()
        return model


class JobRepository(BaseRepository):
    """Repository for quantum job operations."""
    
    def __init__(self, session: Session):
        super().__init__(session, JobModel)
    
    def get_by_job_id(self, job_id: str) -> Optional[JobModel]:
        """Get job by job_id."""
        return self.session.query(JobModel).filter(
            JobModel.job_id == job_id
        ).first()
    
    def get_by_status(self, status: str, limit: int = 100) -> List[JobModel]:
        """Get jobs by status."""
        return self.session.query(JobModel).filter(
            JobModel.status == status
        ).order_by(desc(JobModel.submitted_at)).limit(limit).all()
    
    def get_queued_jobs(self, backend: Optional[str] = None) -> List[JobModel]:
        """Get queued jobs, optionally filtered by backend."""
        query = self.session.query(JobModel).filter(
            JobModel.status == "queued"
        )
        
        if backend:
            query = query.filter(JobModel.backend == backend)
        
        return query.order_by(desc(JobModel.priority), JobModel.submitted_at).all()
    
    def get_by_experiment(self, experiment_id: str) -> List[JobModel]:
        """Get all jobs for an experiment."""
        return self.session.query(JobModel).filter(
            JobModel.experiment_id == experiment_id
        ).order_by(desc(JobModel.submitted_at)).all()
    
    def update_status(
        self,
        job_id: str,
        status: str,
        **kwargs
    ) -> Optional[JobModel]:
        """Update job status and other fields."""
        job = self.get_by_job_id(job_id)
        if job:
            job.status = status
            
            # Update timestamps based on status
            if status == "running" and not job.started_at:
                job.started_at = datetime.utcnow()
                if job.submitted_at:
                    job.queue_time = (job.started_at - job.submitted_at).total_seconds()
            
            elif status in ["completed", "failed", "cancelled"] and not job.completed_at:
                job.completed_at = datetime.utcnow()
                if job.started_at:
                    job.actual_runtime = (job.completed_at - job.started_at).total_seconds()
            
            # Update additional fields
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            
            job.updated_at = datetime.utcnow()
            self.session.flush()
        
        return job
    
    def get_cost_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        backend: Optional[str] = None
    ) -> Dict[str, float]:
        """Get cost summary for jobs."""
        query = self.session.query(JobModel).filter(
            JobModel.actual_cost.isnot(None)
        )
        
        if start_date:
            query = query.filter(JobModel.completed_at >= start_date)
        
        if end_date:
            query = query.filter(JobModel.completed_at <= end_date)
        
        if backend:
            query = query.filter(JobModel.backend == backend)
        
        result = query.with_entities(
            func.sum(JobModel.actual_cost).label('total_cost'),
            func.count(JobModel.id).label('job_count'),
            func.avg(JobModel.actual_cost).label('avg_cost')
        ).first()
        
        return {
            'total_cost': float(result.total_cost or 0),
            'job_count': int(result.job_count or 0),
            'avg_cost': float(result.avg_cost or 0)
        }
    
    def get_performance_stats(self, backend: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics for jobs."""
        query = self.session.query(JobModel).filter(
            JobModel.status == "completed"
        )
        
        if backend:
            query = query.filter(JobModel.backend == backend)
        
        jobs = query.all()
        
        if not jobs:
            return {}
        
        runtimes = [job.actual_runtime for job in jobs if job.actual_runtime]
        queue_times = [job.queue_time for job in jobs if job.queue_time]
        
        return {
            'total_jobs': len(jobs),
            'avg_runtime': sum(runtimes) / len(runtimes) if runtimes else 0,
            'avg_queue_time': sum(queue_times) / len(queue_times) if queue_times else 0,
            'success_rate': len([j for j in jobs if j.status == "completed"]) / len(jobs)
        }


class MetricRepository(BaseRepository):
    """Repository for metrics data operations."""
    
    def __init__(self, session: Session):
        super().__init__(session, MetricModel)
    
    def log_metric(
        self,
        run_id: str,
        metric_name: str,
        value: float,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MetricModel:
        """Log a metric value."""
        return self.create(
            run_id=run_id,
            metric_name=metric_name,
            metric_value=value,
            step=step,
            metadata=metadata or {}
        )
    
    def get_metrics(
        self,
        run_id: str,
        metric_name: Optional[str] = None
    ) -> List[MetricModel]:
        """Get metrics for a run."""
        query = self.session.query(MetricModel).filter(
            MetricModel.run_id == run_id
        )
        
        if metric_name:
            query = query.filter(MetricModel.metric_name == metric_name)
        
        return query.order_by(MetricModel.step, MetricModel.timestamp).all()
    
    def get_metric_history(
        self,
        run_id: str,
        metric_name: str
    ) -> List[Tuple[int, float, datetime]]:
        """Get metric history as tuples of (step, value, timestamp)."""
        metrics = self.session.query(MetricModel).filter(
            and_(
                MetricModel.run_id == run_id,
                MetricModel.metric_name == metric_name
            )
        ).order_by(MetricModel.step, MetricModel.timestamp).all()
        
        return [(m.step, m.metric_value, m.timestamp) for m in metrics]


class DeploymentRepository(BaseRepository):
    """Repository for model deployment operations."""
    
    def __init__(self, session: Session):
        super().__init__(session, DeploymentModel)
    
    def get_by_deployment_id(self, deployment_id: str) -> Optional[DeploymentModel]:
        """Get deployment by deployment_id."""
        return self.session.query(DeploymentModel).filter(
            DeploymentModel.deployment_id == deployment_id
        ).first()
    
    def get_by_model(self, model_id: str) -> List[DeploymentModel]:
        """Get all deployments for a model."""
        return self.session.query(DeploymentModel).filter(
            DeploymentModel.model_id == model_id
        ).order_by(desc(DeploymentModel.deployed_at)).all()
    
    def get_active_deployments(self) -> List[DeploymentModel]:
        """Get all active deployments."""
        return self.session.query(DeploymentModel).filter(
            DeploymentModel.status == "active"
        ).all()
    
    def update_usage_stats(
        self,
        deployment_id: str,
        prediction_count: int,
        avg_latency: float,
        success_rate: float
    ) -> Optional[DeploymentModel]:
        """Update deployment usage statistics."""
        deployment = self.get_by_deployment_id(deployment_id)
        if deployment:
            deployment.prediction_count = prediction_count
            deployment.avg_latency = avg_latency
            deployment.success_rate = success_rate
            deployment.last_used_at = datetime.utcnow()
            deployment.updated_at = datetime.utcnow()
            self.session.flush()
        return deployment