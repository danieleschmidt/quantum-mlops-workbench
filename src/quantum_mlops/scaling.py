"""Auto-scaling and performance optimization for quantum MLOps workloads."""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np

from .core import QuantumMLPipeline, QuantumDevice, QuantumModel
from .exceptions import QuantumMLOpsException
from .monitoring import QuantumMonitor
from .i18n import translate as _

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of compute resources."""
    CPU = "cpu"
    QUANTUM_SIMULATOR = "quantum_simulator"
    QUANTUM_HARDWARE = "quantum_hardware"
    MEMORY = "memory"


@dataclass
class ResourceUsage:
    """Resource usage metrics."""
    
    cpu_percent: float
    memory_mb: float
    quantum_shots_used: int
    quantum_queue_time: float
    active_jobs: int
    pending_jobs: int


@dataclass
class ScalingDecision:
    """Auto-scaling decision."""
    
    should_scale: bool
    scale_direction: str  # "up" or "down"
    resource_type: ResourceType
    scale_factor: float
    reason: str


class LoadBalancer:
    """Load balancer for quantum computing jobs."""
    
    def __init__(self, backends: List[QuantumDevice]) -> None:
        """Initialize load balancer.
        
        Args:
            backends: List of available quantum backends
        """
        self.backends = backends
        self.backend_loads: Dict[QuantumDevice, float] = {backend: 0.0 for backend in backends}
        self.backend_queues: Dict[QuantumDevice, queue.Queue] = {
            backend: queue.Queue() for backend in backends
        }
        self._lock = threading.Lock()
    
    def get_best_backend(self, job_requirements: Dict[str, Any]) -> QuantumDevice:
        """Select the best backend for a job based on current load.
        
        Args:
            job_requirements: Job requirements (qubits, shots, etc.)
            
        Returns:
            Best available backend
        """
        with self._lock:
            # Filter backends that can handle the job
            suitable_backends = []
            
            for backend in self.backends:
                if self._can_handle_job(backend, job_requirements):
                    suitable_backends.append(backend)
            
            if not suitable_backends:
                # Fallback to simulator
                return QuantumDevice.SIMULATOR
            
            # Select backend with lowest load
            best_backend = min(suitable_backends, key=lambda b: self.backend_loads[b])
            
            # Update load estimate
            self.backend_loads[best_backend] += self._estimate_job_load(job_requirements)
            
            return best_backend
    
    def _can_handle_job(self, backend: QuantumDevice, requirements: Dict[str, Any]) -> bool:
        """Check if backend can handle the job requirements."""
        n_qubits = requirements.get('n_qubits', 4)
        
        # Backend capacity limits (simplified)
        limits = {
            QuantumDevice.SIMULATOR: 30,
            QuantumDevice.AWS_BRAKET: 25,
            QuantumDevice.IBM_QUANTUM: 127,
            QuantumDevice.IONQ: 11
        }
        
        return n_qubits <= limits.get(backend, 30)
    
    def _estimate_job_load(self, requirements: Dict[str, Any]) -> float:
        """Estimate the computational load of a job."""
        n_qubits = requirements.get('n_qubits', 4)
        shots = requirements.get('shots', 1024)
        epochs = requirements.get('epochs', 50)
        
        # Simplified load calculation
        return (n_qubits ** 2) * (shots / 1000) * (epochs / 50)
    
    def release_backend(self, backend: QuantumDevice, job_load: float) -> None:
        """Release backend resources after job completion.
        
        Args:
            backend: Backend that completed the job
            job_load: Load that was released
        """
        with self._lock:
            self.backend_loads[backend] = max(0, self.backend_loads[backend] - job_load)
    
    def get_load_status(self) -> Dict[QuantumDevice, float]:
        """Get current load status for all backends.
        
        Returns:
            Dictionary mapping backends to their current loads
        """
        with self._lock:
            return self.backend_loads.copy()


class QuantumJobScheduler:
    """Advanced job scheduler for quantum workloads."""
    
    def __init__(
        self,
        max_concurrent_jobs: int = 5,
        max_quantum_jobs: int = 2,
        priority_levels: int = 3
    ) -> None:
        """Initialize quantum job scheduler.
        
        Args:
            max_concurrent_jobs: Maximum concurrent jobs
            max_quantum_jobs: Maximum concurrent quantum hardware jobs
            priority_levels: Number of priority levels
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_quantum_jobs = max_quantum_jobs
        self.priority_levels = priority_levels
        
        # Job queues by priority (higher index = higher priority)
        self.job_queues: List[queue.PriorityQueue] = [
            queue.PriorityQueue() for _ in range(priority_levels)
        ]
        
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.job_history: List[Dict[str, Any]] = []
        
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._running = True
        self._scheduler_thread.start()
    
    def submit_job(
        self,
        job_id: str,
        job_function: callable,
        job_args: Tuple[Any, ...],
        job_kwargs: Dict[str, Any],
        priority: int = 1,
        use_quantum_hardware: bool = False
    ) -> str:
        """Submit a job to the scheduler.
        
        Args:
            job_id: Unique job identifier
            job_function: Function to execute
            job_args: Function arguments
            job_kwargs: Function keyword arguments
            priority: Job priority (0=low, 1=medium, 2=high)
            use_quantum_hardware: Whether job needs quantum hardware
            
        Returns:
            Job ID
        """
        priority = max(0, min(priority, self.priority_levels - 1))
        
        job_info = {
            'id': job_id,
            'function': job_function,
            'args': job_args,
            'kwargs': job_kwargs,
            'priority': priority,
            'use_quantum_hardware': use_quantum_hardware,
            'submitted_at': time.time(),
            'status': 'queued'
        }
        
        # Add to appropriate priority queue
        self.job_queues[priority].put((time.time(), job_info))
        
        logger.info(f"Job {job_id} submitted with priority {priority}")
        return job_id
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                # Check if we can schedule more jobs
                if len(self.active_jobs) >= self.max_concurrent_jobs:
                    time.sleep(0.1)
                    continue
                
                # Count quantum hardware jobs
                quantum_jobs = sum(
                    1 for job in self.active_jobs.values()
                    if job.get('use_quantum_hardware', False)
                )
                
                # Find next job to schedule
                next_job = self._get_next_job(quantum_jobs < self.max_quantum_jobs)
                
                if next_job:
                    self._execute_job(next_job)
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(1)
    
    def _get_next_job(self, can_use_quantum: bool) -> Optional[Dict[str, Any]]:
        """Get the next job to execute."""
        # Check queues from highest to lowest priority
        for priority in range(self.priority_levels - 1, -1, -1):
            try:
                # Non-blocking get
                _, job_info = self.job_queues[priority].get_nowait()
                
                # Check if we can run this job
                if job_info['use_quantum_hardware'] and not can_use_quantum:
                    # Put it back and try next
                    self.job_queues[priority].put((time.time(), job_info))
                    continue
                
                return job_info
                
            except queue.Empty:
                continue
        
        return None
    
    def _execute_job(self, job_info: Dict[str, Any]) -> None:
        """Execute a job."""
        job_id = job_info['id']
        
        # Mark as active
        job_info['status'] = 'running'
        job_info['started_at'] = time.time()
        self.active_jobs[job_id] = job_info
        
        # Submit to executor
        if job_info.get('cpu_intensive', False):
            future = self.process_executor.submit(
                job_info['function'],
                *job_info['args'],
                **job_info['kwargs']
            )
        else:
            future = self.executor.submit(
                job_info['function'],
                *job_info['args'],
                **job_info['kwargs']
            )
        
        # Add completion callback
        future.add_done_callback(lambda f: self._job_completed(job_id, f))
    
    def _job_completed(self, job_id: str, future) -> None:
        """Handle job completion."""
        if job_id not in self.active_jobs:
            return
        
        job_info = self.active_jobs.pop(job_id)
        job_info['completed_at'] = time.time()
        job_info['duration'] = job_info['completed_at'] - job_info['started_at']
        
        try:
            result = future.result()
            job_info['status'] = 'completed'
            job_info['result'] = result
        except Exception as e:
            job_info['status'] = 'failed'
            job_info['error'] = str(e)
            logger.error(f"Job {job_id} failed: {e}")
        
        # Add to history
        self.job_history.append(job_info)
        
        # Keep only recent history
        if len(self.job_history) > 1000:
            self.job_history = self.job_history[-500:]
        
        logger.info(f"Job {job_id} {job_info['status']} in {job_info['duration']:.2f}s")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job."""
        # Check active jobs
        if job_id in self.active_jobs:
            return self.active_jobs[job_id].copy()
        
        # Check history
        for job in self.job_history:
            if job['id'] == job_id:
                return job.copy()
        
        return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get scheduler queue status."""
        queue_sizes = [q.qsize() for q in self.job_queues]
        
        return {
            'active_jobs': len(self.active_jobs),
            'queued_jobs': sum(queue_sizes),
            'queue_sizes_by_priority': queue_sizes,
            'completed_jobs': len([j for j in self.job_history if j['status'] == 'completed']),
            'failed_jobs': len([j for j in self.job_history if j['status'] == 'failed'])
        }
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        # TODO: Implement job cancellation
        logger.warning(f"Job cancellation not yet implemented for {job_id}")
        return False
    
    def shutdown(self) -> None:
        """Shutdown the scheduler."""
        self._running = False
        self.executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)


class AutoScaler:
    """Auto-scaling system for quantum MLOps workloads."""
    
    def __init__(
        self,
        target_cpu_utilization: float = 0.7,
        target_memory_utilization: float = 0.8,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        cooldown_period: int = 300  # 5 minutes
    ) -> None:
        """Initialize auto-scaler.
        
        Args:
            target_cpu_utilization: Target CPU utilization
            target_memory_utilization: Target memory utilization
            scale_up_threshold: Threshold for scaling up
            scale_down_threshold: Threshold for scaling down
            cooldown_period: Cooldown period between scaling actions (seconds)
        """
        self.target_cpu_utilization = target_cpu_utilization
        self.target_memory_utilization = target_memory_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        
        self.last_scaling_action = 0
        self.scaling_history: List[Dict[str, Any]] = []
        
        # Resource monitoring
        self.resource_usage_history: List[ResourceUsage] = []
        self.max_history_size = 100
    
    def record_resource_usage(self, usage: ResourceUsage) -> None:
        """Record current resource usage.
        
        Args:
            usage: Current resource usage metrics
        """
        self.resource_usage_history.append(usage)
        
        # Keep only recent history
        if len(self.resource_usage_history) > self.max_history_size:
            self.resource_usage_history = self.resource_usage_history[-self.max_history_size//2:]
    
    def should_scale(self) -> Optional[ScalingDecision]:
        """Determine if scaling is needed.
        
        Returns:
            ScalingDecision if scaling is recommended, None otherwise
        """
        if not self.resource_usage_history:
            return None
        
        # Check cooldown period
        if time.time() - self.last_scaling_action < self.cooldown_period:
            return None
        
        # Get recent usage (last 10 data points)
        recent_usage = self.resource_usage_history[-10:]
        
        # Calculate average metrics
        avg_cpu = np.mean([u.cpu_percent for u in recent_usage]) / 100.0
        avg_memory = np.mean([u.memory_mb for u in recent_usage])
        avg_queue_time = np.mean([u.quantum_queue_time for u in recent_usage])
        
        # Check CPU scaling
        if avg_cpu > self.scale_up_threshold:
            return ScalingDecision(
                should_scale=True,
                scale_direction="up",
                resource_type=ResourceType.CPU,
                scale_factor=1.5,
                reason=f"High CPU utilization: {avg_cpu:.2%}"
            )
        elif avg_cpu < self.scale_down_threshold:
            return ScalingDecision(
                should_scale=True,
                scale_direction="down",
                resource_type=ResourceType.CPU,
                scale_factor=0.7,
                reason=f"Low CPU utilization: {avg_cpu:.2%}"
            )
        
        # Check quantum queue time
        if avg_queue_time > 300:  # 5 minutes
            return ScalingDecision(
                should_scale=True,
                scale_direction="up",
                resource_type=ResourceType.QUANTUM_HARDWARE,
                scale_factor=1.0,  # Request more quantum resources
                reason=f"High quantum queue time: {avg_queue_time:.1f}s"
            )
        
        return None
    
    def execute_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Execute a scaling decision.
        
        Args:
            decision: Scaling decision to execute
            
        Returns:
            True if scaling was successful
        """
        try:
            logger.info(f"Executing scaling decision: {decision.reason}")
            
            # Record scaling action
            scaling_record = {
                'timestamp': time.time(),
                'decision': decision,
                'success': True
            }
            
            # Here you would implement actual scaling logic
            # For example:
            # - Scale Kubernetes pods
            # - Request more quantum backend access
            # - Adjust resource limits
            
            # Placeholder implementation
            if decision.resource_type == ResourceType.CPU:
                self._scale_cpu_resources(decision.scale_direction, decision.scale_factor)
            elif decision.resource_type == ResourceType.QUANTUM_HARDWARE:
                self._scale_quantum_resources(decision.scale_direction, decision.scale_factor)
            
            self.last_scaling_action = time.time()
            self.scaling_history.append(scaling_record)
            
            return True
            
        except Exception as e:
            logger.error(f"Scaling execution failed: {e}")
            scaling_record['success'] = False
            scaling_record['error'] = str(e)
            self.scaling_history.append(scaling_record)
            return False
    
    def _scale_cpu_resources(self, direction: str, factor: float) -> None:
        """Scale CPU resources."""
        if direction == "up":
            logger.info(f"Scaling CPU resources up by factor {factor}")
            # Implement CPU scaling up (e.g., increase thread pool size)
        else:
            logger.info(f"Scaling CPU resources down by factor {factor}")
            # Implement CPU scaling down
    
    def _scale_quantum_resources(self, direction: str, factor: float) -> None:
        """Scale quantum resources."""
        if direction == "up":
            logger.info("Requesting additional quantum backend access")
            # Implement quantum resource scaling
        else:
            logger.info("Reducing quantum resource usage")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        recent_actions = [
            action for action in self.scaling_history
            if time.time() - action['timestamp'] < 3600  # Last hour
        ]
        
        return {
            'last_scaling_action': self.last_scaling_action,
            'cooldown_remaining': max(0, self.cooldown_period - (time.time() - self.last_scaling_action)),
            'recent_actions': len(recent_actions),
            'successful_actions': len([a for a in recent_actions if a['success']]),
            'current_targets': {
                'cpu_utilization': self.target_cpu_utilization,
                'memory_utilization': self.target_memory_utilization
            }
        }


class PerformanceOptimizer:
    """Performance optimization for quantum ML workloads."""
    
    def __init__(self) -> None:
        """Initialize performance optimizer."""
        self.optimization_cache: Dict[str, Any] = {}
        self.performance_metrics: List[Dict[str, Any]] = []
    
    def optimize_circuit_execution(
        self,
        pipeline: QuantumMLPipeline,
        X: np.ndarray,
        optimization_level: int = 2
    ) -> QuantumMLPipeline:
        """Optimize quantum circuit execution.
        
        Args:
            pipeline: Quantum ML pipeline to optimize
            X: Input data for optimization
            optimization_level: Level of optimization (0-3)
            
        Returns:
            Optimized pipeline
        """
        cache_key = f"circuit_opt_{pipeline.n_qubits}_{optimization_level}_{hash(X.tobytes())}"
        
        if cache_key in self.optimization_cache:
            logger.info("Using cached circuit optimization")
            return self.optimization_cache[cache_key]
        
        start_time = time.time()
        
        # Circuit-level optimizations
        if optimization_level >= 1:
            pipeline = self._optimize_circuit_depth(pipeline)
        
        if optimization_level >= 2:
            pipeline = self._optimize_parameter_encoding(pipeline, X)
        
        if optimization_level >= 3:
            pipeline = self._optimize_measurement_strategy(pipeline)
        
        optimization_time = time.time() - start_time
        
        # Cache the result
        self.optimization_cache[cache_key] = pipeline
        
        # Record performance metrics
        self.performance_metrics.append({
            'timestamp': time.time(),
            'optimization_level': optimization_level,
            'optimization_time': optimization_time,
            'n_qubits': pipeline.n_qubits,
            'data_size': X.shape[0]
        })
        
        logger.info(f"Circuit optimization completed in {optimization_time:.3f}s")
        return pipeline
    
    def _optimize_circuit_depth(self, pipeline: QuantumMLPipeline) -> QuantumMLPipeline:
        """Optimize circuit depth."""
        # Placeholder for circuit depth optimization
        logger.debug("Optimizing circuit depth")
        return pipeline
    
    def _optimize_parameter_encoding(self, pipeline: QuantumMLPipeline, X: np.ndarray) -> QuantumMLPipeline:
        """Optimize parameter encoding strategy."""
        # Analyze data distribution for optimal encoding
        data_range = np.ptp(X, axis=0)
        data_std = np.std(X, axis=0)
        
        # Optimize encoding based on data characteristics
        if np.any(data_range > 10):
            logger.debug("Applying normalization for parameter encoding")
            # Could modify pipeline encoding strategy here
        
        return pipeline
    
    def _optimize_measurement_strategy(self, pipeline: QuantumMLPipeline) -> QuantumMLPipeline:
        """Optimize measurement strategy."""
        logger.debug("Optimizing measurement strategy")
        # Could implement adaptive shot allocation, measurement basis optimization, etc.
        return pipeline
    
    def optimize_batch_processing(
        self,
        data_batches: List[np.ndarray],
        pipeline: QuantumMLPipeline
    ) -> List[np.ndarray]:
        """Optimize batch processing strategy.
        
        Args:
            data_batches: List of data batches
            pipeline: Quantum ML pipeline
            
        Returns:
            Optimized batch order
        """
        # Sort batches by size for efficient processing
        sorted_batches = sorted(data_batches, key=lambda x: x.shape[0])
        
        # Group similar-sized batches together
        optimized_batches = []
        current_group = []
        current_size = 0
        
        for batch in sorted_batches:
            if not current_group or abs(batch.shape[0] - current_size) <= current_size * 0.2:
                current_group.append(batch)
                current_size = batch.shape[0]
            else:
                optimized_batches.extend(current_group)
                current_group = [batch]
                current_size = batch.shape[0]
        
        if current_group:
            optimized_batches.extend(current_group)
        
        return optimized_batches
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance optimization report.
        
        Returns:
            Performance metrics and recommendations
        """
        if not self.performance_metrics:
            return {"message": "No performance data available"}
        
        recent_metrics = self.performance_metrics[-50:]  # Last 50 optimizations
        
        avg_optimization_time = np.mean([m['optimization_time'] for m in recent_metrics])
        cache_hit_rate = len(self.optimization_cache) / len(self.performance_metrics)
        
        return {
            'total_optimizations': len(self.performance_metrics),
            'cache_size': len(self.optimization_cache),
            'cache_hit_rate': cache_hit_rate,
            'avg_optimization_time': avg_optimization_time,
            'recent_optimizations': len(recent_metrics),
            'recommendations': self._generate_recommendations(recent_metrics)
        }
    
    def _generate_recommendations(self, metrics: List[Dict[str, Any]]) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        avg_opt_time = np.mean([m['optimization_time'] for m in metrics])
        if avg_opt_time > 1.0:
            recommendations.append("Consider reducing optimization level for faster processing")
        
        large_circuits = [m for m in metrics if m['n_qubits'] > 20]
        if len(large_circuits) > len(metrics) * 0.5:
            recommendations.append("Many large circuits detected - consider circuit decomposition")
        
        if len(self.optimization_cache) > 1000:
            recommendations.append("Cache size is large - consider periodic cleanup")
        
        return recommendations


# Global instances
_load_balancer: Optional[LoadBalancer] = None
_job_scheduler: Optional[QuantumJobScheduler] = None
_auto_scaler: Optional[AutoScaler] = None
_performance_optimizer: Optional[PerformanceOptimizer] = None


def get_load_balancer(backends: Optional[List[QuantumDevice]] = None) -> LoadBalancer:
    """Get global load balancer instance."""
    global _load_balancer
    if _load_balancer is None:
        if backends is None:
            backends = [QuantumDevice.SIMULATOR, QuantumDevice.AWS_BRAKET]
        _load_balancer = LoadBalancer(backends)
    return _load_balancer


def get_job_scheduler() -> QuantumJobScheduler:
    """Get global job scheduler instance."""
    global _job_scheduler
    if _job_scheduler is None:
        _job_scheduler = QuantumJobScheduler()
    return _job_scheduler


def get_auto_scaler() -> AutoScaler:
    """Get global auto-scaler instance."""
    global _auto_scaler
    if _auto_scaler is None:
        _auto_scaler = AutoScaler()
    return _auto_scaler


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer