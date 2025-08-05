"""Performance optimization and scaling features for quantum MLOps workbench."""

import asyncio
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import numpy as np
import pickle
import hashlib
import os
from pathlib import Path
import json
from queue import Queue, Empty
from functools import wraps, lru_cache
import weakref

from .exceptions import QuantumMLOpsException, ErrorCategory, ErrorSeverity
from .logging_config import get_logger, log_performance
from .cache.manager import CacheManager

logger = get_logger("optimization")


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    throughput: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self, success: bool = True, error_message: str = None):
        """Mark operation as complete."""
        self.end_time = datetime.utcnow()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.success = success
        self.error_message = error_message
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation_name": self.operation_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_percent": self.cpu_percent,
            "throughput": self.throughput,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata
        }


class PerformanceProfiler:
    """Performance profiler for quantum operations."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self.active_operations: Dict[str, PerformanceMetrics] = {}
        self.completed_operations: List[PerformanceMetrics] = []
        self.max_history = 10000
        
    def start_operation(self, operation_id: str, operation_name: str, **metadata) -> PerformanceMetrics:
        """Start profiling an operation."""
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=datetime.utcnow(),
            metadata=metadata
        )
        self.active_operations[operation_id] = metrics
        return metrics
        
    def end_operation(self, operation_id: str, success: bool = True, 
                     error_message: str = None, **metadata) -> Optional[PerformanceMetrics]:
        """End profiling an operation."""
        metrics = self.active_operations.pop(operation_id, None)
        if metrics:
            metrics.complete(success=success, error_message=error_message)
            metrics.metadata.update(metadata)
            
            # Store in history
            self.completed_operations.append(metrics)
            if len(self.completed_operations) > self.max_history:
                self.completed_operations = self.completed_operations[-self.max_history:]
                
            # Log performance
            if metrics.success:
                logger.info(f"Operation completed: {metrics.operation_name} ({metrics.duration_seconds:.3f}s)")
            else:
                logger.error(f"Operation failed: {metrics.operation_name} - {error_message}")
                
        return metrics
        
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for an operation type."""
        ops = [op for op in self.completed_operations if op.operation_name == operation_name]
        
        if not ops:
            return {"count": 0}
            
        durations = [op.duration_seconds for op in ops if op.duration_seconds is not None]
        successful_ops = [op for op in ops if op.success]
        
        return {
            "count": len(ops),
            "success_rate": len(successful_ops) / len(ops),
            "avg_duration": np.mean(durations) if durations else None,
            "min_duration": np.min(durations) if durations else None,
            "max_duration": np.max(durations) if durations else None,
            "p95_duration": np.percentile(durations, 95) if durations else None,
            "total_operations": len(ops),
            "failed_operations": len(ops) - len(successful_ops)
        }


def performance_monitor(operation_name: str = None):
    """Decorator for performance monitoring."""
    def decorator(func):
        actual_operation_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = get_performance_profiler()
            operation_id = f"{actual_operation_name}_{threading.get_ident()}_{time.time()}"
            
            profiler.start_operation(operation_id, actual_operation_name)
            
            try:
                result = func(*args, **kwargs)
                profiler.end_operation(operation_id, success=True)
                return result
            except Exception as e:
                profiler.end_operation(operation_id, success=False, error_message=str(e))
                raise
                
        return wrapper
    return decorator


class ConnectionPool:
    """Generic connection pool for managing resources."""
    
    def __init__(self, create_connection: Callable, max_connections: int = 10,
                 connection_timeout: float = 30.0, idle_timeout: float = 300.0):
        """Initialize connection pool."""
        self.create_connection = create_connection
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.idle_timeout = idle_timeout
        
        self._pool = Queue(maxsize=max_connections)
        self._active_connections = weakref.WeakSet()
        self._connection_count = 0
        self._lock = threading.Lock()
        
    def get_connection(self):
        """Get a connection from the pool."""
        with self._lock:
            try:
                # Try to get existing connection
                connection = self._pool.get_nowait()
                # TODO: Check if connection is still valid
                return connection
            except Empty:
                # Create new connection if under limit
                if self._connection_count < self.max_connections:
                    connection = self.create_connection()
                    self._connection_count += 1
                    self._active_connections.add(connection)
                    return connection
                else:
                    # Wait for available connection
                    try:
                        connection = self._pool.get(timeout=self.connection_timeout)
                        return connection
                    except Empty:
                        raise TimeoutError("Connection pool timeout")
                        
    def release_connection(self, connection):
        """Release a connection back to the pool."""
        with self._lock:
            try:
                self._pool.put_nowait(connection)
            except:
                # Pool is full, close connection
                if hasattr(connection, 'close'):
                    connection.close()
                self._connection_count -= 1
                
    def close_all(self):
        """Close all connections in the pool."""
        with self._lock:
            while not self._pool.empty():
                try:
                    connection = self._pool.get_nowait()
                    if hasattr(connection, 'close'):
                        connection.close()
                except Empty:
                    break
                    
            self._connection_count = 0


class BatchProcessor:
    """Batch processor for efficient bulk operations."""
    
    def __init__(self, process_func: Callable, batch_size: int = 100,
                 max_workers: int = None, timeout: float = 300.0):
        """Initialize batch processor."""
        self.process_func = process_func
        self.batch_size = batch_size
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.timeout = timeout
        
    def process_items(self, items: List[Any], progress_callback: Callable = None) -> List[Any]:
        """Process items in batches."""
        if not items:
            return []
            
        # Split into batches
        batches = [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self._process_batch, batch): batch 
                for batch in batches
            }
            
            # Collect results
            completed_batches = 0
            for future in as_completed(future_to_batch, timeout=self.timeout):
                batch = future_to_batch[future]
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                    completed_batches += 1
                    
                    if progress_callback:
                        progress = completed_batches / len(batches)
                        progress_callback(progress)
                        
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    # Continue with other batches
                    
        return results
        
    def _process_batch(self, batch: List[Any]) -> List[Any]:
        """Process a single batch."""
        return [self.process_func(item) for item in batch]


class AsyncBatchProcessor:
    """Asynchronous batch processor."""
    
    def __init__(self, process_coro: Callable, batch_size: int = 100,
                 max_concurrent: int = 10):
        """Initialize async batch processor."""
        self.process_coro = process_coro
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        
    async def process_items(self, items: List[Any]) -> List[Any]:
        """Process items asynchronously in batches."""
        if not items:
            return []
            
        # Split into batches
        batches = [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]
        
        # Process batches with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_batch_with_semaphore(batch):
            async with semaphore:
                return await self._process_batch(batch)
                
        # Execute all batches
        tasks = [process_batch_with_semaphore(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)
            
        return results
        
    async def _process_batch(self, batch: List[Any]) -> List[Any]:
        """Process a single batch asynchronously."""
        tasks = [self.process_coro(item) for item in batch]
        return await asyncio.gather(*tasks)


class CircuitCompilationCache:
    """Cache for compiled quantum circuits."""
    
    def __init__(self, cache_manager: CacheManager = None, max_size: int = 1000):
        """Initialize circuit compilation cache."""
        self.cache_manager = cache_manager or CacheManager()
        self.max_size = max_size
        self._local_cache = {}
        
    def get_cache_key(self, circuit: Dict[str, Any], backend: str) -> str:
        """Generate cache key for circuit + backend combination."""
        circuit_str = json.dumps(circuit, sort_keys=True, separators=(',', ':'))
        key_data = f"circuit:{backend}:{circuit_str}"
        return hashlib.sha256(key_data.encode()).hexdigest()
        
    def get_compiled_circuit(self, circuit: Dict[str, Any], backend: str) -> Optional[Dict[str, Any]]:
        """Get compiled circuit from cache."""
        cache_key = self.get_cache_key(circuit, backend)
        
        # Try local cache first
        if cache_key in self._local_cache:
            return self._local_cache[cache_key]
            
        # Try distributed cache
        try:
            cached_data = self.cache_manager.get(f"compiled_circuit:{cache_key}")
            if cached_data:
                # Store in local cache
                self._local_cache[cache_key] = cached_data
                self._enforce_local_cache_size()
                return cached_data
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            
        return None
        
    def store_compiled_circuit(self, circuit: Dict[str, Any], backend: str, 
                              compiled_circuit: Dict[str, Any], ttl: int = 3600):
        """Store compiled circuit in cache."""
        cache_key = self.get_cache_key(circuit, backend)
        
        # Store in local cache
        self._local_cache[cache_key] = compiled_circuit
        self._enforce_local_cache_size()
        
        # Store in distributed cache
        try:
            self.cache_manager.set(f"compiled_circuit:{cache_key}", compiled_circuit, ttl=ttl)
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
            
    def _enforce_local_cache_size(self):
        """Enforce local cache size limit."""
        if len(self._local_cache) > self.max_size:
            # Remove oldest entries (simple FIFO)
            to_remove = len(self._local_cache) - self.max_size
            keys_to_remove = list(self._local_cache.keys())[:to_remove]
            for key in keys_to_remove:
                del self._local_cache[key]
                
    def clear_cache(self):
        """Clear all cached circuits."""
        self._local_cache.clear()
        try:
            # Clear distributed cache (pattern-based)
            self.cache_manager.delete_pattern("compiled_circuit:*")
        except Exception as e:
            logger.warning(f"Cache clearing failed: {e}")


class QuantumJobScheduler:
    """Scheduler for quantum job execution with optimization."""
    
    def __init__(self, max_concurrent_jobs: int = 5):
        """Initialize quantum job scheduler."""
        self.max_concurrent_jobs = max_concurrent_jobs
        self.job_queue = Queue()
        self.active_jobs = {}
        self.completed_jobs = []
        self.scheduler_thread = None
        self.running = False
        
    def submit_job(self, job_id: str, job_func: Callable, priority: int = 0, 
                   backend_preference: List[str] = None, **kwargs) -> str:
        """Submit a quantum job for execution."""
        job = {
            "job_id": job_id,
            "job_func": job_func,
            "priority": priority,
            "backend_preference": backend_preference or [],
            "submit_time": datetime.utcnow(),
            "kwargs": kwargs
        }
        
        self.job_queue.put((priority, job))
        logger.info(f"Job submitted: {job_id}")
        return job_id
        
    def start_scheduler(self):
        """Start the job scheduler."""
        if self.running:
            return
            
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("Quantum job scheduler started")
        
    def stop_scheduler(self):
        """Stop the job scheduler."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=10)
        logger.info("Quantum job scheduler stopped")
        
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            try:
                # Check if we can start new jobs
                if len(self.active_jobs) < self.max_concurrent_jobs:
                    try:
                        priority, job = self.job_queue.get(timeout=1.0)
                        self._start_job(job)
                    except Empty:
                        continue
                        
                # Check for completed jobs
                self._check_completed_jobs()
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                
    def _start_job(self, job: Dict[str, Any]):
        """Start executing a job."""
        job_id = job["job_id"]
        
        def job_wrapper():
            try:
                result = job["job_func"](**job["kwargs"])
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Job {job_id} failed: {e}")
                return {"success": False, "error": str(e)}
                
        # Start job in thread
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(job_wrapper)
        
        self.active_jobs[job_id] = {
            "job": job,
            "future": future,
            "executor": executor,
            "start_time": datetime.utcnow()
        }
        
        logger.info(f"Job started: {job_id}")
        
    def _check_completed_jobs(self):
        """Check for completed jobs."""
        completed_job_ids = []
        
        for job_id, job_info in self.active_jobs.items():
            if job_info["future"].done():
                completed_job_ids.append(job_id)
                
        # Process completed jobs
        for job_id in completed_job_ids:
            job_info = self.active_jobs.pop(job_id)
            
            try:
                result = job_info["future"].result()
                end_time = datetime.utcnow()
                duration = (end_time - job_info["start_time"]).total_seconds()
                
                completed_job = {
                    "job_id": job_id,
                    "start_time": job_info["start_time"],
                    "end_time": end_time,
                    "duration": duration,
                    "result": result
                }
                
                self.completed_jobs.append(completed_job)
                logger.info(f"Job completed: {job_id} ({duration:.2f}s)")
                
            except Exception as e:
                logger.error(f"Error retrieving job result for {job_id}: {e}")
            finally:
                job_info["executor"].shutdown()
                
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a job."""
        # Check active jobs
        if job_id in self.active_jobs:
            job_info = self.active_jobs[job_id]
            return {
                "status": "running",
                "start_time": job_info["start_time"].isoformat(),
                "duration": (datetime.utcnow() - job_info["start_time"]).total_seconds()
            }
            
        # Check completed jobs
        for completed_job in self.completed_jobs:
            if completed_job["job_id"] == job_id:
                return {
                    "status": "completed",
                    "start_time": completed_job["start_time"].isoformat(),
                    "end_time": completed_job["end_time"].isoformat(),
                    "duration": completed_job["duration"],
                    "success": completed_job["result"]["success"],
                    "result": completed_job["result"]
                }
                
        # Check queue
        queue_items = []
        temp_queue = Queue()
        
        while not self.job_queue.empty():
            try:
                item = self.job_queue.get_nowait()
                queue_items.append(item)
                temp_queue.put(item)
            except Empty:
                break
                
        # Restore queue
        while not temp_queue.empty():
            self.job_queue.put(temp_queue.get())
            
        for priority, job in queue_items:
            if job["job_id"] == job_id:
                return {
                    "status": "queued",
                    "submit_time": job["submit_time"].isoformat(),
                    "priority": priority,
                    "queue_position": queue_items.index((priority, job))
                }
                
        return {"status": "not_found"}


class LoadBalancer:
    """Load balancer for quantum backends."""
    
    def __init__(self):
        """Initialize load balancer."""
        self.backend_stats = {}
        self.backend_weights = {}
        
    def update_backend_stats(self, backend: str, response_time: float,
                           success: bool, queue_length: int = 0):
        """Update backend statistics."""
        if backend not in self.backend_stats:
            self.backend_stats[backend] = {
                "total_requests": 0,
                "successful_requests": 0,
                "avg_response_time": 0.0,
                "queue_length": 0,
                "last_update": datetime.utcnow()
            }
            
        stats = self.backend_stats[backend]
        stats["total_requests"] += 1
        
        if success:
            stats["successful_requests"] += 1
            
        # Update average response time (exponential moving average)
        alpha = 0.1
        stats["avg_response_time"] = (
            alpha * response_time + (1 - alpha) * stats["avg_response_time"]
        )
        
        stats["queue_length"] = queue_length
        stats["last_update"] = datetime.utcnow()
        
        # Update weight based on performance
        self._update_backend_weight(backend)
        
    def _update_backend_weight(self, backend: str):
        """Update backend weight based on performance."""
        stats = self.backend_stats[backend]
        
        if stats["total_requests"] == 0:
            weight = 1.0
        else:
            success_rate = stats["successful_requests"] / stats["total_requests"]
            response_time_factor = 1.0 / max(stats["avg_response_time"], 0.001)
            queue_factor = 1.0 / max(stats["queue_length"] + 1, 1)
            
            weight = success_rate * response_time_factor * queue_factor
            
        self.backend_weights[backend] = weight
        
    def select_backend(self, available_backends: List[str]) -> str:
        """Select optimal backend based on load balancing."""
        if not available_backends:
            raise ValueError("No available backends")
            
        if len(available_backends) == 1:
            return available_backends[0]
            
        # Calculate weighted scores
        backend_scores = {}
        for backend in available_backends:
            weight = self.backend_weights.get(backend, 1.0)
            
            # Add some randomness to prevent all requests going to one backend
            random_factor = np.random.uniform(0.8, 1.2)
            backend_scores[backend] = weight * random_factor
            
        # Select backend with highest score
        best_backend = max(backend_scores.items(), key=lambda x: x[1])[0]
        return best_backend
        
    def get_backend_stats(self) -> Dict[str, Any]:
        """Get current backend statistics."""
        return {
            backend: {
                **stats,
                "weight": self.backend_weights.get(backend, 1.0),
                "last_update": stats["last_update"].isoformat()
            }
            for backend, stats in self.backend_stats.items()
        }


# Global instances
_global_performance_profiler: Optional[PerformanceProfiler] = None
_global_circuit_cache: Optional[CircuitCompilationCache] = None
_global_job_scheduler: Optional[QuantumJobScheduler] = None
_global_load_balancer: Optional[LoadBalancer] = None


def get_performance_profiler() -> PerformanceProfiler:
    """Get global performance profiler."""
    global _global_performance_profiler
    if _global_performance_profiler is None:
        _global_performance_profiler = PerformanceProfiler()
    return _global_performance_profiler


def get_circuit_cache() -> CircuitCompilationCache:
    """Get global circuit compilation cache."""
    global _global_circuit_cache
    if _global_circuit_cache is None:
        _global_circuit_cache = CircuitCompilationCache()
    return _global_circuit_cache


def get_job_scheduler() -> QuantumJobScheduler:
    """Get global quantum job scheduler."""
    global _global_job_scheduler
    if _global_job_scheduler is None:
        _global_job_scheduler = QuantumJobScheduler()
        _global_job_scheduler.start_scheduler()
    return _global_job_scheduler


def get_load_balancer() -> LoadBalancer:
    """Get global load balancer."""
    global _global_load_balancer
    if _global_load_balancer is None:
        _global_load_balancer = LoadBalancer()
    return _global_load_balancer


# Optimization utilities
def optimize_numpy_operations():
    """Optimize NumPy operations for better performance."""
    try:
        import numpy as np
        
        # Set optimal thread count for NumPy operations
        cpu_count = os.cpu_count() or 1
        optimal_threads = min(cpu_count, 8)  # Don't use too many threads
        
        os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
        os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_threads)
        
        logger.info(f"NumPy optimization: set thread count to {optimal_threads}")
        
    except Exception as e:
        logger.warning(f"NumPy optimization failed: {e}")


def auto_tune_batch_sizes(operation_func: Callable, test_data: List[Any],
                         min_batch_size: int = 1, max_batch_size: int = 1000,
                         target_time: float = 1.0) -> int:
    """Automatically tune batch size for optimal performance."""
    
    def test_batch_size(batch_size: int) -> float:
        """Test performance of a specific batch size."""
        test_batch = test_data[:min(batch_size, len(test_data))]
        
        start_time = time.time()
        try:
            for item in test_batch:
                operation_func(item)
            duration = time.time() - start_time
            return duration / len(test_batch)  # Time per item
        except Exception:
            return float('inf')  # Invalid batch size
            
    # Binary search for optimal batch size
    best_batch_size = min_batch_size
    best_time_per_item = float('inf')
    
    for batch_size in [1, 10, 50, 100, 500, 1000]:
        if batch_size > max_batch_size or batch_size > len(test_data):
            continue
            
        time_per_item = test_batch_size(batch_size)
        total_time = time_per_item * batch_size
        
        # Prefer batch sizes that get close to target time
        if abs(total_time - target_time) < abs(best_time_per_item * best_batch_size - target_time):
            best_batch_size = batch_size
            best_time_per_item = time_per_item
            
    logger.info(f"Auto-tuned batch size: {best_batch_size} (avg {best_time_per_item:.4f}s per item)")
    return best_batch_size