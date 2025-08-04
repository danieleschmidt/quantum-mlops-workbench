"""Quantum circuit executor with backend integration."""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..core import QuantumDevice
from .backend_manager import BackendManager
from .base import QuantumJob, CircuitResult, JobStatus


logger = logging.getLogger(__name__)


class QuantumExecutor:
    """High-level quantum circuit executor."""
    
    def __init__(self, backend_manager: Optional[BackendManager] = None):
        """Initialize quantum executor.
        
        Args:
            backend_manager: Backend manager instance. If None, creates a new one.
        """
        self.backend_manager = backend_manager or BackendManager()
        self._setup_default_backends()
        
    def _setup_default_backends(self) -> None:
        """Setup default backends for immediate use."""
        try:
            # Try to create basic simulators that should always work
            from .pennylane_backend import PennyLaneBackend
            from .qiskit_backend import QiskitBackend
            
            # PennyLane default qubit simulator
            try:
                pennylane_backend = PennyLaneBackend("default.qubit", wires=8, shots=1024)
                self.backend_manager.register_backend(pennylane_backend)
            except Exception as e:
                logger.warning(f"Failed to setup PennyLane backend: {e}")
                
            # Qiskit Aer simulator
            try:
                qiskit_backend = QiskitBackend("qasm_simulator", shots=1024)
                self.backend_manager.register_backend(qiskit_backend)
            except Exception as e:
                logger.warning(f"Failed to setup Qiskit backend: {e}")
                
        except Exception as e:
            logger.warning(f"Failed to setup default backends: {e}")
            
    def create_backend(self, device: QuantumDevice, **config: Any) -> str:
        """Create and register a new backend.
        
        Args:
            device: Device type to create
            **config: Backend configuration
            
        Returns:
            Backend name for future reference
        """
        backend = self.backend_manager.create_backend(device, **config)
        return backend.name
        
    def execute(
        self,
        circuits: Union[Dict[str, Any], List[Dict[str, Any]]],
        backend_names: Optional[List[str]] = None,
        shots: int = 1024,
        optimization_target: str = "cost_performance"
    ) -> Union[CircuitResult, List[CircuitResult]]:
        """Execute quantum circuits.
        
        Args:
            circuits: Single circuit or list of circuits to execute
            backend_names: Preferred backend names (will use auto-selection if None)
            shots: Number of measurement shots
            optimization_target: Backend optimization target
            
        Returns:
            Circuit results (single result if single circuit input)
        """
        # Normalize input
        if isinstance(circuits, dict):
            circuits = [circuits]
            single_circuit = True
        else:
            single_circuit = False
            
        # Validate circuits
        validated_circuits = []
        for circuit in circuits:
            validated_circuit = self.backend_manager.validate_circuit(circuit)
            validated_circuits.append(validated_circuit)
            
        # Select backends if not specified
        if backend_names is None:
            backend_names = self.backend_manager.optimize_backend_selection(
                validated_circuits, optimization_target
            )
            
        if not backend_names:
            # Fallback to default order
            backend_names = self.backend_manager.fallback_order
            
        logger.info(f"Executing {len(validated_circuits)} circuits on backends: {backend_names}")
        
        # Execute with fallback
        job = self.backend_manager.execute_with_fallback(
            validated_circuits, backend_names, shots
        )
        
        if job.status != JobStatus.COMPLETED:
            raise RuntimeError(f"Circuit execution failed: {job.error_message}")
            
        results = job.results or []
        
        # Return single result if single circuit input
        if single_circuit and len(results) == 1:
            return results[0]
        return results
        
    def execute_async(
        self,
        circuits: Union[Dict[str, Any], List[Dict[str, Any]]],
        backend_names: Optional[List[str]] = None,
        shots: int = 1024
    ) -> str:
        """Execute circuits asynchronously.
        
        Args:
            circuits: Single circuit or list of circuits to execute
            backend_names: Preferred backend names
            shots: Number of measurement shots
            
        Returns:
            Job ID for tracking execution
        """
        # For now, execute synchronously and return completed job ID
        # In a full implementation, this would submit to a job queue
        results = self.execute(circuits, backend_names, shots)
        
        if isinstance(results, list):
            job_id = results[0].circuit_id.split("_circuit_")[0]
        else:
            job_id = results.circuit_id.split("_circuit_")[0]
            
        return job_id
        
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get status of an async job."""
        # Try to find job in all backends
        for backend in self.backend_manager.backends.values():
            try:
                return backend.get_job_status(job_id)
            except ValueError:
                continue
                
        raise ValueError(f"Job {job_id} not found in any backend")
        
    def get_job_results(self, job_id: str) -> List[CircuitResult]:
        """Get results from an async job."""
        # Try to find job in all backends
        for backend in self.backend_manager.backends.values():
            try:
                return backend.get_job_results(job_id)
            except ValueError:
                continue
                
        raise ValueError(f"Job {job_id} not found in any backend")
        
    def get_backend_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all backends."""
        return self.backend_manager.get_backend_status()
        
    def list_available_backends(self) -> List[str]:
        """List all available backend names."""
        status = self.get_backend_status()
        return [name for name, info in status.items() if info.get("available", False)]
        
    def estimate_execution_cost(
        self,
        circuits: Union[Dict[str, Any], List[Dict[str, Any]]],
        backend_name: str,
        shots: int = 1024
    ) -> Dict[str, Any]:
        """Estimate cost for circuit execution.
        
        Args:
            circuits: Circuits to estimate cost for
            backend_name: Target backend name
            shots: Number of shots
            
        Returns:
            Cost estimation details
        """
        if isinstance(circuits, dict):
            circuits = [circuits]
            
        backend = self.backend_manager.get_backend(backend_name)
        if backend is None:
            raise ValueError(f"Backend {backend_name} not found")
            
        try:
            properties = backend.get_device_properties()
            
            # Simplified cost estimation
            cost_per_shot = 0.0
            if properties.get("provider") == "AWS":
                cost_per_shot = 0.00035  # AWS Braket approximate cost
            elif properties.get("provider") == "IBM" and not properties.get("simulator"):
                cost_per_shot = 0.0  # IBM Quantum free tier
                
            total_shots = len(circuits) * shots
            total_cost = total_shots * cost_per_shot
            
            return {
                "backend": backend_name,
                "circuits": len(circuits),
                "shots_per_circuit": shots,
                "total_shots": total_shots,
                "cost_per_shot": cost_per_shot,
                "total_cost": total_cost,
                "currency": "USD",
                "estimated_time_minutes": total_shots / 1000 * 0.1,  # Rough estimate
            }
            
        except Exception as e:
            logger.warning(f"Failed to estimate cost for {backend_name}: {e}")
            return {
                "backend": backend_name,
                "error": str(e),
                "total_cost": 0.0,
            }
            
    def benchmark_backends(
        self,
        test_circuit: Optional[Dict[str, Any]] = None,
        backend_names: Optional[List[str]] = None,
        shots: int = 100
    ) -> Dict[str, Dict[str, Any]]:
        """Benchmark backend performance with a test circuit.
        
        Args:
            test_circuit: Circuit to use for benchmarking (uses default if None)
            backend_names: Backends to benchmark (uses all available if None)
            shots: Number of shots for benchmarking
            
        Returns:
            Benchmark results for each backend
        """
        # Default test circuit
        if test_circuit is None:
            test_circuit = {
                "gates": [
                    {"type": "h", "qubit": 0},
                    {"type": "cx", "control": 0, "target": 1},
                    {"type": "rx", "qubit": 0, "angle": np.pi/4},
                    {"type": "ry", "qubit": 1, "angle": np.pi/3},
                ],
                "n_qubits": 2,
                "measurements": [{"type": "computational", "qubits": [0, 1]}]
            }
            
        if backend_names is None:
            backend_names = self.list_available_backends()
            
        results = {}
        
        for backend_name in backend_names:
            try:
                logger.info(f"Benchmarking backend: {backend_name}")
                
                import time
                start_time = time.time()
                
                # Execute test circuit
                result = self.execute(
                    test_circuit,
                    backend_names=[backend_name],
                    shots=shots
                )
                
                execution_time = time.time() - start_time
                
                results[backend_name] = {
                    "success": True,
                    "execution_time": execution_time,
                    "shots": shots,
                    "expectation_value": result.expectation_value,
                    "total_counts": sum(result.counts.values()),
                }
                
            except Exception as e:
                logger.warning(f"Benchmark failed for {backend_name}: {e}")
                results[backend_name] = {
                    "success": False,
                    "error": str(e),
                    "execution_time": None,
                }
                
        return results