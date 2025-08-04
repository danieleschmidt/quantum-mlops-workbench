"""Quantum backend manager with error handling and fallback mechanisms."""

import logging
import time
from typing import Any, Dict, List, Optional, Type, Union

from ..core import QuantumDevice
from .base import (
    QuantumBackend, 
    QuantumJob, 
    CircuitResult, 
    JobStatus,
    BackendConnectionError,
    CircuitExecutionError,
)
from .pennylane_backend import PennyLaneBackend
from .qiskit_backend import QiskitBackend
from .braket_backend import BraketBackend


logger = logging.getLogger(__name__)


class BackendManager:
    """Manager for quantum backends with automatic fallback and error handling."""
    
    def __init__(self):
        """Initialize backend manager."""
        self.backends: Dict[str, QuantumBackend] = {}
        self.fallback_order = [
            "pennylane_default.qubit",
            "qiskit_qasm_simulator", 
            "braket_local"
        ]
        self.retry_attempts = 3
        self.retry_delay = 1.0  # seconds
        
    def register_backend(self, backend: QuantumBackend) -> None:
        """Register a quantum backend."""
        self.backends[backend.name] = backend
        logger.info(f"Registered quantum backend: {backend.name}")
        
    def get_backend(self, name: str) -> Optional[QuantumBackend]:
        """Get backend by name."""
        return self.backends.get(name)
        
    def create_backend(
        self, 
        device: QuantumDevice, 
        **config: Any
    ) -> QuantumBackend:
        """Create and register a backend based on device type."""
        
        if device == QuantumDevice.SIMULATOR:
            # Default to PennyLane simulator
            backend = PennyLaneBackend("default.qubit", **config)
            
        elif device == QuantumDevice.AWS_BRAKET:
            device_arn = config.get("device_arn", "local:braket/braket.local.qubit")
            backend = BraketBackend(device_arn, **config)
            
        elif device == QuantumDevice.IBM_QUANTUM:
            backend_name = config.get("backend", "qasm_simulator")
            use_runtime = config.get("use_ibm_runtime", False)
            backend = QiskitBackend(backend_name, use_runtime, **config)
            
        elif device == QuantumDevice.IONQ:
            # IonQ through Braket
            device_arn = config.get("device_arn", "arn:aws:braket:::device/qpu/ionq/ionQdevice")
            backend = BraketBackend(device_arn, **config)
            
        else:
            raise ValueError(f"Unsupported device type: {device}")
            
        self.register_backend(backend)
        return backend
        
    def connect_with_fallback(self, preferred_backends: List[str]) -> QuantumBackend:
        """Connect to backend with automatic fallback."""
        
        # Try preferred backends first
        all_backends = preferred_backends + [
            name for name in self.fallback_order 
            if name not in preferred_backends
        ]
        
        last_error = None
        
        for backend_name in all_backends:
            try:
                backend = self.backends.get(backend_name)
                if backend is None:
                    # Try to create fallback backends
                    backend = self._create_fallback_backend(backend_name)
                    if backend is None:
                        continue
                        
                logger.info(f"Attempting to connect to backend: {backend_name}")
                backend.connect()
                
                if backend.is_available():
                    logger.info(f"Successfully connected to backend: {backend_name}")
                    return backend
                else:
                    logger.warning(f"Backend {backend_name} is not available")
                    backend.disconnect()
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to connect to backend {backend_name}: {e}")
                
        # If all backends failed, raise the last error
        raise BackendConnectionError(
            f"Failed to connect to any backend. Last error: {last_error}"
        )
        
    def _create_fallback_backend(self, backend_name: str) -> Optional[QuantumBackend]:
        """Create fallback backends on demand."""
        try:
            if backend_name == "pennylane_default.qubit":
                backend = PennyLaneBackend("default.qubit")
                self.register_backend(backend)
                return backend
                
            elif backend_name == "qiskit_qasm_simulator":
                backend = QiskitBackend("qasm_simulator")
                self.register_backend(backend)
                return backend
                
            elif backend_name == "braket_local":
                backend = BraketBackend("local:braket/braket.local.qubit")
                self.register_backend(backend)
                return backend
                
        except Exception as e:
            logger.warning(f"Failed to create fallback backend {backend_name}: {e}")
            
        return None
        
    def execute_with_retry(
        self, 
        backend: QuantumBackend,
        circuits: List[Dict[str, Any]],
        shots: int = 1024
    ) -> QuantumJob:
        """Execute circuits with retry logic."""
        
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                logger.debug(f"Execution attempt {attempt + 1}/{self.retry_attempts}")
                
                # Submit job
                job = backend.submit_job(circuits, shots)
                
                # Wait for completion with timeout
                timeout = 300  # 5 minutes
                start_time = time.time()
                
                while job.status in [JobStatus.QUEUED, JobStatus.RUNNING]:
                    if time.time() - start_time > timeout:
                        backend.cancel_job(job.job_id)
                        raise CircuitExecutionError("Job execution timeout")
                        
                    time.sleep(1)  # Poll every second
                    job.status = backend.get_job_status(job.job_id)
                    
                if job.status == JobStatus.COMPLETED:
                    return job
                elif job.status == JobStatus.FAILED:
                    raise CircuitExecutionError(f"Job failed: {job.error_message}")
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Execution attempt {attempt + 1} failed: {e}")
                
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                    
        raise CircuitExecutionError(f"All retry attempts failed. Last error: {last_error}")
        
    def execute_with_fallback(
        self,
        circuits: List[Dict[str, Any]],
        preferred_backends: List[str],
        shots: int = 1024
    ) -> QuantumJob:
        """Execute circuits with backend fallback."""
        
        # Try to connect to a backend
        backend = self.connect_with_fallback(preferred_backends)
        
        try:
            # Execute with retry
            return self.execute_with_retry(backend, circuits, shots)
            
        except Exception as e:
            logger.error(f"Execution failed on {backend.name}: {e}")
            
            # Try fallback backends
            remaining_backends = [
                name for name in self.fallback_order 
                if name != backend.name
            ]
            
            if remaining_backends:
                logger.info("Attempting fallback to other backends")
                fallback_backend = self.connect_with_fallback(remaining_backends)
                return self.execute_with_retry(fallback_backend, circuits, shots)
            else:
                raise CircuitExecutionError(f"All backends failed: {e}")
                
    def validate_circuit(self, circuit: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize circuit format."""
        
        # Required fields
        if "gates" not in circuit:
            circuit["gates"] = []
            
        if "n_qubits" not in circuit:
            # Infer from gates
            max_qubit = 0
            for gate in circuit["gates"]:
                wires = gate.get("wires", gate.get("qubit", 0))
                if isinstance(wires, int):
                    max_qubit = max(max_qubit, wires)
                elif isinstance(wires, list):
                    max_qubit = max(max_qubit, max(wires))
            circuit["n_qubits"] = max_qubit + 1
            
        # Validate gate types
        valid_gates = {
            "rx", "ry", "rz", "h", "hadamard", "x", "pauli_x", 
            "y", "pauli_y", "z", "pauli_z", "cnot", "cx"
        }
        
        for gate in circuit["gates"]:
            gate_type = gate.get("type", "").lower()
            if gate_type not in valid_gates:
                logger.warning(f"Unknown gate type: {gate_type}")
                
        return circuit
        
    def optimize_backend_selection(
        self,
        circuits: List[Dict[str, Any]],
        optimization_target: str = "cost_performance"
    ) -> List[str]:
        """Select optimal backends based on circuit requirements."""
        
        # Analyze circuit requirements
        max_qubits = max(circuit.get("n_qubits", 1) for circuit in circuits)
        total_gates = sum(len(circuit.get("gates", [])) for circuit in circuits)
        has_parameters = any(circuit.get("parameters") for circuit in circuits)
        
        # Backend scoring
        backend_scores = {}
        
        for name, backend in self.backends.items():
            try:
                props = backend.get_device_properties()
                score = 0
                
                # Qubit capacity score
                if props.get("n_qubits", 0) >= max_qubits:
                    score += 10
                    
                # Performance score
                if props.get("simulator", True):
                    score += 5  # Simulators are faster for development
                    
                # Cost score (simulators are free)
                if props.get("provider") in ["Local", "Aer"]:
                    score += 8
                    
                # Reliability score
                if name in self.fallback_order:
                    score += 3
                    
                backend_scores[name] = score
                
            except Exception:
                # Skip backends that can't provide properties
                continue
                
        # Sort by score
        sorted_backends = sorted(
            backend_scores.keys(), 
            key=lambda x: backend_scores[x], 
            reverse=True
        )
        
        return sorted_backends[:3]  # Return top 3 candidates
        
    def get_backend_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered backends."""
        status = {}
        
        for name, backend in self.backends.items():
            try:
                backend_status = {
                    "connected": backend._is_connected,
                    "available": backend.is_available() if backend._is_connected else False,
                    "properties": backend.get_device_properties() if backend._is_connected else None,
                }
            except Exception as e:
                backend_status = {
                    "connected": False,
                    "available": False, 
                    "error": str(e)
                }
                
            status[name] = backend_status
            
        return status