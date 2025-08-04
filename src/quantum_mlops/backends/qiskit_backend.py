"""Qiskit quantum backend implementation."""

import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from .base import (
    QuantumBackend, 
    QuantumJob, 
    CircuitResult, 
    JobStatus,
    BackendConnectionError,
    CircuitExecutionError,
)

try:
    from qiskit import QuantumCircuit, transpile, Aer
    from qiskit.circuit import Parameter
    from qiskit.primitives import Sampler, Estimator
    from qiskit.quantum_info import SparsePauliOp
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Session
    from qiskit_ibm_runtime.fake_provider import FakeProvider
    IBM_RUNTIME_AVAILABLE = True
except ImportError:
    IBM_RUNTIME_AVAILABLE = False


class QiskitBackend(QuantumBackend):
    """Qiskit quantum computing backend supporting local simulators and IBM Quantum."""
    
    def __init__(
        self, 
        backend_name: str = "qasm_simulator",
        use_ibm_runtime: bool = False,
        **config: Any
    ):
        """Initialize Qiskit backend.
        
        Args:
            backend_name: Qiskit backend name 
            use_ibm_runtime: Whether to use IBM Quantum Runtime
            **config: Backend configuration (shots, optimization_level, etc.)
        """
        super().__init__(f"qiskit_{backend_name}", **config)
        
        if not QISKIT_AVAILABLE:
            raise ImportError(
                "Qiskit is not available. Install it with: "
                "pip install qiskit qiskit-machine-learning"
            )
            
        self.backend_name = backend_name
        self.use_ibm_runtime = use_ibm_runtime
        self.backend = None
        self.service = None
        self.session = None
        self._active_jobs: Dict[str, QuantumJob] = {}
        
        # Default configuration
        self.shots = self.config.get("shots", 1024)
        self.optimization_level = self.config.get("optimization_level", 1)
        
    def connect(self) -> None:
        """Establish connection to Qiskit backend."""
        try:
            if self.use_ibm_runtime:
                if not IBM_RUNTIME_AVAILABLE:
                    raise ImportError(
                        "IBM Runtime is not available. Install it with: "
                        "pip install qiskit-ibm-runtime"
                    )
                
                # Connect to IBM Quantum Runtime
                token = self.config.get("ibm_token")
                if token:
                    self.service = QiskitRuntimeService(token=token)
                else:
                    # Try to load from saved credentials
                    try:
                        self.service = QiskitRuntimeService()
                    except Exception:
                        # Use fake provider for testing if no credentials
                        fake_provider = FakeProvider()
                        self.backend = fake_provider.get_backend("fake_manila")
                        self._is_connected = True
                        return
                        
                # Get IBM backend
                if self.backend_name.startswith("fake_"):
                    fake_provider = FakeProvider()
                    self.backend = fake_provider.get_backend(self.backend_name)
                else:
                    available_backends = self.service.backends()
                    backend_names = [b.name for b in available_backends]
                    
                    if self.backend_name not in backend_names:
                        # Fallback to simulator
                        self.backend_name = "ibmq_qasm_simulator"
                        if self.backend_name not in backend_names:
                            self.backend = available_backends[0]  # Use first available
                        else:
                            self.backend = self.service.backend(self.backend_name)
                    else:
                        self.backend = self.service.backend(self.backend_name)
                        
            else:
                # Use local Aer simulators
                if self.backend_name == "qasm_simulator":
                    self.backend = Aer.get_backend("qasm_simulator")
                elif self.backend_name == "statevector_simulator":
                    self.backend = Aer.get_backend("statevector_simulator")
                elif self.backend_name == "unitary_simulator":
                    self.backend = Aer.get_backend("unitary_simulator")
                else:
                    # Default to qasm_simulator
                    self.backend = Aer.get_backend("qasm_simulator")
                    
            self._is_connected = True
            
        except Exception as e:
            raise BackendConnectionError(f"Failed to connect to Qiskit backend: {e}")
            
    def disconnect(self) -> None:
        """Close connection to Qiskit backend."""
        if self.session:
            self.session.close()
            self.session = None
        self.backend = None
        self.service = None
        self._is_connected = False
        
    def is_available(self) -> bool:
        """Check if Qiskit backend is available."""
        if not self._is_connected:
            return False
            
        try:
            # For IBM backends, check status
            if hasattr(self.backend, 'status'):
                status = self.backend.status()
                return status.operational
            return True
        except Exception:
            return False
            
    def get_device_properties(self) -> Dict[str, Any]:
        """Get properties of the Qiskit backend."""
        self.ensure_connected()
        
        properties = {
            "name": self.backend_name,
            "provider": "IBM" if self.use_ibm_runtime else "Aer",
            "n_qubits": self.backend.num_qubits if hasattr(self.backend, 'num_qubits') else 32,
            "simulator": not self.use_ibm_runtime or self.backend_name.endswith("_simulator"),
        }
        
        # Add backend-specific properties
        if hasattr(self.backend, 'configuration'):
            config = self.backend.configuration()
            properties.update({
                "coupling_map": getattr(config, 'coupling_map', None),
                "basis_gates": getattr(config, 'basis_gates', []),
                "max_shots": getattr(config, 'max_shots', 8192),
                "memory": getattr(config, 'memory', False),
            })
            
        return properties
        
    def compile_circuit(self, circuit: Dict[str, Any]) -> Dict[str, Any]:
        """Compile circuit for Qiskit execution."""
        gates = circuit.get("gates", [])
        n_qubits = circuit.get("n_qubits", len(gates))
        parameters = circuit.get("parameters", [])
        
        # Create Qiskit quantum circuit
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # Add parameters
        param_objects = []
        for i, param_name in enumerate(circuit.get("parameter_names", [])):
            param_objects.append(Parameter(param_name))
            
        param_idx = 0
        
        # Add gates to circuit
        for gate in gates:
            gate_type = gate.get("type", "").lower()
            wires = gate.get("wires", gate.get("qubit", 0))
            
            if isinstance(wires, int):
                wires = [wires]
                
            if gate_type == "rx":
                angle = gate.get("angle")
                if angle is None and param_idx < len(param_objects):
                    angle = param_objects[param_idx]
                    param_idx += 1
                elif angle is None:
                    angle = 0
                qc.rx(angle, wires[0])
                
            elif gate_type == "ry":
                angle = gate.get("angle")
                if angle is None and param_idx < len(param_objects):
                    angle = param_objects[param_idx]
                    param_idx += 1
                elif angle is None:
                    angle = 0
                qc.ry(angle, wires[0])
                
            elif gate_type == "rz":
                angle = gate.get("angle")
                if angle is None and param_idx < len(param_objects):
                    angle = param_objects[param_idx]
                    param_idx += 1
                elif angle is None:
                    angle = 0
                qc.rz(angle, wires[0])
                
            elif gate_type in ["h", "hadamard"]:
                qc.h(wires[0])
                
            elif gate_type in ["x", "pauli_x"]:
                qc.x(wires[0])
                
            elif gate_type in ["y", "pauli_y"]:
                qc.y(wires[0])
                
            elif gate_type in ["z", "pauli_z"]:
                qc.z(wires[0])
                
            elif gate_type in ["cnot", "cx"]:
                control = gate.get("control", wires[0])
                target = gate.get("target", wires[1] if len(wires) > 1 else wires[0] + 1)
                qc.cx(control, target)
                
        # Add measurements if specified
        measurements = circuit.get("measurements", [{"type": "computational", "qubits": list(range(n_qubits))}])
        
        for measurement in measurements:
            if measurement.get("type") == "computational":
                qubits = measurement.get("qubits", list(range(n_qubits)))
                for qubit in qubits:
                    if qubit < n_qubits:
                        qc.measure(qubit, qubit)
                        
        # Transpile for backend
        try:
            transpiled_qc = transpile(
                qc, 
                backend=self.backend,
                optimization_level=self.optimization_level
            )
        except Exception:
            # Fallback: use circuit as is
            transpiled_qc = qc
            
        return {
            "original": circuit,
            "qiskit_circuit": transpiled_qc,
            "parameters": param_objects,
            "n_qubits": n_qubits,
        }
        
    def submit_job(
        self, 
        circuits: List[Dict[str, Any]], 
        shots: int = 1024
    ) -> QuantumJob:
        """Submit circuits for execution on Qiskit backend."""
        self.ensure_connected()
        
        job_id = str(uuid.uuid4())
        
        # Create job
        job = QuantumJob(
            job_id=job_id,
            backend_name=self.name,
            status=JobStatus.QUEUED,
            circuits=circuits,
            shots=shots,
            created_at=time.time(),
        )
        
        self._active_jobs[job_id] = job
        
        try:
            job.status = JobStatus.RUNNING
            results = []
            
            for i, circuit in enumerate(circuits):
                start_time = time.time()
                
                # Compile circuit
                compiled_circuit = self.compile_circuit(circuit)
                qc = compiled_circuit["qiskit_circuit"]
                param_objects = compiled_circuit["parameters"]
                
                # Execute circuit
                parameters = circuit.get("parameters", [])
                
                if param_objects and parameters:
                    # Bind parameters
                    if isinstance(parameters[0], (list, np.ndarray)):
                        # Multiple parameter sets - execute multiple times
                        all_counts = {}
                        expectation_values = []
                        
                        for param_set in parameters:
                            param_dict = {param_objects[j]: param_set[j] 
                                        for j in range(min(len(param_objects), len(param_set)))}
                            bound_qc = qc.bind_parameters(param_dict)
                            
                            # Execute circuit
                            if hasattr(self.backend, 'run'):
                                qiskit_job = self.backend.run(bound_qc, shots=shots)
                                qiskit_result = qiskit_job.result()
                                counts = qiskit_result.get_counts()
                            else:
                                # Use primitives for newer Qiskit versions
                                sampler = Sampler()
                                job_result = sampler.run([bound_qc], shots=shots).result()
                                counts = job_result.quasi_dists[0]
                                # Convert quasi-distribution to counts
                                counts = {format(k, f'0{qc.num_clbits}b'): int(v * shots) 
                                        for k, v in counts.items()}
                            
                            # Merge counts
                            for state, count in counts.items():
                                all_counts[state] = all_counts.get(state, 0) + count
                                
                            # Calculate expectation value (simplified)
                            prob_0 = counts.get('0' * qc.num_clbits, 0) / shots
                            expectation_values.append(2 * prob_0 - 1)  # Map to [-1, 1]
                            
                        expectation_value = np.mean(expectation_values)
                        counts = all_counts
                        
                    else:
                        # Single parameter set
                        param_dict = {param_objects[j]: parameters[j] 
                                    for j in range(min(len(param_objects), len(parameters)))}
                        bound_qc = qc.bind_parameters(param_dict)
                        
                        # Execute circuit
                        if hasattr(self.backend, 'run'):
                            qiskit_job = self.backend.run(bound_qc, shots=shots)
                            qiskit_result = qiskit_job.result()
                            counts = qiskit_result.get_counts()
                        else:
                            # Use primitives
                            sampler = Sampler()
                            job_result = sampler.run([bound_qc], shots=shots).result()
                            counts = job_result.quasi_dists[0]
                            counts = {format(k, f'0{qc.num_clbits}b'): int(v * shots) 
                                    for k, v in counts.items()}
                        
                        # Calculate expectation value
                        prob_0 = counts.get('0' * qc.num_clbits, 0) / shots
                        expectation_value = 2 * prob_0 - 1
                        
                else:
                    # No parameters - execute directly
                    if hasattr(self.backend, 'run'):
                        qiskit_job = self.backend.run(qc, shots=shots)
                        qiskit_result = qiskit_job.result()
                        counts = qiskit_result.get_counts()
                    else:
                        # Use primitives
                        sampler = Sampler()
                        job_result = sampler.run([qc], shots=shots).result()
                        counts = job_result.quasi_dists[0]
                        counts = {format(k, f'0{qc.num_clbits}b'): int(v * shots) 
                                for k, v in counts.items()}
                    
                    prob_0 = counts.get('0' * qc.num_clbits, 0) / shots
                    expectation_value = 2 * prob_0 - 1
                
                execution_time = time.time() - start_time
                
                result = CircuitResult(
                    circuit_id=f"{job_id}_circuit_{i}",
                    counts=counts,
                    expectation_value=expectation_value,
                    execution_time=execution_time,
                    shots=shots,
                    metadata={"backend": self.name, "qiskit_backend": self.backend_name}
                )
                
                results.append(result)
                
            # Complete job
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            job.results = results
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = time.time()
            
        return job
        
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get status of submitted job."""
        if job_id not in self._active_jobs:
            raise ValueError(f"Job {job_id} not found")
            
        return self._active_jobs[job_id].status
        
    def get_job_results(self, job_id: str) -> List[CircuitResult]:
        """Retrieve results from completed job."""
        if job_id not in self._active_jobs:
            raise ValueError(f"Job {job_id} not found")
            
        job = self._active_jobs[job_id]
        
        if job.status != JobStatus.COMPLETED:
            raise ValueError(f"Job {job_id} is not completed (status: {job.status})")
            
        return job.results or []
        
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        if job_id not in self._active_jobs:
            return False
            
        job = self._active_jobs[job_id]
        
        if job.status in [JobStatus.QUEUED, JobStatus.RUNNING]:
            job.status = JobStatus.CANCELLED
            job.completed_at = time.time()
            return True
            
        return False