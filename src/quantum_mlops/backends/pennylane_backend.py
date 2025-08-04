"""PennyLane quantum backend implementation."""

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
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False


class PennyLaneBackend(QuantumBackend):
    """PennyLane quantum computing backend."""
    
    def __init__(self, device_name: str = "default.qubit", **config: Any):
        """Initialize PennyLane backend.
        
        Args:
            device_name: PennyLane device name (e.g., 'default.qubit', 'lightning.qubit')
            **config: Device configuration (wires, shots, etc.)
        """
        super().__init__(f"pennylane_{device_name}", **config)
        
        if not PENNYLANE_AVAILABLE:
            raise ImportError(
                "PennyLane is not available. Install it with: "
                "pip install pennylane pennylane-lightning"
            )
            
        self.device_name = device_name
        self.device = None
        self._active_jobs: Dict[str, QuantumJob] = {}
        
    def connect(self) -> None:
        """Establish connection to PennyLane device."""
        try:
            # Get device parameters from config
            wires = self.config.get("wires", 4)
            shots = self.config.get("shots", 1024)
            
            # Create PennyLane device
            self.device = qml.device(
                self.device_name,
                wires=wires,
                shots=shots if shots > 0 else None,  # None for analytic mode
                **{k: v for k, v in self.config.items() 
                   if k not in ["wires", "shots"]}
            )
            
            self._is_connected = True
            
        except Exception as e:
            raise BackendConnectionError(f"Failed to connect to PennyLane device: {e}")
            
    def disconnect(self) -> None:
        """Close connection to PennyLane device."""
        self.device = None
        self._is_connected = False
        
    def is_available(self) -> bool:
        """Check if PennyLane backend is available."""
        return self._is_connected and self.device is not None
        
    def get_device_properties(self) -> Dict[str, Any]:
        """Get properties of the PennyLane device."""
        self.ensure_connected()
        
        return {
            "name": self.device_name,
            "wires": self.device.num_wires,
            "shots": getattr(self.device, "shots", None),
            "supports_finite_shots": True,
            "supports_analytic": self.device_name.startswith("default."),
            "operations": list(self.device.operations),
            "observables": list(self.device.observables),
            "measurement_processes": list(getattr(self.device, "measurement_processes", [])),
        }
        
    def compile_circuit(self, circuit: Dict[str, Any]) -> Dict[str, Any]:
        """Compile circuit for PennyLane execution.
        
        Args:
            circuit: Circuit description with gates and measurements
            
        Returns:
            Compiled circuit ready for execution
        """
        # PennyLane circuits are defined as functions, so we create
        # a compiled representation that can be executed
        compiled_circuit = {
            "original": circuit,
            "gates": circuit.get("gates", []),
            "measurements": circuit.get("measurements", []),
            "parameters": circuit.get("parameters", []),
            "n_qubits": circuit.get("n_qubits", self.device.num_wires),
        }
        
        # Validate circuit can run on this device
        n_qubits = compiled_circuit["n_qubits"]
        if n_qubits > self.device.num_wires:
            raise CircuitExecutionError(
                f"Circuit requires {n_qubits} qubits but device only has {self.device.num_wires}"
            )
            
        return compiled_circuit
        
    def _create_qnode_from_circuit(self, circuit: Dict[str, Any]) -> Any:
        """Create a PennyLane QNode from circuit description."""
        gates = circuit.get("gates", [])
        measurements = circuit.get("measurements", [])
        parameters = circuit.get("parameters", [])
        
        def quantum_function(*params):
            """Quantum function to be turned into QNode."""
            param_idx = 0
            
            # Apply gates
            for gate in gates:
                gate_type = gate.get("type", "").lower()
                wires = gate.get("wires", gate.get("qubit", 0))
                
                if gate_type == "rx":
                    angle = gate.get("angle", params[param_idx] if param_idx < len(params) else 0)
                    if "angle" not in gate and param_idx < len(params):
                        param_idx += 1
                    qml.RX(angle, wires=wires)
                    
                elif gate_type == "ry":
                    angle = gate.get("angle", params[param_idx] if param_idx < len(params) else 0)
                    if "angle" not in gate and param_idx < len(params):
                        param_idx += 1
                    qml.RY(angle, wires=wires)
                    
                elif gate_type == "rz":
                    angle = gate.get("angle", params[param_idx] if param_idx < len(params) else 0)
                    if "angle" not in gate and param_idx < len(params):
                        param_idx += 1
                    qml.RZ(angle, wires=wires)
                    
                elif gate_type in ["h", "hadamard"]:
                    qml.Hadamard(wires=wires)
                    
                elif gate_type in ["x", "pauli_x"]:
                    qml.PauliX(wires=wires)
                    
                elif gate_type in ["y", "pauli_y"]:
                    qml.PauliY(wires=wires)
                    
                elif gate_type in ["z", "pauli_z"]:
                    qml.PauliZ(wires=wires)
                    
                elif gate_type in ["cnot", "cx"]:
                    control = gate.get("control", wires[0] if isinstance(wires, list) else 0)
                    target = gate.get("target", wires[1] if isinstance(wires, list) else 1)
                    qml.CNOT(wires=[control, target])
                    
            # Apply measurements
            if measurements:
                results = []
                for measurement in measurements:
                    meas_type = measurement.get("type", "expectation")
                    wires = measurement.get("wires", 0)
                    observable = measurement.get("observable", "Z")
                    
                    if meas_type == "expectation":
                        if observable == "Z":
                            results.append(qml.expval(qml.PauliZ(wires)))
                        elif observable == "X":
                            results.append(qml.expval(qml.PauliX(wires)))
                        elif observable == "Y":
                            results.append(qml.expval(qml.PauliY(wires)))
                            
                return results if len(results) > 1 else (results[0] if results else qml.expval(qml.PauliZ(0)))
            else:
                # Default measurement
                return qml.expval(qml.PauliZ(0))
                
        return qml.QNode(quantum_function, self.device)
        
    def submit_job(
        self, 
        circuits: List[Dict[str, Any]], 
        shots: int = 1024
    ) -> QuantumJob:
        """Submit circuits for execution on PennyLane device."""
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
            # Execute circuits immediately (PennyLane is typically local)
            job.status = JobStatus.RUNNING
            results = []
            
            for i, circuit in enumerate(circuits):
                start_time = time.time()
                
                # Compile circuit
                compiled_circuit = self.compile_circuit(circuit)
                
                # Create QNode
                qnode = self._create_qnode_from_circuit(compiled_circuit)
                
                # Execute circuit
                parameters = compiled_circuit.get("parameters", [])
                if parameters:
                    if isinstance(parameters[0], (list, np.ndarray)):
                        # Multiple parameter sets
                        expectation_values = []
                        for param_set in parameters:
                            expectation_values.append(float(qnode(*param_set)))
                        expectation_value = np.mean(expectation_values)
                    else:
                        # Single parameter set
                        expectation_value = float(qnode(*parameters))
                else:
                    # No parameters
                    expectation_value = float(qnode())
                
                execution_time = time.time() - start_time
                
                # Generate mock counts for shot-based simulation
                if shots > 0:
                    prob_0 = (1 + expectation_value) / 2  # Convert expectation to probability
                    prob_1 = 1 - prob_0
                    counts_0 = int(prob_0 * shots)
                    counts_1 = shots - counts_0
                    counts = {"0": counts_0, "1": counts_1}
                else:
                    counts = {"expectation": 1}  # Analytic mode
                
                result = CircuitResult(
                    circuit_id=f"{job_id}_circuit_{i}",
                    counts=counts,
                    expectation_value=expectation_value,
                    execution_time=execution_time,
                    shots=shots,
                    metadata={"backend": self.name, "device": self.device_name}
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