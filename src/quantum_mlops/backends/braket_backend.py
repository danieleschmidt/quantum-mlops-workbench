"""AWS Braket quantum backend implementation."""

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
    from braket.circuits import Circuit
    from braket.devices import LocalSimulator
    from braket.aws import AwsDevice
    import boto3
    BRAKET_AVAILABLE = True
except ImportError:
    BRAKET_AVAILABLE = False


class BraketBackend(QuantumBackend):
    """AWS Braket quantum computing backend."""
    
    def __init__(
        self, 
        device_arn: str = "local:braket/braket.local.qubit",
        aws_region: str = "us-east-1",
        **config: Any
    ):
        """Initialize AWS Braket backend.
        
        Args:
            device_arn: Braket device ARN (local simulator or AWS device)
            aws_region: AWS region for cloud devices
            **config: Backend configuration (shots, s3_bucket, etc.)
        """
        super().__init__(f"braket_{device_arn.split('/')[-1]}", **config)
        
        if not BRAKET_AVAILABLE:
            raise ImportError(
                "AWS Braket SDK is not available. Install it with: "
                "pip install amazon-braket-sdk"
            )
            
        self.device_arn = device_arn
        self.aws_region = aws_region
        self.device = None
        self.is_local = device_arn.startswith("local:")
        self._active_jobs: Dict[str, QuantumJob] = {}
        
        # Configuration
        self.shots = self.config.get("shots", 1024)
        self.s3_bucket = self.config.get("s3_bucket")
        self.s3_prefix = self.config.get("s3_prefix", "braket-results")
        
    def connect(self) -> None:
        """Establish connection to Braket device."""
        try:
            if self.is_local:
                # Use local Braket simulator
                self.device = LocalSimulator()
            else:
                # Use AWS Braket cloud device
                if not self.s3_bucket:
                    raise ValueError("S3 bucket required for AWS Braket cloud devices")
                    
                self.device = AwsDevice(
                    self.device_arn,
                    aws_session=boto3.Session(region_name=self.aws_region)
                )
                
            self._is_connected = True
            
        except Exception as e:
            raise BackendConnectionError(f"Failed to connect to Braket device: {e}")
            
    def disconnect(self) -> None:
        """Close connection to Braket device."""
        self.device = None
        self._is_connected = False
        
    def is_available(self) -> bool:
        """Check if Braket device is available."""
        if not self._is_connected:
            return False
            
        try:
            if not self.is_local:
                # Check device status for AWS devices
                status = self.device.status
                return status == "ONLINE"
            return True
        except Exception:
            return False
            
    def get_device_properties(self) -> Dict[str, Any]:
        """Get properties of the Braket device."""
        self.ensure_connected()
        
        properties = {
            "name": self.device_arn.split("/")[-1],
            "arn": self.device_arn,
            "provider": "AWS" if not self.is_local else "Local",
            "type": "simulator" if self.is_local or "simulator" in self.device_arn else "hardware",
        }
        
        if not self.is_local:
            try:
                device_properties = self.device.properties
                properties.update({
                    "paradigm": device_properties.paradigm,
                    "provider_name": device_properties.provider.name,
                    "qubit_count": getattr(device_properties.paradigm, 'qubitCount', None),
                    "connectivity": getattr(device_properties.paradigm, 'connectivity', None),
                    "native_gate_set": getattr(device_properties.paradigm, 'nativeGateSet', []),
                })
            except Exception:
                # Fallback for local or when properties unavailable
                properties["qubit_count"] = 30  # Default for local simulator
                
        else:
            properties["qubit_count"] = 30  # Local simulator default
            
        return properties
        
    def compile_circuit(self, circuit: Dict[str, Any]) -> Dict[str, Any]:
        """Compile circuit for Braket execution."""
        gates = circuit.get("gates", [])
        n_qubits = circuit.get("n_qubits", len(gates))
        parameters = circuit.get("parameters", [])
        
        # Create Braket circuit
        braket_circuit = Circuit()
        
        # Track parameter usage
        param_idx = 0
        parameter_values = []
        
        # Add gates to circuit
        for gate in gates:
            gate_type = gate.get("type", "").lower()
            wires = gate.get("wires", gate.get("qubit", 0))
            
            if isinstance(wires, int):
                wires = [wires]
                
            if gate_type == "rx":
                angle = gate.get("angle")
                if angle is None and param_idx < len(parameters):
                    angle = parameters[param_idx]
                    param_idx += 1
                elif angle is None:
                    angle = 0
                braket_circuit.rx(wires[0], angle)
                
            elif gate_type == "ry":
                angle = gate.get("angle")
                if angle is None and param_idx < len(parameters):
                    angle = parameters[param_idx]
                    param_idx += 1
                elif angle is None:
                    angle = 0
                braket_circuit.ry(wires[0], angle)
                
            elif gate_type == "rz":
                angle = gate.get("angle")
                if angle is None and param_idx < len(parameters):
                    angle = parameters[param_idx]
                    param_idx += 1
                elif angle is None:
                    angle = 0
                braket_circuit.rz(wires[0], angle)
                
            elif gate_type in ["h", "hadamard"]:
                braket_circuit.h(wires[0])
                
            elif gate_type in ["x", "pauli_x"]:
                braket_circuit.x(wires[0])
                
            elif gate_type in ["y", "pauli_y"]:
                braket_circuit.y(wires[0])
                
            elif gate_type in ["z", "pauli_z"]:
                braket_circuit.z(wires[0])
                
            elif gate_type in ["cnot", "cx"]:
                control = gate.get("control", wires[0])
                target = gate.get("target", wires[1] if len(wires) > 1 else wires[0] + 1)
                braket_circuit.cnot(control, target)
                
        return {
            "original": circuit,
            "braket_circuit": braket_circuit,
            "n_qubits": n_qubits,
            "parameters": parameters,
        }
        
    def submit_job(
        self, 
        circuits: List[Dict[str, Any]], 
        shots: int = 1024
    ) -> QuantumJob:
        """Submit circuits for execution on Braket device."""
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
                braket_circuit = compiled_circuit["braket_circuit"]
                
                # Execute circuit
                if self.is_local:
                    # Local execution
                    task_result = self.device.run(braket_circuit, shots=shots)
                    measurement_counts = task_result.measurement_counts
                    
                else:
                    # AWS cloud execution
                    task = self.device.run(
                        braket_circuit, 
                        shots=shots,
                        s3_destination_folder=(self.s3_bucket, self.s3_prefix)
                    )
                    
                    # Wait for completion (simplified - in production would poll)
                    task_result = task.result()
                    measurement_counts = task_result.measurement_counts
                
                # Process results
                counts = {}
                total_shots = sum(measurement_counts.values())
                
                for bitstring, count in measurement_counts.items():
                    counts[bitstring] = count
                    
                # Calculate expectation value (simplified)
                prob_0 = counts.get('0' * len(bitstring), 0) / total_shots if total_shots > 0 else 0
                expectation_value = 2 * prob_0 - 1  # Map to [-1, 1]
                
                execution_time = time.time() - start_time
                
                result = CircuitResult(
                    circuit_id=f"{job_id}_circuit_{i}",
                    counts=counts,
                    expectation_value=expectation_value,
                    execution_time=execution_time,
                    shots=shots,
                    metadata={
                        "backend": self.name, 
                        "device_arn": self.device_arn,
                        "is_local": self.is_local
                    }
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
            # For AWS Braket, could cancel the actual task here
            job.status = JobStatus.CANCELLED
            job.completed_at = time.time()
            return True
            
        return False