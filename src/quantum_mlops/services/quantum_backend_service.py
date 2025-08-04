"""Quantum backend management and routing service."""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np

from ..core import QuantumDevice

try:
    from ..backends import BackendManager, QuantumExecutor
    REAL_BACKENDS_AVAILABLE = True
except ImportError:
    REAL_BACKENDS_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class BackendInfo:
    """Information about a quantum backend."""
    name: str
    device: QuantumDevice
    qubits: int
    coherence_time: float  # in microseconds
    gate_fidelity: float
    queue_time: float  # estimated in minutes
    cost_per_shot: float
    is_available: bool
    last_calibration: Optional[str] = None


class BackendStatus(Enum):
    """Backend availability status."""
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    QUEUE_FULL = "queue_full"


class QuantumBackendService:
    """Service for managing quantum computing backends."""
    
    def __init__(self):
        self.backends = self._initialize_backends()
        self.circuit_cache: Dict[str, Any] = {}
        
        # Initialize real backend integration if available
        if REAL_BACKENDS_AVAILABLE:
            try:
                self.quantum_executor = QuantumExecutor()
                self.backend_manager = self.quantum_executor.backend_manager
                logger.info("Real quantum backends initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize real backends: {e}")
                self.quantum_executor = None
                self.backend_manager = None
        else:
            self.quantum_executor = None
            self.backend_manager = None
        
    def _initialize_backends(self) -> Dict[str, BackendInfo]:
        """Initialize available quantum backends."""
        return {
            "simulator_local": BackendInfo(
                name="Local Simulator",
                device=QuantumDevice.SIMULATOR,
                qubits=32,
                coherence_time=float('inf'),
                gate_fidelity=1.0,
                queue_time=0.0,
                cost_per_shot=0.0,
                is_available=True
            ),
            "aws_braket_aria": BackendInfo(
                name="AWS Braket Aria-1",
                device=QuantumDevice.AWS_BRAKET,
                qubits=25,
                coherence_time=15.0,
                gate_fidelity=0.995,
                queue_time=2.5,
                cost_per_shot=0.00035,
                is_available=True,
                last_calibration="2025-01-15T08:00:00Z"
            ),
            "ibm_quantum_toronto": BackendInfo(
                name="IBM Quantum Toronto",
                device=QuantumDevice.IBM_QUANTUM,
                qubits=27,
                coherence_time=100.0,
                gate_fidelity=0.992,
                queue_time=15.0,
                cost_per_shot=0.0,  # Free tier
                is_available=True,
                last_calibration="2025-01-14T12:00:00Z"
            ),
            "ionq_harmony": BackendInfo(
                name="IonQ Harmony",
                device=QuantumDevice.IONQ,
                qubits=11,
                coherence_time=1000.0,
                gate_fidelity=0.998,
                queue_time=3.0,
                cost_per_shot=0.00095,
                is_available=True,
                last_calibration="2025-01-15T10:30:00Z"
            )
        }
    
    def get_available_backends(self) -> List[BackendInfo]:
        """Get list of available quantum backends."""
        return [backend for backend in self.backends.values() if backend.is_available]
    
    def select_optimal_backend(
        self,
        n_qubits: int,
        optimization_target: str = "cost_performance",
        max_queue_time: Optional[float] = None,
        max_cost: Optional[float] = None
    ) -> BackendInfo:
        """Select optimal backend based on requirements.
        
        Args:
            n_qubits: Required number of qubits
            optimization_target: 'cost', 'performance', 'cost_performance', 'speed'
            max_queue_time: Maximum acceptable queue time in minutes
            max_cost: Maximum cost per shot
            
        Returns:
            Selected backend information
        """
        available_backends = [
            backend for backend in self.get_available_backends()
            if backend.qubits >= n_qubits
        ]
        
        if not available_backends:
            raise ValueError(f"No backends available with {n_qubits} qubits")
        
        # Apply constraints
        if max_queue_time is not None:
            available_backends = [
                b for b in available_backends if b.queue_time <= max_queue_time
            ]
        
        if max_cost is not None:
            available_backends = [
                b for b in available_backends if b.cost_per_shot <= max_cost
            ]
        
        if not available_backends:
            raise ValueError("No backends meet the specified constraints")
        
        # Select based on optimization target
        if optimization_target == "cost":
            return min(available_backends, key=lambda b: b.cost_per_shot)
        elif optimization_target == "speed":
            return min(available_backends, key=lambda b: b.queue_time)
        elif optimization_target == "performance":
            return max(available_backends, key=lambda b: b.gate_fidelity * b.coherence_time)
        else:  # cost_performance
            def score_backend(backend: BackendInfo) -> float:
                # Normalize metrics and compute composite score
                cost_score = 1.0 / (1.0 + backend.cost_per_shot * 1000)  # Lower cost = higher score
                perf_score = backend.gate_fidelity * min(1.0, backend.coherence_time / 100.0)
                speed_score = 1.0 / (1.0 + backend.queue_time / 10.0)  # Lower queue = higher score
                return (cost_score + perf_score + speed_score) / 3
            
            return max(available_backends, key=score_backend)
    
    def check_backend_status(self, backend_name: str) -> BackendStatus:
        """Check current status of a quantum backend."""
        if backend_name not in self.backends:
            raise ValueError(f"Unknown backend: {backend_name}")
        
        backend = self.backends[backend_name]
        
        # Simulate status checks (in production, would query actual APIs)
        if not backend.is_available:
            return BackendStatus.OFFLINE
        
        if backend.device == QuantumDevice.SIMULATOR:
            return BackendStatus.ONLINE
        
        # Simulate queue status based on time and backend
        import random
        random.seed(int(time.time()) + hash(backend_name))
        
        if random.random() < 0.1:  # 10% chance of maintenance
            return BackendStatus.MAINTENANCE
        elif backend.queue_time > 30:  # Queue too full
            return BackendStatus.QUEUE_FULL
        else:
            return BackendStatus.ONLINE
    
    def estimate_execution_time(
        self,
        backend_name: str,
        n_circuits: int,
        shots_per_circuit: int = 1024
    ) -> Dict[str, float]:
        """Estimate total execution time for quantum circuits.
        
        Returns:
            Dictionary with time estimates in minutes
        """
        if backend_name not in self.backends:
            raise ValueError(f"Unknown backend: {backend_name}")
        
        backend = self.backends[backend_name]
        
        # Base execution time per circuit (simplified model)
        if backend.device == QuantumDevice.SIMULATOR:
            exec_time_per_circuit = 0.1  # seconds
        else:
            exec_time_per_circuit = 2.0  # seconds for hardware
        
        total_exec_time = (n_circuits * exec_time_per_circuit * shots_per_circuit / 1024) / 60  # minutes
        
        return {
            "queue_time": backend.queue_time,
            "execution_time": total_exec_time,
            "total_time": backend.queue_time + total_exec_time
        }
    
    def estimate_cost(
        self,
        backend_name: str,
        n_circuits: int,
        shots_per_circuit: int = 1024
    ) -> Dict[str, float]:
        """Estimate cost for quantum circuit execution."""
        if backend_name not in self.backends:
            raise ValueError(f"Unknown backend: {backend_name}")
        
        backend = self.backends[backend_name]
        total_shots = n_circuits * shots_per_circuit
        total_cost = total_shots * backend.cost_per_shot
        
        return {
            "cost_per_shot": backend.cost_per_shot,
            "total_shots": total_shots,
            "total_cost": total_cost,
            "currency": "USD"
        }
    
    def submit_circuits(
        self,
        backend_name: str,
        circuits: List[Dict[str, Any]],
        shots: int = 1024
    ) -> str:
        """Submit quantum circuits for execution.
        
        Returns:
            Job ID for tracking execution
        """
        if backend_name not in self.backends:
            raise ValueError(f"Unknown backend: {backend_name}")
        
        backend = self.backends[backend_name]
        status = self.check_backend_status(backend_name)
        
        if status != BackendStatus.ONLINE:
            raise RuntimeError(f"Backend {backend_name} is not available: {status}")
        
        # Generate job ID
        job_id = f"qml_job_{int(time.time())}_{hash(str(circuits)) % 10000:04d}"
        
        logger.info(
            f"Submitted {len(circuits)} circuits to {backend_name} "
            f"with {shots} shots each (Job ID: {job_id})"
        )
        
        # In production, would submit to actual backend
        return job_id
    
    def get_job_results(self, job_id: str) -> Dict[str, Any]:
        """Retrieve results from quantum job execution.
        
        This is a simplified simulation - in production would query actual backends.
        """
        # Simulate job completion
        time.sleep(0.1)  # Simulate processing delay
        
        # Generate mock results
        results = {
            "job_id": job_id,
            "status": "completed",
            "results": [
                {
                    "circuit_id": i,
                    "counts": {"0": 512, "1": 512},  # Mock measurement results
                    "expectation_value": np.random.uniform(-1, 1)
                }
                for i in range(5)  # Assume 5 circuits
            ],
            "execution_time": 2.5,
            "timestamp": time.time()
        }
        
        return results
    
    def optimize_circuit_for_backend(
        self,
        circuit: Dict[str, Any],
        backend_name: str
    ) -> Dict[str, Any]:
        """Optimize quantum circuit for specific backend constraints."""
        if backend_name not in self.backends:
            raise ValueError(f"Unknown backend: {backend_name}")
        
        backend = self.backends[backend_name]
        
        # Cache key for optimization
        cache_key = f"{backend_name}_{hash(str(circuit))}"
        if cache_key in self.circuit_cache:
            return self.circuit_cache[cache_key]
        
        # Simulate circuit optimization
        optimized_circuit = circuit.copy()
        
        # Add backend-specific optimizations
        if backend.device == QuantumDevice.IBM_QUANTUM:
            # IBM-specific gate decomposition
            optimized_circuit["gates"] = self._decompose_to_native_gates(
                circuit.get("gates", []), ["cx", "rz", "sx"]
            )
        elif backend.device == QuantumDevice.IONQ:
            # IonQ native gates
            optimized_circuit["gates"] = self._decompose_to_native_gates(
                circuit.get("gates", []), ["gpi", "gpi2", "ms"]
            )
        
        # Add optimization metadata
        optimized_circuit["optimization_info"] = {
            "original_depth": circuit.get("depth", 0),
            "optimized_depth": max(1, circuit.get("depth", 0) - 2),
            "gate_reduction": 0.15,
            "backend": backend_name
        }
        
        # Cache the result
        self.circuit_cache[cache_key] = optimized_circuit
        
        return optimized_circuit
    
    def _decompose_to_native_gates(
        self,
        gates: List[Dict[str, Any]],
        native_gates: List[str]
    ) -> List[Dict[str, Any]]:
        """Decompose gates to backend-native gate set."""
        # Simplified gate decomposition
        decomposed = []
        
        for gate in gates:
            gate_type = gate.get("type", "")
            
            if gate_type in native_gates:
                decomposed.append(gate)
            else:
                # Simulate decomposition
                if gate_type in ["h", "hadamard"]:
                    # H = RZ(π) RY(π/2)
                    decomposed.extend([
                        {"type": "rz", "angle": np.pi, "qubit": gate.get("qubit", 0)},
                        {"type": "ry", "angle": np.pi/2, "qubit": gate.get("qubit", 0)}
                    ])
                else:
                    # Default: keep original gate
                    decomposed.append(gate)
        
        return decomposed
        
    def execute_circuits_real(
        self,
        circuits: List[Dict[str, Any]],
        preferred_backends: Optional[List[str]] = None,
        shots: int = 1024
    ) -> Dict[str, Any]:
        """Execute circuits using real quantum backends if available.
        
        Args:
            circuits: List of circuit descriptions
            preferred_backends: Preferred backend names
            shots: Number of measurement shots
            
        Returns:
            Execution results with real backend data
        """
        if not REAL_BACKENDS_AVAILABLE or not self.quantum_executor:
            logger.warning("Real backends not available, falling back to simulation")
            return self._simulate_execution(circuits, shots)
            
        try:
            # Execute using real quantum executor
            results = []
            
            for circuit in circuits:
                try:
                    result = self.quantum_executor.execute(
                        circuit,
                        backend_names=preferred_backends,
                        shots=shots
                    )
                    
                    results.append({
                        "circuit_id": getattr(result, 'circuit_id', 'unknown'),
                        "counts": getattr(result, 'counts', {}),  
                        "expectation_value": getattr(result, 'expectation_value', 0.0),
                        "execution_time": getattr(result, 'execution_time', 0.0),
                        "backend_used": "real_backend"
                    })
                    
                except Exception as e:
                    logger.warning(f"Real backend execution failed for circuit: {e}")
                    # Fallback to simulation for this circuit
                    sim_result = self._simulate_single_circuit(circuit, shots)
                    results.append(sim_result)
                    
            return {
                "status": "completed",
                "results": results,
                "backend_type": "real" if any(r.get("backend_used") == "real_backend" for r in results) else "simulation",
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Real backend execution completely failed: {e}")
            return self._simulate_execution(circuits, shots)
            
    def _simulate_execution(self, circuits: List[Dict[str, Any]], shots: int) -> Dict[str, Any]:
        """Fallback simulation execution."""
        results = []
        
        for i, circuit in enumerate(circuits):
            result = self._simulate_single_circuit(circuit, shots, circuit_id=f"sim_circuit_{i}")
            results.append(result)
            
        return {
            "status": "completed", 
            "results": results,
            "backend_type": "simulation",
            "timestamp": time.time()
        }
        
    def _simulate_single_circuit(self, circuit: Dict[str, Any], shots: int, circuit_id: str = None) -> Dict[str, Any]:
        """Simulate single circuit execution."""
        # Simple simulation - in practice would use more sophisticated methods
        n_qubits = circuit.get("n_qubits", 2) 
        
        # Generate mock counts
        states = [format(i, f'0{n_qubits}b') for i in range(2**min(n_qubits, 3))]  # Limit to 3 qubits for simplicity
        counts = {}
        
        for state in states:
            # Random distribution with bias toward certain states
            prob = np.random.exponential(scale=0.5)
            counts[state] = int(prob * shots / len(states))
            
        # Normalize to exact shot count
        total = sum(counts.values())
        if total > 0:
            factor = shots / total
            counts = {state: max(1, int(count * factor)) for state, count in counts.items()}
            
        # Adjust to exact shot count
        current_total = sum(counts.values())
        if current_total != shots:
            adjustment = shots - current_total
            first_state = list(counts.keys())[0]
            counts[first_state] = max(0, counts[first_state] + adjustment)
        
        # Calculate mock expectation value
        prob_zero = counts.get('0' * n_qubits, 0) / shots
        expectation_value = 2 * prob_zero - 1  # Map to [-1, 1]
        
        return {
            "circuit_id": circuit_id or "simulated",
            "counts": counts,
            "expectation_value": expectation_value,
            "execution_time": 0.1,  # Mock execution time
            "backend_used": "simulation"
        }
        
    def get_real_backend_status(self) -> Dict[str, Any]:
        """Get status of real quantum backends."""
        if not REAL_BACKENDS_AVAILABLE or not self.quantum_executor:
            return {
                "available": False,
                "reason": "Real backends not initialized",
                "backends": {}
            }
            
        try:
            status = self.quantum_executor.get_backend_status()
            available_backends = self.quantum_executor.list_available_backends()
            
            return {
                "available": True,
                "backends": status,
                "available_backend_names": available_backends,
                "quantum_executor_ready": True
            }
            
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
                "backends": {}
            }
            
    def benchmark_real_backends(self, shots: int = 100) -> Dict[str, Any]:
        """Benchmark real quantum backends if available."""
        if not REAL_BACKENDS_AVAILABLE or not self.quantum_executor:
            return {"error": "Real backends not available"}
            
        try:
            return self.quantum_executor.benchmark_backends(shots=shots)
        except Exception as e:
            logger.error(f"Backend benchmarking failed: {e}")
            return {"error": str(e)}