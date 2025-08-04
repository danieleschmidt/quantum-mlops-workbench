"""Base classes for quantum backend implementations."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np


class JobStatus(Enum):
    """Status of a quantum job."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CircuitResult:
    """Result from executing a quantum circuit."""
    circuit_id: str
    counts: Dict[str, int]
    expectation_value: Optional[float] = None
    execution_time: Optional[float] = None
    shots: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass 
class QuantumJob:
    """Represents a quantum computing job."""
    job_id: str
    backend_name: str
    status: JobStatus
    circuits: List[Dict[str, Any]]
    shots: int
    created_at: float
    completed_at: Optional[float] = None
    results: Optional[List[CircuitResult]] = None
    error_message: Optional[str] = None


class QuantumBackendError(Exception):
    """Base exception for quantum backend errors."""
    pass


class BackendConnectionError(QuantumBackendError):
    """Error connecting to quantum backend."""
    pass


class CircuitExecutionError(QuantumBackendError):
    """Error executing quantum circuit."""
    pass


class QuantumBackend(ABC):
    """Abstract base class for quantum computing backends."""
    
    def __init__(self, name: str, **config: Any):
        """Initialize quantum backend.
        
        Args:
            name: Backend identifier
            **config: Backend-specific configuration
        """
        self.name = name
        self.config = config
        self._is_connected = False
        
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to quantum backend."""
        pass
        
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to quantum backend."""
        pass
        
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available for job submission."""
        pass
        
    @abstractmethod
    def get_device_properties(self) -> Dict[str, Any]:
        """Get properties of the quantum device."""
        pass
        
    @abstractmethod
    def compile_circuit(self, circuit: Dict[str, Any]) -> Dict[str, Any]:
        """Compile circuit for this backend."""
        pass
        
    @abstractmethod
    def submit_job(
        self, 
        circuits: List[Dict[str, Any]], 
        shots: int = 1024
    ) -> QuantumJob:
        """Submit circuits for execution."""
        pass
        
    @abstractmethod
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get status of submitted job."""
        pass
        
    @abstractmethod
    def get_job_results(self, job_id: str) -> List[CircuitResult]:
        """Retrieve results from completed job."""
        pass
        
    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        pass
        
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        
    def ensure_connected(self) -> None:
        """Ensure backend connection is established."""
        if not self._is_connected:
            self.connect()