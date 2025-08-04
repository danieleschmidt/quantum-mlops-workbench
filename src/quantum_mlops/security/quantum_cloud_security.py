"""Secure quantum cloud access and backend security wrapper."""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps

from .credential_manager import CredentialManager, get_credential_manager
from .authorization import AuthorizationManager, Permission, get_authorization_manager
from .audit_logger import AuditLogger

logger = logging.getLogger(__name__)


@dataclass
class SecurityPolicy:
    """Security policy for quantum backend access."""
    
    backend_name: str
    max_qubits: int = 100
    max_jobs_per_hour: int = 50
    max_cost_per_job: float = 10.0
    allowed_operations: List[str] = None
    require_approval: bool = False
    audit_level: str = "standard"  # minimal, standard, detailed
    
    def __post_init__(self):
        if self.allowed_operations is None:
            self.allowed_operations = ["measure", "gate", "circuit"]


@dataclass
class JobMetrics:
    """Security metrics for quantum job execution."""
    
    job_id: str
    user_id: str
    backend_name: str
    qubits_used: int
    gates_count: int
    estimated_cost: float
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    security_flags: List[str] = None
    
    def __post_init__(self):
        if self.security_flags is None:
            self.security_flags = []


class QuantumBackendSecurityWrapper:
    """Security wrapper for quantum backends."""
    
    def __init__(self, backend, security_policy: SecurityPolicy,
                 credential_manager: CredentialManager = None,
                 auth_manager: AuthorizationManager = None,
                 audit_logger: AuditLogger = None):
        """Initialize security wrapper."""
        self.backend = backend
        self.policy = security_policy
        self.credential_manager = credential_manager or get_credential_manager()
        self.auth_manager = auth_manager or get_authorization_manager()
        self.audit_logger = audit_logger
        
        # Rate limiting
        self._job_history: List[Dict[str, Any]] = []
        self._cost_tracking: Dict[str, float] = {}
        
    def _check_rate_limits(self, user_id: str) -> bool:
        """Check rate limits for user."""
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        
        # Count jobs in last hour
        recent_jobs = [
            job for job in self._job_history 
            if job['user_id'] == user_id and 
               datetime.fromisoformat(job['timestamp']) > hour_ago
        ]
        
        if len(recent_jobs) >= self.policy.max_jobs_per_hour:
            logger.warning(f"Rate limit exceeded for user {user_id}")
            return False
            
        return True
        
    def _validate_circuit_security(self, circuit: Dict[str, Any], user_id: str) -> List[str]:
        """Validate circuit for security issues."""
        security_flags = []
        
        # Check qubit count
        n_qubits = circuit.get('n_qubits', 0)
        if n_qubits > self.policy.max_qubits:
            security_flags.append(f"excessive_qubits:{n_qubits}")
            
        # Check gate count
        gates = circuit.get('gates', [])
        if len(gates) > 10000:  # Arbitrary large circuit threshold
            security_flags.append(f"large_circuit:{len(gates)}")
            
        # Check for suspicious patterns
        gate_types = [gate.get('type', '').lower() for gate in gates]
        
        # Check for excessive measurements
        measurement_count = gate_types.count('measure') + gate_types.count('measurement')
        if measurement_count > n_qubits * 2:
            security_flags.append(f"excessive_measurements:{measurement_count}")
            
        # Check for unusual gate sequences
        if len(set(gate_types)) == 1 and len(gate_types) > 100:
            security_flags.append("repetitive_gates")
            
        # Check operation permissions
        for gate_type in set(gate_types):
            if gate_type not in self.policy.allowed_operations:
                security_flags.append(f"unauthorized_operation:{gate_type}")
                
        return security_flags
        
    def _estimate_cost(self, circuit: Dict[str, Any], shots: int = 1024) -> float:
        """Estimate job cost."""
        # Simplified cost estimation
        n_qubits = circuit.get('n_qubits', 1)
        n_gates = len(circuit.get('gates', []))
        
        # Cost factors (these would be backend-specific in practice)
        base_cost = 0.01
        qubit_cost = n_qubits * 0.005
        gate_cost = n_gates * 0.001
        shot_cost = shots * 0.0001
        
        return base_cost + qubit_cost + gate_cost + shot_cost
        
    def _setup_secure_credentials(self, user_id: str) -> bool:
        """Setup secure credentials for backend access."""
        try:
            # Get credentials based on backend type
            if "aws" in self.policy.backend_name.lower() or "braket" in self.policy.backend_name.lower():
                creds = self.credential_manager.get_aws_credentials("default_aws")
                if creds:
                    os.environ["AWS_ACCESS_KEY_ID"] = creds["access_key_id"]
                    os.environ["AWS_SECRET_ACCESS_KEY"] = creds["secret_access_key"]
                    os.environ["AWS_DEFAULT_REGION"] = creds["region"]
                    return True
                    
            elif "ibm" in self.policy.backend_name.lower():
                creds = self.credential_manager.get_ibm_credentials("default_ibm")
                if creds:
                    os.environ["IBM_QUANTUM_TOKEN"] = creds["token"]
                    os.environ["IBM_QUANTUM_INSTANCE"] = creds["instance"]
                    return True
                    
            elif "ionq" in self.policy.backend_name.lower():
                creds = self.credential_manager.get_ionq_credentials("default_ionq")
                if creds:
                    os.environ["IONQ_API_KEY"] = creds["api_key"]
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Failed to setup credentials: {e}")
            return False
            
    def secure_execute(self, circuits: List[Dict[str, Any]], shots: int = 1024,
                      user_id: str = None, **kwargs) -> Dict[str, Any]:
        """Execute circuits with security controls."""
        start_time = time.time()
        
        if not user_id:
            raise ValueError("User ID required for secure execution")
            
        # Check authorization
        if not self.auth_manager.has_permission(user_id, Permission.QUANTUM_EXECUTE):
            raise ValueError("User lacks quantum execution permission")
            
        if not self.auth_manager.check_quantum_backend_access(user_id, self.policy.backend_name):
            raise ValueError(f"User not authorized for backend: {self.policy.backend_name}")
            
        # Check rate limits
        if not self._check_rate_limits(user_id):
            raise ValueError("Rate limit exceeded")
            
        # Validate and analyze circuits
        total_cost = 0.0
        all_security_flags = []
        
        for i, circuit in enumerate(circuits):
            security_flags = self._validate_circuit_security(circuit, user_id)
            all_security_flags.extend(security_flags)
            
            # Check for blocking security issues
            blocking_flags = [flag for flag in security_flags 
                            if flag.startswith(('excessive_qubits', 'unauthorized_operation'))]
            if blocking_flags:
                raise ValueError(f"Security violation in circuit {i}: {blocking_flags}")
                
            total_cost += self._estimate_cost(circuit, shots)
            
        # Check cost limits
        if total_cost > self.policy.max_cost_per_job:
            raise ValueError(f"Job cost {total_cost:.2f} exceeds limit {self.policy.max_cost_per_job:.2f}")
            
        # Check resource limits
        resource_limits = self.auth_manager.get_resource_limits(user_id)
        current_cost = self._cost_tracking.get(user_id, 0.0)
        
        if (resource_limits.get("max_cost_per_day", -1) != -1 and 
            current_cost + total_cost > resource_limits["max_cost_per_day"]):
            raise ValueError("Daily cost limit would be exceeded")
            
        # Setup secure credentials
        if not self._setup_secure_credentials(user_id):
            logger.warning(f"No credentials available for backend {self.policy.backend_name}")
            
        # Execute job with monitoring
        job_id = f"secure_{int(time.time())}_{user_id}"
        
        try:
            # Audit job start
            if self.audit_logger:
                self.audit_logger.log_quantum_job_start(
                    user_id=user_id,
                    backend=self.policy.backend_name,
                    job_id=job_id,
                    circuits_count=len(circuits),
                    estimated_cost=total_cost
                )
                
            # Execute on backend
            if hasattr(self.backend, 'submit_job'):
                # New backend interface
                quantum_job = self.backend.submit_job(circuits, shots)
                result = self.backend.get_job_result(quantum_job.job_id)
            else:
                # Legacy interface - wrap in compatibility layer
                result = self._execute_legacy(circuits, shots)
                
            execution_time = time.time() - start_time
            
            # Track metrics
            metrics = JobMetrics(
                job_id=job_id,
                user_id=user_id,
                backend_name=self.policy.backend_name,
                qubits_used=max(circuit.get('n_qubits', 0) for circuit in circuits),
                gates_count=sum(len(circuit.get('gates', [])) for circuit in circuits),
                estimated_cost=total_cost,
                execution_time=execution_time,
                success=True,
                security_flags=all_security_flags
            )
            
            # Update tracking
            self._job_history.append({
                'user_id': user_id,
                'timestamp': datetime.utcnow().isoformat(),
                'cost': total_cost,
                'success': True
            })
            self._cost_tracking[user_id] = self._cost_tracking.get(user_id, 0.0) + total_cost
            
            # Audit successful execution
            if self.audit_logger:
                self.audit_logger.log_quantum_job_complete(
                    user_id=user_id,
                    job_id=job_id,
                    success=True,
                    execution_time=execution_time,
                    actual_cost=total_cost
                )
                
            return {
                'result': result,
                'metrics': metrics,
                'security_flags': all_security_flags,
                'cost': total_cost
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Track failed job
            metrics = JobMetrics(
                job_id=job_id,
                user_id=user_id,
                backend_name=self.policy.backend_name,
                qubits_used=max(circuit.get('n_qubits', 0) for circuit in circuits),
                gates_count=sum(len(circuit.get('gates', [])) for circuit in circuits),
                estimated_cost=total_cost,
                execution_time=execution_time,
                success=False,
                error_message=str(e),
                security_flags=all_security_flags
            )
            
            # Audit failed execution
            if self.audit_logger:
                self.audit_logger.log_quantum_job_complete(
                    user_id=user_id,
                    job_id=job_id,
                    success=False,
                    execution_time=execution_time,
                    error=str(e)
                )
                
            logger.error(f"Secure execution failed for job {job_id}: {e}")
            raise
            
    def _execute_legacy(self, circuits: List[Dict[str, Any]], shots: int) -> Any:
        """Execute using legacy backend interface."""
        # This would integrate with the existing backend interface
        # For now, return a mock result
        return {
            'counts': [{'00': shots//2, '01': shots//4, '10': shots//8, '11': shots//8}] * len(circuits),
            'success': True
        }
        
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        return {
            'backend_name': self.policy.backend_name,
            'policy': {
                'max_qubits': self.policy.max_qubits,
                'max_jobs_per_hour': self.policy.max_jobs_per_hour,
                'max_cost_per_job': self.policy.max_cost_per_job,
                'audit_level': self.policy.audit_level
            },
            'recent_jobs': len(self._job_history),
            'total_users': len(set(job['user_id'] for job in self._job_history)),
            'total_cost_tracked': sum(self._cost_tracking.values())
        }


class SecureQuantumBackendManager:
    """Secure manager for quantum backends."""
    
    def __init__(self, credential_manager: CredentialManager = None,
                 auth_manager: AuthorizationManager = None,
                 audit_logger: AuditLogger = None):
        """Initialize secure backend manager."""
        self.credential_manager = credential_manager or get_credential_manager()
        self.auth_manager = auth_manager or get_authorization_manager()
        self.audit_logger = audit_logger
        self.secure_backends: Dict[str, QuantumBackendSecurityWrapper] = {}
        self.security_policies: Dict[str, SecurityPolicy] = {}
        
        # Default security policies
        self._create_default_policies()
        
    def _create_default_policies(self) -> None:
        """Create default security policies."""
        policies = [
            SecurityPolicy(
                backend_name="simulator",
                max_qubits=30,
                max_jobs_per_hour=100,
                max_cost_per_job=0.0,
                allowed_operations=["measure", "gate", "circuit", "h", "x", "y", "z", "cnot", "rx", "ry", "rz"],
                audit_level="minimal"
            ),
            SecurityPolicy(
                backend_name="aws_braket",
                max_qubits=50,
                max_jobs_per_hour=20,
                max_cost_per_job=50.0,
                allowed_operations=["measure", "gate", "circuit", "h", "x", "y", "z", "cnot", "rx", "ry", "rz"],
                require_approval=True,
                audit_level="detailed"
            ),
            SecurityPolicy(
                backend_name="ibm_quantum",
                max_qubits=127,
                max_jobs_per_hour=10,
                max_cost_per_job=100.0,
                allowed_operations=["measure", "gate", "circuit", "h", "x", "y", "z", "cnot", "rx", "ry", "rz"],
                require_approval=True,
                audit_level="detailed"
            ),
            SecurityPolicy(
                backend_name="ionq",
                max_qubits=32,
                max_jobs_per_hour=5,
                max_cost_per_job=200.0,
                allowed_operations=["measure", "gate", "circuit", "h", "x", "y", "z", "cnot", "rx", "ry", "rz"],
                require_approval=True,
                audit_level="detailed"
            )
        ]
        
        for policy in policies:
            self.security_policies[policy.backend_name] = policy
            
    def wrap_backend(self, backend, backend_name: str) -> QuantumBackendSecurityWrapper:
        """Wrap backend with security controls."""
        policy = self.security_policies.get(backend_name)
        if not policy:
            # Create default policy
            policy = SecurityPolicy(backend_name=backend_name)
            self.security_policies[backend_name] = policy
            
        wrapper = QuantumBackendSecurityWrapper(
            backend=backend,
            security_policy=policy,
            credential_manager=self.credential_manager,
            auth_manager=self.auth_manager,
            audit_logger=self.audit_logger
        )
        
        self.secure_backends[backend_name] = wrapper
        return wrapper
        
    def get_secure_backend(self, backend_name: str) -> Optional[QuantumBackendSecurityWrapper]:
        """Get secure backend wrapper."""
        return self.secure_backends.get(backend_name)
        
    def update_security_policy(self, backend_name: str, policy: SecurityPolicy) -> None:
        """Update security policy for backend."""
        self.security_policies[backend_name] = policy
        
        # Update wrapper if it exists
        if backend_name in self.secure_backends:
            self.secure_backends[backend_name].policy = policy
            
    def get_security_report(self) -> Dict[str, Any]:
        """Generate security report for all backends."""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'backends': {},
            'summary': {
                'total_backends': len(self.secure_backends),
                'high_security_backends': 0,
                'total_jobs_tracked': 0
            }
        }
        
        for name, wrapper in self.secure_backends.items():
            status = wrapper.get_security_status()
            report['backends'][name] = status
            
            if wrapper.policy.audit_level == "detailed":
                report['summary']['high_security_backends'] += 1
            report['summary']['total_jobs_tracked'] += status['recent_jobs']
            
        return report


def secure_quantum_execution(backend_name: str, require_approval: bool = False):
    """Decorator for secure quantum execution."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            user_id = kwargs.get('current_user')
            if not user_id:
                raise ValueError("User authentication required for quantum execution")
                
            auth_manager = get_authorization_manager()
            
            # Check basic quantum permission
            if not auth_manager.has_permission(user_id, Permission.QUANTUM_EXECUTE):
                raise ValueError("Quantum execution permission required")
                
            # Check backend-specific access
            if not auth_manager.check_quantum_backend_access(user_id, backend_name):
                raise ValueError(f"Access denied to backend: {backend_name}")
                
            # TODO: Implement approval workflow if required
            if require_approval:
                logger.info(f"Approval required for {user_id} to access {backend_name}")
                
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Global secure backend manager
_global_secure_manager: Optional[SecureQuantumBackendManager] = None

def get_secure_backend_manager() -> SecureQuantumBackendManager:
    """Get global secure backend manager."""
    global _global_secure_manager
    if _global_secure_manager is None:
        _global_secure_manager = SecureQuantumBackendManager()
    return _global_secure_manager