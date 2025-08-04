"""Role-based access control and authorization for quantum MLOps."""

from enum import Enum, auto
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class Permission(Enum):
    """System permissions."""
    
    # General permissions
    READ = auto()
    WRITE = auto()
    DELETE = auto()
    ADMIN = auto()
    
    # Quantum-specific permissions
    QUANTUM_READ = auto()
    QUANTUM_EXECUTE = auto()
    QUANTUM_ADMIN = auto()
    
    # Backend permissions
    BACKEND_CONNECT = auto()
    BACKEND_ADMIN = auto()
    
    # Model permissions
    MODEL_READ = auto()
    MODEL_WRITE = auto()
    MODEL_DELETE = auto()
    MODEL_DEPLOY = auto()
    
    # Experiment permissions
    EXPERIMENT_READ = auto()
    EXPERIMENT_WRITE = auto()
    EXPERIMENT_DELETE = auto()
    
    # Cost management permissions
    COST_READ = auto()
    COST_MANAGE = auto()
    
    # Security permissions
    SECURITY_READ = auto()
    SECURITY_ADMIN = auto()
    
    # Monitoring permissions
    MONITORING_READ = auto()
    MONITORING_ADMIN = auto()


@dataclass
class Role:
    """Role definition with permissions."""
    
    name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    inherits_from: List[str] = field(default_factory=list)
    quantum_backends: List[str] = field(default_factory=list)  # Allowed backends
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    
    def add_permission(self, permission: Permission) -> None:
        """Add permission to role."""
        self.permissions.add(permission)
        
    def remove_permission(self, permission: Permission) -> None:
        """Remove permission from role."""
        self.permissions.discard(permission)
        
    def has_permission(self, permission: Permission) -> bool:
        """Check if role has specific permission."""
        return permission in self.permissions


class RoleManager:
    """Manages system roles and permissions."""
    
    def __init__(self):
        """Initialize role manager with default roles."""
        self.roles: Dict[str, Role] = {}
        self._create_default_roles()
        
    def _create_default_roles(self) -> None:
        """Create default system roles."""
        
        # Guest role - minimal read access
        guest = Role(
            name="guest",
            description="Read-only access to public resources",
            permissions={Permission.READ}
        )
        
        # User role - basic quantum operations
        user = Role(
            name="user",
            description="Standard user with quantum simulation access",
            permissions={
                Permission.READ,
                Permission.WRITE,
                Permission.QUANTUM_READ,
                Permission.QUANTUM_EXECUTE,
                Permission.BACKEND_CONNECT,
                Permission.MODEL_READ,
                Permission.MODEL_WRITE,
                Permission.EXPERIMENT_READ,
                Permission.EXPERIMENT_WRITE,
                Permission.COST_READ,
                Permission.MONITORING_READ
            },
            quantum_backends=["simulator", "pennylane_default.qubit"],
            resource_limits={
                "max_qubits": 20,
                "max_jobs_per_day": 100,
                "max_cost_per_day": 50.0
            }
        )
        
        # Researcher role - access to real quantum hardware
        researcher = Role(
            name="researcher",
            description="Researcher with hardware quantum access",
            permissions={
                Permission.READ,
                Permission.WRITE,
                Permission.QUANTUM_READ,
                Permission.QUANTUM_EXECUTE,
                Permission.BACKEND_CONNECT,
                Permission.MODEL_READ,
                Permission.MODEL_WRITE,
                Permission.MODEL_DEPLOY,
                Permission.EXPERIMENT_READ,
                Permission.EXPERIMENT_WRITE,
                Permission.COST_READ,
                Permission.MONITORING_READ
            },
            inherits_from=["user"],
            quantum_backends=["simulator", "aws_braket", "ibm_quantum", "ionq"],
            resource_limits={
                "max_qubits": 50,
                "max_jobs_per_day": 500,
                "max_cost_per_day": 200.0
            }
        )
        
        # Developer role - system development and testing
        developer = Role(
            name="developer",
            description="Developer with extended system access",
            permissions={
                Permission.READ,
                Permission.WRITE,
                Permission.DELETE,
                Permission.QUANTUM_READ,
                Permission.QUANTUM_EXECUTE,
                Permission.BACKEND_CONNECT,
                Permission.BACKEND_ADMIN,
                Permission.MODEL_READ,
                Permission.MODEL_WRITE,
                Permission.MODEL_DELETE,
                Permission.MODEL_DEPLOY,
                Permission.EXPERIMENT_READ,
                Permission.EXPERIMENT_WRITE,
                Permission.EXPERIMENT_DELETE,
                Permission.COST_READ,
                Permission.COST_MANAGE,
                Permission.MONITORING_READ,
                Permission.MONITORING_ADMIN
            },
            inherits_from=["researcher"],
            quantum_backends=["*"],  # All backends
            resource_limits={
                "max_qubits": 100,
                "max_jobs_per_day": 1000,
                "max_cost_per_day": 500.0
            }
        )
        
        # Admin role - full system access
        admin = Role(
            name="admin",
            description="Administrator with full system access",
            permissions=set(Permission),  # All permissions
            inherits_from=["developer"],
            quantum_backends=["*"],
            resource_limits={
                "max_qubits": -1,  # Unlimited
                "max_jobs_per_day": -1,
                "max_cost_per_day": -1
            }
        )
        
        # Security Officer role - security-focused permissions
        security_officer = Role(
            name="security_officer",
            description="Security officer with security and audit access",
            permissions={
                Permission.READ,
                Permission.SECURITY_READ,
                Permission.SECURITY_ADMIN,
                Permission.MONITORING_READ,
                Permission.MONITORING_ADMIN,
                Permission.COST_READ,
                Permission.BACKEND_ADMIN
            },
            quantum_backends=["simulator"],
            resource_limits={
                "max_qubits": 10,
                "max_jobs_per_day": 50,
                "max_cost_per_day": 0.0
            }
        )
        
        # Store default roles
        for role in [guest, user, researcher, developer, admin, security_officer]:
            self.roles[role.name] = role
            
    def create_role(self, role: Role) -> None:
        """Create a new role."""
        self.roles[role.name] = role
        logger.info(f"Created role: {role.name}")
        
    def get_role(self, name: str) -> Optional[Role]:
        """Get role by name."""
        return self.roles.get(name)
        
    def delete_role(self, name: str) -> bool:
        """Delete a role."""
        if name in ["admin", "user", "guest"]:  # Protect system roles
            raise ValueError(f"Cannot delete system role: {name}")
            
        if name in self.roles:
            del self.roles[name]
            logger.info(f"Deleted role: {name}")
            return True
        return False
        
    def list_roles(self) -> List[str]:
        """List all role names."""
        return list(self.roles.keys())
        
    def get_effective_permissions(self, role_names: List[str]) -> Set[Permission]:
        """Get effective permissions for list of roles."""
        permissions = set()
        
        for role_name in role_names:
            role = self.get_role(role_name)
            if role:
                permissions.update(role.permissions)
                
                # Add inherited permissions
                for parent_role_name in role.inherits_from:
                    parent_permissions = self.get_effective_permissions([parent_role_name])
                    permissions.update(parent_permissions)
                    
        return permissions
        
    def get_allowed_backends(self, role_names: List[str]) -> List[str]:
        """Get allowed quantum backends for roles."""
        backends = set()
        
        for role_name in role_names:
            role = self.get_role(role_name)
            if role:
                if "*" in role.quantum_backends:
                    return ["*"]  # All backends allowed
                backends.update(role.quantum_backends)
                
                # Add inherited backends
                for parent_role_name in role.inherits_from:
                    parent_backends = self.get_allowed_backends([parent_role_name])
                    if "*" in parent_backends:
                        return ["*"]
                    backends.update(parent_backends)
                    
        return list(backends)
        
    def get_resource_limits(self, role_names: List[str]) -> Dict[str, Any]:
        """Get effective resource limits for roles."""
        limits = {
            "max_qubits": 0,
            "max_jobs_per_day": 0,
            "max_cost_per_day": 0.0
        }
        
        for role_name in role_names:
            role = self.get_role(role_name)
            if role:
                for key, value in role.resource_limits.items():
                    if value == -1:  # Unlimited
                        limits[key] = -1
                    elif limits[key] != -1:  # Not already unlimited
                        limits[key] = max(limits[key], value)
                        
                # Check inherited limits
                for parent_role_name in role.inherits_from:
                    parent_limits = self.get_resource_limits([parent_role_name])
                    for key, value in parent_limits.items():
                        if value == -1:
                            limits[key] = -1
                        elif limits[key] != -1:
                            limits[key] = max(limits[key], value)
                            
        return limits


@dataclass
class AccessContext:
    """Context for access control decisions."""
    
    user_id: str
    roles: List[str]
    resource_type: str
    resource_id: Optional[str] = None
    action: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AuthorizationManager:
    """Main authorization manager."""
    
    def __init__(self, role_manager: RoleManager = None):
        """Initialize authorization manager."""
        self.role_manager = role_manager or RoleManager()
        self._user_roles: Dict[str, List[str]] = {}
        
    def assign_role(self, user_id: str, role_name: str) -> None:
        """Assign role to user."""
        if role_name not in self.role_manager.roles:
            raise ValueError(f"Role {role_name} does not exist")
            
        if user_id not in self._user_roles:
            self._user_roles[user_id] = []
            
        if role_name not in self._user_roles[user_id]:
            self._user_roles[user_id].append(role_name)
            logger.info(f"Assigned role {role_name} to user {user_id}")
            
    def revoke_role(self, user_id: str, role_name: str) -> None:
        """Revoke role from user."""
        if user_id in self._user_roles and role_name in self._user_roles[user_id]:
            self._user_roles[user_id].remove(role_name)
            logger.info(f"Revoked role {role_name} from user {user_id}")
            
    def get_user_roles(self, user_id: str) -> List[str]:
        """Get roles assigned to user."""
        return self._user_roles.get(user_id, ["guest"])
        
    def has_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has specific permission."""
        user_roles = self.get_user_roles(user_id)
        effective_permissions = self.role_manager.get_effective_permissions(user_roles)
        return permission in effective_permissions
        
    def check_access(self, context: AccessContext) -> bool:
        """Check access based on context."""
        user_roles = self.get_user_roles(context.user_id)
        effective_permissions = self.role_manager.get_effective_permissions(user_roles)
        
        # Map resource types and actions to permissions
        permission_map = {
            ("quantum", "read"): Permission.QUANTUM_READ,
            ("quantum", "execute"): Permission.QUANTUM_EXECUTE,
            ("quantum", "admin"): Permission.QUANTUM_ADMIN,
            ("backend", "connect"): Permission.BACKEND_CONNECT,
            ("backend", "admin"): Permission.BACKEND_ADMIN,
            ("model", "read"): Permission.MODEL_READ,
            ("model", "write"): Permission.MODEL_WRITE,
            ("model", "delete"): Permission.MODEL_DELETE,
            ("model", "deploy"): Permission.MODEL_DEPLOY,
            ("experiment", "read"): Permission.EXPERIMENT_READ,
            ("experiment", "write"): Permission.EXPERIMENT_WRITE,
            ("experiment", "delete"): Permission.EXPERIMENT_DELETE,
            ("cost", "read"): Permission.COST_READ,
            ("cost", "manage"): Permission.COST_MANAGE,
            ("security", "read"): Permission.SECURITY_READ,
            ("security", "admin"): Permission.SECURITY_ADMIN,
            ("monitoring", "read"): Permission.MONITORING_READ,
            ("monitoring", "admin"): Permission.MONITORING_ADMIN,
        }
        
        required_permission = permission_map.get((context.resource_type, context.action))
        if not required_permission:
            # If no specific permission mapped, check for general permissions
            if context.action == "read":
                required_permission = Permission.READ
            elif context.action in ["write", "create", "update"]:
                required_permission = Permission.WRITE
            elif context.action == "delete":
                required_permission = Permission.DELETE
            else:
                return False
                
        return required_permission in effective_permissions
        
    def check_quantum_backend_access(self, user_id: str, backend_name: str) -> bool:
        """Check if user can access specific quantum backend."""
        user_roles = self.get_user_roles(user_id)
        allowed_backends = self.role_manager.get_allowed_backends(user_roles)
        
        return "*" in allowed_backends or backend_name in allowed_backends
        
    def get_resource_limits(self, user_id: str) -> Dict[str, Any]:
        """Get effective resource limits for user."""
        user_roles = self.get_user_roles(user_id)
        return self.role_manager.get_resource_limits(user_roles)
        
    def require_permission(self, permission: Permission):
        """Decorator to require specific permission."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                user_id = kwargs.get('current_user')
                if not user_id:
                    raise ValueError("User authentication required")
                    
                if not self.has_permission(user_id, permission):
                    raise ValueError(f"Permission denied: {permission.name}")
                    
                return func(*args, **kwargs)
            return wrapper
        return decorator
        
    def require_quantum_access(self, backend_name: str = None):
        """Decorator to require quantum backend access."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                user_id = kwargs.get('current_user')
                if not user_id:
                    raise ValueError("User authentication required")
                    
                if not self.has_permission(user_id, Permission.QUANTUM_EXECUTE):
                    raise ValueError("Quantum execution permission required")
                    
                if backend_name and not self.check_quantum_backend_access(user_id, backend_name):
                    raise ValueError(f"Access denied to backend: {backend_name}")
                    
                return func(*args, **kwargs)
            return wrapper
        return decorator
        
    def enforce_resource_limits(self, user_id: str, resource_type: str, 
                              current_usage: Dict[str, Any]) -> bool:
        """Check if resource usage is within limits."""
        limits = self.get_resource_limits(user_id)
        
        for limit_key, limit_value in limits.items():
            if limit_value == -1:  # Unlimited
                continue
                
            current_value = current_usage.get(limit_key, 0)
            if current_value >= limit_value:
                logger.warning(f"Resource limit exceeded for user {user_id}: {limit_key}")
                return False
                
        return True
        
    def get_user_summary(self, user_id: str) -> Dict[str, Any]:
        """Get authorization summary for user."""
        user_roles = self.get_user_roles(user_id)
        effective_permissions = self.role_manager.get_effective_permissions(user_roles)
        allowed_backends = self.role_manager.get_allowed_backends(user_roles)
        resource_limits = self.get_resource_limits(user_id)
        
        return {
            "user_id": user_id,
            "roles": user_roles,
            "permissions": [p.name for p in effective_permissions],
            "allowed_backends": allowed_backends,
            "resource_limits": resource_limits
        }


# Global authorization manager
_global_authz_manager: Optional[AuthorizationManager] = None

def get_authorization_manager() -> AuthorizationManager:
    """Get global authorization manager."""
    global _global_authz_manager
    if _global_authz_manager is None:
        _global_authz_manager = AuthorizationManager()
    return _global_authz_manager