"""Quantum MLOps Security Framework.

This module provides comprehensive security features for the quantum MLOps workbench,
including credential management, authentication, authorization, encryption, and
quantum-specific security measures.
"""

from .credential_manager import CredentialManager, SecureCredentialStore
from .authentication import AuthenticationManager, JWTAuthenticator
from .authorization import AuthorizationManager, Role, Permission
from .encryption import EncryptionManager, DataEncryption
from .audit_logger import AuditLogger, SecurityEvent
from .quantum_security import QuantumSecurityValidator, CircuitSanitizer
from .input_validator import InputValidator, QuantumInputValidator
from .config_manager import SecurityConfigManager

__all__ = [
    "CredentialManager",
    "SecureCredentialStore", 
    "AuthenticationManager",
    "JWTAuthenticator",
    "AuthorizationManager", 
    "Role",
    "Permission",
    "EncryptionManager",
    "DataEncryption",
    "AuditLogger",
    "SecurityEvent",
    "QuantumSecurityValidator",
    "CircuitSanitizer", 
    "InputValidator",
    "QuantumInputValidator",
    "SecurityConfigManager",
]

__version__ = "0.1.0"