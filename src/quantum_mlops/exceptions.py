"""Centralized exception handling for quantum MLOps workbench."""

import logging
import traceback
from typing import Optional, Dict, Any, List
from enum import Enum, auto
from dataclasses import dataclass
from datetime import datetime


logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    
    LOW = auto()        # Warnings, minor issues
    MEDIUM = auto()     # Recoverable errors
    HIGH = auto()       # Critical errors
    CRITICAL = auto()   # System failures


class ErrorCategory(Enum):
    """Error categories for classification."""
    
    # System errors
    CONFIGURATION = auto()
    VALIDATION = auto()
    AUTHENTICATION = auto()
    AUTHORIZATION = auto()
    
    # Quantum-specific errors
    QUANTUM_CIRCUIT = auto()
    QUANTUM_BACKEND = auto()
    QUANTUM_EXECUTION = auto()
    QUANTUM_HARDWARE = auto()
    
    # Data errors
    DATA_PROCESSING = auto()
    MODEL_TRAINING = auto()
    MODEL_INFERENCE = auto()
    
    # Infrastructure errors
    NETWORK = auto()
    DATABASE = auto()
    STORAGE = auto()
    
    # External service errors
    CLOUD_PROVIDER = auto()
    API_EXTERNAL = auto()
    
    # Unknown errors
    UNKNOWN = auto()


@dataclass
class ErrorContext:
    """Context information for errors."""
    
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class QuantumMLOpsException(Exception):
    """Base exception for quantum MLOps workbench."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        suggestions: Optional[List[str]] = None,
        error_code: Optional[str] = None
    ):
        """Initialize quantum MLOps exception."""
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.cause = cause
        self.suggestions = suggestions or []
        self.error_code = error_code
        self.traceback_str = traceback.format_exc()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "message": self.message,
            "category": self.category.name,
            "severity": self.severity.name,
            "error_code": self.error_code,
            "context": {
                "user_id": self.context.user_id,
                "session_id": self.context.session_id,
                "request_id": self.context.request_id,
                "component": self.context.component,
                "operation": self.context.operation,
                "parameters": self.context.parameters,
                "timestamp": self.context.timestamp.isoformat() if self.context.timestamp else None
            },
            "suggestions": self.suggestions,
            "cause": str(self.cause) if self.cause else None,
            "traceback": self.traceback_str
        }
        
    def get_user_friendly_message(self) -> str:
        """Get user-friendly error message."""
        base_message = self.message
        
        if self.suggestions:
            base_message += "\n\nSuggestions:"
            for suggestion in self.suggestions:
                base_message += f"\n- {suggestion}"
                
        return base_message


# Configuration and validation errors
class ConfigurationError(QuantumMLOpsException):
    """Configuration-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class ValidationError(QuantumMLOpsException):
    """Input validation errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


# Authentication and authorization errors
class AuthenticationError(QuantumMLOpsException):
    """Authentication-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            suggestions=["Check your credentials", "Verify your authentication token"],
            **kwargs
        )


class AuthorizationError(QuantumMLOpsException):
    """Authorization-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            suggestions=["Check your permissions", "Contact your administrator"],
            **kwargs
        )


# Quantum-specific errors
class QuantumCircuitError(QuantumMLOpsException):
    """Quantum circuit-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.QUANTUM_CIRCUIT,
            severity=ErrorSeverity.MEDIUM,
            suggestions=[
                "Verify circuit gates and parameters",
                "Check qubit connectivity",
                "Validate circuit depth"
            ],
            **kwargs
        )


class QuantumBackendError(QuantumMLOpsException):
    """Quantum backend-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.QUANTUM_BACKEND,
            severity=ErrorSeverity.HIGH,
            suggestions=[
                "Check backend availability",
                "Try a different quantum backend",
                "Verify backend credentials"
            ],
            **kwargs
        )


class QuantumExecutionError(QuantumMLOpsException):
    """Quantum job execution errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.QUANTUM_EXECUTION,
            severity=ErrorSeverity.MEDIUM,
            suggestions=[
                "Check job parameters",
                "Verify shot count and timeout",
                "Review circuit complexity"
            ],
            **kwargs
        )


class QuantumHardwareError(QuantumMLOpsException):
    """Quantum hardware-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.QUANTUM_HARDWARE,
            severity=ErrorSeverity.HIGH,
            suggestions=[
                "Try a different quantum device",
                "Check hardware maintenance schedule",
                "Use simulator as fallback"
            ],
            **kwargs
        )


# Data and model errors
class DataProcessingError(QuantumMLOpsException):
    """Data processing errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DATA_PROCESSING,
            severity=ErrorSeverity.MEDIUM,
            suggestions=[
                "Verify data format and structure",
                "Check data preprocessing steps",
                "Validate input dimensions"
            ],
            **kwargs
        )


class ModelTrainingError(QuantumMLOpsException):
    """Model training errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.MODEL_TRAINING,
            severity=ErrorSeverity.MEDIUM,
            suggestions=[
                "Adjust hyperparameters",
                "Check training data quality",
                "Review optimization settings"
            ],
            **kwargs
        )


class ModelInferenceError(QuantumMLOpsException):
    """Model inference errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.MODEL_INFERENCE,
            severity=ErrorSeverity.MEDIUM,
            suggestions=[
                "Verify model is trained",
                "Check input data format",
                "Validate model parameters"
            ],
            **kwargs
        )


# Infrastructure errors
class NetworkError(QuantumMLOpsException):
    """Network-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.HIGH,
            suggestions=[
                "Check network connectivity",
                "Verify firewall settings",
                "Try again after a moment"
            ],
            **kwargs
        )


class DatabaseError(QuantumMLOpsException):
    """Database-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            suggestions=[
                "Check database connectivity",
                "Verify database credentials",
                "Review database configuration"
            ],
            **kwargs
        )


class StorageError(QuantumMLOpsException):
    """Storage-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.STORAGE,
            severity=ErrorSeverity.HIGH,
            suggestions=[
                "Check storage permissions",
                "Verify available disk space",
                "Review storage configuration"
            ],
            **kwargs
        )


# External service errors
class CloudProviderError(QuantumMLOpsException):
    """Cloud provider service errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CLOUD_PROVIDER,
            severity=ErrorSeverity.HIGH,
            suggestions=[
                "Check cloud service status",
                "Verify API credentials",
                "Review service quotas and limits"
            ],
            **kwargs
        )


class ExternalAPIError(QuantumMLOpsException):
    """External API errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.API_EXTERNAL,
            severity=ErrorSeverity.MEDIUM,
            suggestions=[
                "Check external service status",
                "Verify API credentials",
                "Review API rate limits"
            ],
            **kwargs
        )


class ErrorHandler:
    """Centralized error handler for quantum MLOps."""
    
    def __init__(self, enable_logging: bool = True):
        """Initialize error handler."""
        self.enable_logging = enable_logging
        self.error_counts = {}
        self.recent_errors = []
        self.max_recent_errors = 100
        
    def handle_exception(
        self,
        exception: Exception,
        context: Optional[ErrorContext] = None,
        reraise: bool = True
    ) -> Optional[QuantumMLOpsException]:
        """Handle and potentially convert exception."""
        
        # Convert to QuantumMLOpsException if not already
        if isinstance(exception, QuantumMLOpsException):
            qml_exception = exception
        else:
            qml_exception = self._convert_exception(exception, context)
            
        # Log the error
        if self.enable_logging:
            self._log_error(qml_exception)
            
        # Track error statistics
        self._track_error(qml_exception)
        
        # Reraise if requested
        if reraise:
            raise qml_exception
            
        return qml_exception
        
    def _convert_exception(
        self,
        exception: Exception,
        context: Optional[ErrorContext] = None
    ) -> QuantumMLOpsException:
        """Convert standard exception to QuantumMLOpsException."""
        
        # Map common exceptions to quantum MLOps exceptions
        if isinstance(exception, (ConnectionError, TimeoutError)):
            return NetworkError(
                f"Network error: {str(exception)}",
                context=context,
                cause=exception
            )
        elif isinstance(exception, ValueError):
            return ValidationError(
                f"Validation error: {str(exception)}",
                context=context,
                cause=exception
            )
        elif isinstance(exception, PermissionError):
            return AuthorizationError(
                f"Permission denied: {str(exception)}",
                context=context,
                cause=exception
            )
        elif isinstance(exception, FileNotFoundError):
            return ConfigurationError(
                f"File not found: {str(exception)}",
                context=context,
                cause=exception
            )
        else:
            # Generic conversion
            return QuantumMLOpsException(
                f"Unexpected error: {str(exception)}",
                category=ErrorCategory.UNKNOWN,
                severity=ErrorSeverity.MEDIUM,
                context=context,
                cause=exception
            )
            
    def _log_error(self, exception: QuantumMLOpsException) -> None:
        """Log error with appropriate level."""
        
        log_data = {
            "error_code": exception.error_code,
            "category": exception.category.name,
            "severity": exception.severity.name,
            "user_id": getattr(exception.context, 'user_id', None) if hasattr(exception.context, 'user_id') else None,
            "component": getattr(exception.context, 'component', None) if hasattr(exception.context, 'component') else None,
            "operation": getattr(exception.context, 'operation', None) if hasattr(exception.context, 'operation') else None
        }
        
        if exception.severity == ErrorSeverity.CRITICAL:
            logger.critical(exception.message, extra=log_data, exc_info=True)
        elif exception.severity == ErrorSeverity.HIGH:
            logger.error(exception.message, extra=log_data, exc_info=True)
        elif exception.severity == ErrorSeverity.MEDIUM:
            logger.warning(exception.message, extra=log_data)
        else:
            logger.info(exception.message, extra=log_data)
            
    def _track_error(self, exception: QuantumMLOpsException) -> None:
        """Track error statistics."""
        
        # Count errors by category
        category_key = exception.category.name
        self.error_counts[category_key] = self.error_counts.get(category_key, 0) + 1
        
        # Store recent errors
        self.recent_errors.append({
            "timestamp": getattr(exception.context, 'timestamp', datetime.utcnow()) if hasattr(exception.context, 'timestamp') else datetime.utcnow(),
            "category": exception.category.name,
            "severity": exception.severity.name,
            "message": exception.message[:100],  # Truncate for storage
            "user_id": getattr(exception.context, 'user_id', None) if hasattr(exception.context, 'user_id') else None
        })
        
        # Maintain size limit
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors = self.recent_errors[-self.max_recent_errors:]
            
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "error_counts_by_category": self.error_counts.copy(),
            "recent_errors_count": len(self.recent_errors),
            "total_errors": sum(self.error_counts.values()),
            "most_common_errors": sorted(
                self.error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
        
    def get_recent_errors(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent errors."""
        return self.recent_errors[-limit:]
        
    def clear_statistics(self) -> None:
        """Clear error statistics."""
        self.error_counts.clear()
        self.recent_errors.clear()


# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get global error handler."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def handle_quantum_error(func):
    """Decorator to handle quantum-specific errors."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_handler = get_error_handler()
            context = ErrorContext(
                component="quantum_operation",
                operation=func.__name__,
                parameters={"args_count": len(args), "kwargs_keys": list(kwargs.keys())}
            )
            error_handler.handle_exception(e, context=context)
    return wrapper


def handle_api_error(func):
    """Decorator to handle API errors."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_handler = get_error_handler()
            context = ErrorContext(
                component="api",
                operation=func.__name__,
                parameters={"args_count": len(args), "kwargs_keys": list(kwargs.keys())}
            )
            error_handler.handle_exception(e, context=context)
    return wrapper


def safe_execute(
    func,
    *args,
    default_return=None,
    context: Optional[ErrorContext] = None,
    **kwargs
):
    """Safely execute function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_handler = get_error_handler()
        error_handler.handle_exception(e, context=context, reraise=False)
        return default_return