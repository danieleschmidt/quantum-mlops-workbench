"""Centralized logging configuration for quantum MLOps workbench."""

import os
import sys
import logging
import logging.handlers
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import threading
from queue import Queue, Empty
from dataclasses import dataclass, asdict


@dataclass
class LogConfig:
    """Logging configuration."""
    
    level: str = "INFO"
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = True
    log_to_console: bool = True
    log_directory: str = "~/.quantum_mlops/logs"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_structured_logging: bool = True
    enable_performance_logging: bool = True
    enable_audit_logging: bool = True
    log_quantum_events: bool = True
    log_security_events: bool = True


class QuantumJSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, include_trace: bool = True):
        """Initialize JSON formatter."""
        super().__init__()
        self.include_trace = include_trace
        
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        
        # Base log data
        log_data = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add process/thread info
        log_data["process_id"] = record.process
        log_data["thread_id"] = record.thread
        log_data["thread_name"] = record.threadName
        
        # Add extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message']:
                extra_fields[key] = value
                
        if extra_fields:
            log_data["extra"] = extra_fields
            
        # Add exception info if present
        if record.exc_info and self.include_trace:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None
            }
            
        return json.dumps(log_data, separators=(',', ':'))


class QuantumLogFilter(logging.Filter):
    """Custom filter for quantum-specific logs."""
    
    def __init__(self, component: str = None, min_level: int = logging.INFO):
        """Initialize quantum log filter."""
        super().__init__()
        self.component = component
        self.min_level = min_level
        
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records."""
        
        # Filter by level
        if record.levelno < self.min_level:
            return False
            
        # Filter by component if specified
        if self.component and not record.name.startswith(f"quantum_mlops.{self.component}"):
            return False
            
        # Add quantum-specific metadata
        if not hasattr(record, 'quantum_component'):
            record.quantum_component = self._extract_component(record.name)
            
        return True
        
    def _extract_component(self, logger_name: str) -> str:
        """Extract component name from logger name."""
        parts = logger_name.split('.')
        if len(parts) >= 2 and parts[0] == 'quantum_mlops':
            return parts[1]
        return 'unknown'


class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler to prevent blocking."""
    
    def __init__(self, target_handler: logging.Handler, queue_size: int = 1000):
        """Initialize async log handler."""
        super().__init__()
        self.target_handler = target_handler
        self.queue = Queue(maxsize=queue_size)
        self.worker_thread = threading.Thread(target=self._log_worker, daemon=True)
        self._shutdown = False
        self.worker_thread.start()
        
    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record asynchronously."""
        try:
            self.queue.put_nowait(record)
        except:
            # Queue is full, drop the record
            pass
            
    def _log_worker(self) -> None:
        """Background worker for processing log records."""
        while not self._shutdown:
            try:
                record = self.queue.get(timeout=1.0)
                if record is None:  # Shutdown signal
                    break
                self.target_handler.emit(record)
                self.queue.task_done()
            except Empty:
                continue
            except Exception as e:
                # Log to stderr to avoid recursion
                print(f"Error in log worker: {e}", file=sys.stderr)
                
    def close(self) -> None:
        """Close the handler."""
        self._shutdown = True
        self.queue.put(None)  # Shutdown signal
        self.worker_thread.join(timeout=5.0)
        self.target_handler.close()
        super().close()


class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize performance logger."""
        self.logger = logger
        self._start_times = {}
        
    def start_operation(self, operation_id: str, operation_name: str, **kwargs) -> None:
        """Start timing an operation."""
        start_time = datetime.utcnow()
        self._start_times[operation_id] = start_time
        
        self.logger.info(
            f"Operation started: {operation_name}",
            extra={
                "operation_id": operation_id,
                "operation_name": operation_name,
                "operation_status": "started",
                "start_time": start_time.isoformat(),
                **kwargs
            }
        )
        
    def end_operation(self, operation_id: str, operation_name: str, 
                     success: bool = True, **kwargs) -> float:
        """End timing an operation."""
        end_time = datetime.utcnow()
        start_time = self._start_times.pop(operation_id, end_time)
        duration = (end_time - start_time).total_seconds()
        
        self.logger.info(
            f"Operation {'completed' if success else 'failed'}: {operation_name}",
            extra={
                "operation_id": operation_id,
                "operation_name": operation_name,
                "operation_status": "completed" if success else "failed",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                **kwargs
            }
        )
        
        return duration
        
    def log_metric(self, metric_name: str, value: float, **kwargs) -> None:
        """Log a performance metric."""
        self.logger.info(
            f"Performance metric: {metric_name}",
            extra={
                "metric_name": metric_name,
                "metric_value": value,
                "metric_timestamp": datetime.utcnow().isoformat(),
                **kwargs
            }
        )


class QuantumEventLogger:
    """Logger for quantum-specific events."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize quantum event logger."""
        self.logger = logger
        
    def log_quantum_job(self, job_id: str, backend: str, action: str, 
                       success: bool = True, **kwargs) -> None:
        """Log quantum job event."""
        self.logger.info(
            f"Quantum job {action}: {job_id}",
            extra={
                "quantum_job_id": job_id,
                "quantum_backend": backend,
                "quantum_action": action,
                "quantum_success": success,
                "event_type": "quantum_job",
                **kwargs
            }
        )
        
    def log_circuit_execution(self, circuit_id: str, backend: str, 
                            shots: int, execution_time: float, **kwargs) -> None:
        """Log circuit execution."""
        self.logger.info(
            f"Circuit executed: {circuit_id}",
            extra={
                "circuit_id": circuit_id,
                "quantum_backend": backend,
                "shots": shots,
                "execution_time": execution_time,
                "event_type": "circuit_execution",
                **kwargs
            }
        )
        
    def log_backend_status(self, backend: str, status: str, **kwargs) -> None:
        """Log backend status change."""
        self.logger.info(
            f"Backend status: {backend} - {status}",
            extra={
                "quantum_backend": backend,
                "backend_status": status,
                "event_type": "backend_status",
                **kwargs
            }
        )


class LoggingManager:
    """Central logging manager for quantum MLOps."""
    
    def __init__(self, config: LogConfig = None):
        """Initialize logging manager."""
        self.config = config or LogConfig()
        self.loggers = {}
        self.handlers = []
        self.performance_logger = None
        self.quantum_logger = None
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        
        # Create log directory
        log_dir = Path(self.config.log_directory).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set global logging level
        logging.getLogger().setLevel(getattr(logging, self.config.level.upper()))
        
        # Setup root quantum_mlops logger
        root_logger = logging.getLogger("quantum_mlops")
        root_logger.setLevel(getattr(logging, self.config.level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if self.config.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            if self.config.enable_structured_logging:
                console_handler.setFormatter(QuantumJSONFormatter())
            else:
                console_handler.setFormatter(logging.Formatter(self.config.format_string))
            console_handler.addFilter(QuantumLogFilter())
            root_logger.addHandler(console_handler)
            self.handlers.append(console_handler)
            
        # File handlers
        if self.config.log_to_file:
            
            # Main application log
            app_log_file = log_dir / "quantum_mlops.log"
            app_handler = logging.handlers.RotatingFileHandler(
                app_log_file,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
            if self.config.enable_structured_logging:
                app_handler.setFormatter(QuantumJSONFormatter())
            else:
                app_handler.setFormatter(logging.Formatter(self.config.format_string))
            app_handler.addFilter(QuantumLogFilter())
            
            # Use async handler for file logging
            async_app_handler = AsyncLogHandler(app_handler)
            root_logger.addHandler(async_app_handler)
            self.handlers.append(async_app_handler)
            
            # Error log (errors and above only)
            error_log_file = log_dir / "quantum_mlops_errors.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(QuantumJSONFormatter())
            
            async_error_handler = AsyncLogHandler(error_handler)
            root_logger.addHandler(async_error_handler)
            self.handlers.append(async_error_handler)
            
            # Performance log
            if self.config.enable_performance_logging:
                perf_log_file = log_dir / "quantum_mlops_performance.log"
                perf_handler = logging.handlers.RotatingFileHandler(
                    perf_log_file,
                    maxBytes=self.config.max_file_size,
                    backupCount=self.config.backup_count
                )
                perf_handler.setFormatter(QuantumJSONFormatter())
                perf_handler.addFilter(QuantumLogFilter(component="performance"))
                
                perf_logger = logging.getLogger("quantum_mlops.performance")
                perf_logger.addHandler(perf_handler)
                perf_logger.setLevel(logging.INFO)
                perf_logger.propagate = False
                
                self.performance_logger = PerformanceLogger(perf_logger)
                self.handlers.append(perf_handler)
                
            # Quantum events log
            if self.config.log_quantum_events:
                quantum_log_file = log_dir / "quantum_mlops_quantum.log"
                quantum_handler = logging.handlers.RotatingFileHandler(
                    quantum_log_file,
                    maxBytes=self.config.max_file_size,
                    backupCount=self.config.backup_count
                )
                quantum_handler.setFormatter(QuantumJSONFormatter())
                quantum_handler.addFilter(QuantumLogFilter(component="quantum"))
                
                quantum_logger = logging.getLogger("quantum_mlops.quantum")
                quantum_logger.addHandler(quantum_handler)
                quantum_logger.setLevel(logging.INFO)
                quantum_logger.propagate = False
                
                self.quantum_logger = QuantumEventLogger(quantum_logger)
                self.handlers.append(quantum_handler)
                
        # Set file permissions
        try:
            for log_file in log_dir.glob("*.log*"):
                os.chmod(log_file, 0o600)
        except OSError:
            pass
            
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create logger."""
        if name not in self.loggers:
            logger = logging.getLogger(f"quantum_mlops.{name}")
            self.loggers[name] = logger
        return self.loggers[name]
        
    def get_performance_logger(self) -> Optional[PerformanceLogger]:
        """Get performance logger."""
        return self.performance_logger
        
    def get_quantum_logger(self) -> Optional[QuantumEventLogger]:
        """Get quantum event logger."""
        return self.quantum_logger
        
    def update_log_level(self, level: str) -> None:
        """Update logging level."""
        self.config.level = level
        log_level = getattr(logging, level.upper())
        
        # Update all loggers
        logging.getLogger("quantum_mlops").setLevel(log_level)
        for logger in self.loggers.values():
            logger.setLevel(log_level)
            
    def add_log_filter(self, component: str, min_level: int = logging.INFO) -> None:
        """Add component-specific log filter."""
        log_filter = QuantumLogFilter(component=component, min_level=min_level)
        
        # Add filter to all handlers
        for handler in self.handlers:
            handler.addFilter(log_filter)
            
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        stats = {
            "config": asdict(self.config),
            "active_loggers": list(self.loggers.keys()),
            "handler_count": len(self.handlers),
            "performance_logging_enabled": self.performance_logger is not None,
            "quantum_logging_enabled": self.quantum_logger is not None
        }
        
        # Add handler-specific stats
        handler_stats = []
        for handler in self.handlers:
            handler_info = {
                "type": type(handler).__name__,
                "level": handler.level,
                "filter_count": len(handler.filters)
            }
            
            if hasattr(handler, 'baseFilename'):
                handler_info["log_file"] = handler.baseFilename
                try:
                    file_size = os.path.getsize(handler.baseFilename)
                    handler_info["file_size_bytes"] = file_size
                except OSError:
                    pass
                    
            handler_stats.append(handler_info)
            
        stats["handlers"] = handler_stats
        return stats
        
    def shutdown(self) -> None:
        """Shutdown logging system."""
        for handler in self.handlers:
            handler.close()
        self.handlers.clear()
        self.loggers.clear()


# Global logging manager
_global_logging_manager: Optional[LoggingManager] = None


def get_logging_manager() -> LoggingManager:
    """Get global logging manager."""
    global _global_logging_manager
    if _global_logging_manager is None:
        _global_logging_manager = LoggingManager()
    return _global_logging_manager


def get_logger(name: str) -> logging.Logger:
    """Get logger for component."""
    return get_logging_manager().get_logger(name)


def log_performance(operation_name: str):
    """Decorator for performance logging."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            perf_logger = get_logging_manager().get_performance_logger()
            if perf_logger:
                operation_id = f"{operation_name}_{threading.get_ident()}_{datetime.utcnow().timestamp()}"
                perf_logger.start_operation(operation_id, operation_name)
                try:
                    result = func(*args, **kwargs)
                    perf_logger.end_operation(operation_id, operation_name, success=True)
                    return result
                except Exception as e:
                    perf_logger.end_operation(operation_id, operation_name, success=False, error=str(e))
                    raise
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def log_quantum_event(action: str):
    """Decorator for quantum event logging."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            quantum_logger = get_logging_manager().get_quantum_logger()
            if quantum_logger:
                try:
                    result = func(*args, **kwargs)
                    quantum_logger.log_quantum_job(
                        job_id=f"{func.__name__}_{threading.get_ident()}",
                        backend=kwargs.get('backend', 'unknown'),
                        action=action,
                        success=True
                    )
                    return result
                except Exception as e:
                    quantum_logger.log_quantum_job(
                        job_id=f"{func.__name__}_{threading.get_ident()}",
                        backend=kwargs.get('backend', 'unknown'),
                        action=action,
                        success=False,
                        error=str(e)
                    )
                    raise
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator