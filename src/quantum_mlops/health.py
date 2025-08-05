"""Health monitoring and reliability features for quantum MLOps workbench."""

import os
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import asyncio
import logging
from pathlib import Path
import json

from .exceptions import QuantumMLOpsException, ErrorCategory, ErrorSeverity, get_error_handler
from .logging_config import get_logger

logger = get_logger("health")


class HealthStatus(Enum):
    """Health check status."""
    
    HEALTHY = auto()
    WARNING = auto()
    CRITICAL = auto()
    UNKNOWN = auto()


class ComponentType(Enum):
    """Types of system components."""
    
    CORE = auto()
    DATABASE = auto()
    QUANTUM_BACKEND = auto()
    STORAGE = auto()
    NETWORK = auto()
    EXTERNAL_SERVICE = auto()
    CACHE = auto()


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    
    component_name: str
    component_type: ComponentType
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component_name": self.component_name,
            "component_type": self.component_type.name,
            "status": self.status.name,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms
        }


@dataclass
class SystemMetrics:
    """System performance metrics."""
    
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    disk_free_gb: float
    load_average: List[float]
    uptime_seconds: float
    process_count: int
    network_connections: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "disk_usage_percent": self.disk_usage_percent,
            "disk_free_gb": self.disk_free_gb,
            "load_average": self.load_average,
            "uptime_seconds": self.uptime_seconds,
            "process_count": self.process_count,
            "network_connections": self.network_connections,
            "timestamp": self.timestamp.isoformat()
        }


class BaseHealthCheck:
    """Base class for health checks."""
    
    def __init__(self, name: str, component_type: ComponentType, 
                 timeout_seconds: float = 30.0):
        """Initialize health check."""
        self.name = name
        self.component_type = component_type
        self.timeout_seconds = timeout_seconds
        
    def check(self) -> HealthCheckResult:
        """Perform health check."""
        start_time = time.time()
        
        try:
            status, message, details = self._perform_check()
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component_name=self.name,
                component_type=self.component_type,
                status=status,
                message=message,
                details=details,
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Health check failed for {self.name}: {e}")
            
            return HealthCheckResult(
                component_name=self.name,
                component_type=self.component_type,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)},
                duration_ms=duration_ms
            )
            
    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Override this method to implement specific health check."""
        raise NotImplementedError


class SystemResourcesCheck(BaseHealthCheck):
    """Check system resources (CPU, memory, disk)."""
    
    def __init__(self, cpu_threshold: float = 90.0, memory_threshold: float = 90.0,
                 disk_threshold: float = 90.0):
        """Initialize system resources check."""
        super().__init__("system_resources", ComponentType.CORE)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
        
    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check system resources."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk usage (root partition)
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / (1024**3)
        
        details = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_usage_percent": disk_percent,
            "disk_free_gb": disk_free_gb,
            "disk_total_gb": disk.total / (1024**3)
        }
        
        # Determine status
        if (cpu_percent > self.cpu_threshold or 
            memory_percent > self.memory_threshold or 
            disk_percent > self.disk_threshold):
            status = HealthStatus.CRITICAL
            message = "Resource usage critical"
        elif (cpu_percent > self.cpu_threshold * 0.8 or 
              memory_percent > self.memory_threshold * 0.8 or 
              disk_percent > self.disk_threshold * 0.8):
            status = HealthStatus.WARNING
            message = "Resource usage elevated"
        else:
            status = HealthStatus.HEALTHY
            message = "Resource usage normal"
            
        return status, message, details


class DatabaseConnectionCheck(BaseHealthCheck):
    """Check database connectivity."""
    
    def __init__(self, connection_string: str = None):
        """Initialize database connection check."""
        super().__init__("database", ComponentType.DATABASE)
        self.connection_string = connection_string
        
    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check database connection."""
        try:
            # Import database connection from the project
            from .database.connection import DatabaseConnection
            
            db = DatabaseConnection()
            
            # Test connection
            start_time = time.time()
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                
            connection_time = (time.time() - start_time) * 1000
            
            details = {
                "connection_time_ms": connection_time,
                "result": result[0] if result else None
            }
            
            if connection_time > 1000:  # More than 1 second
                status = HealthStatus.WARNING
                message = "Database connection slow"
            else:
                status = HealthStatus.HEALTHY
                message = "Database connection healthy"
                
            return status, message, details
            
        except ImportError:
            return HealthStatus.WARNING, "Database module not available", {}
        except Exception as e:
            return HealthStatus.CRITICAL, f"Database connection failed: {str(e)}", {"error": str(e)}


class QuantumBackendCheck(BaseHealthCheck):
    """Check quantum backend availability."""
    
    def __init__(self, backend_name: str):
        """Initialize quantum backend check."""
        super().__init__(f"quantum_backend_{backend_name}", ComponentType.QUANTUM_BACKEND)
        self.backend_name = backend_name
        
    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check quantum backend."""
        try:
            # Import backend manager
            from .backends.backend_manager import BackendManager
            
            manager = BackendManager()
            
            # Check backend availability
            backend_info = manager.get_backend_info(self.backend_name)
            if not backend_info:
                return HealthStatus.CRITICAL, f"Backend {self.backend_name} not found", {}
                
            # Test basic connectivity
            is_available = manager.is_backend_available(self.backend_name)
            
            details = {
                "backend_name": self.backend_name,
                "available": is_available,
                "backend_info": backend_info
            }
            
            if is_available:
                status = HealthStatus.HEALTHY
                message = f"Backend {self.backend_name} available"
            else:
                status = HealthStatus.WARNING
                message = f"Backend {self.backend_name} unavailable"
                
            return status, message, details
            
        except ImportError:
            return HealthStatus.WARNING, "Quantum backend module not available", {}
        except Exception as e:
            return HealthStatus.CRITICAL, f"Backend check failed: {str(e)}", {"error": str(e)}


class StorageCheck(BaseHealthCheck):
    """Check storage accessibility."""
    
    def __init__(self, storage_path: str = None):
        """Initialize storage check."""
        super().__init__("storage", ComponentType.STORAGE)
        self.storage_path = Path(storage_path or "~/.quantum_mlops").expanduser()
        
    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check storage."""
        try:
            # Ensure directory exists
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Test write/read
            test_file = self.storage_path / "health_check.tmp"
            test_data = f"health_check_{datetime.utcnow().isoformat()}"
            
            start_time = time.time()
            
            # Write test
            with open(test_file, 'w') as f:
                f.write(test_data)
                
            # Read test
            with open(test_file, 'r') as f:
                read_data = f.read()
                
            # Cleanup
            test_file.unlink()
            
            io_time = (time.time() - start_time) * 1000
            
            details = {
                "storage_path": str(self.storage_path),
                "io_time_ms": io_time,
                "writable": True,
                "readable": True
            }
            
            if read_data != test_data:
                status = HealthStatus.CRITICAL
                message = "Storage data integrity failed"
            elif io_time > 1000:  # More than 1 second
                status = HealthStatus.WARNING
                message = "Storage I/O slow"
            else:
                status = HealthStatus.HEALTHY
                message = "Storage accessible"
                
            return status, message, details
            
        except Exception as e:
            return HealthStatus.CRITICAL, f"Storage check failed: {str(e)}", {"error": str(e)}


class ExternalServiceCheck(BaseHealthCheck):
    """Check external service availability."""
    
    def __init__(self, service_name: str, url: str, timeout: float = 10.0):
        """Initialize external service check."""
        super().__init__(f"external_{service_name}", ComponentType.EXTERNAL_SERVICE)
        self.service_name = service_name
        self.url = url
        self.request_timeout = timeout
        
    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check external service."""
        try:
            import requests
            
            start_time = time.time()
            response = requests.get(self.url, timeout=self.request_timeout)
            response_time = (time.time() - start_time) * 1000
            
            details = {
                "service_name": self.service_name,
                "url": self.url,
                "status_code": response.status_code,
                "response_time_ms": response_time
            }
            
            if response.status_code == 200:
                if response_time > 5000:  # More than 5 seconds
                    status = HealthStatus.WARNING
                    message = f"Service {self.service_name} slow"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Service {self.service_name} available"
            else:
                status = HealthStatus.WARNING
                message = f"Service {self.service_name} returned {response.status_code}"
                
            return status, message, details
            
        except ImportError:
            return HealthStatus.WARNING, "Requests library not available", {}
        except Exception as e:
            return HealthStatus.CRITICAL, f"Service {self.service_name} unreachable: {str(e)}", {"error": str(e)}


class HealthMonitor:
    """Health monitoring system."""
    
    def __init__(self):
        """Initialize health monitor."""
        self.health_checks: List[BaseHealthCheck] = []
        self.metrics_history: List[SystemMetrics] = []
        self.health_history: List[Dict[str, HealthCheckResult]] = []
        self.max_history_size = 1000
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        self.monitoring_interval = 60  # seconds
        self.alert_callbacks: List[Callable] = []
        
        # Add default health checks
        self._setup_default_checks()
        
    def _setup_default_checks(self):
        """Setup default health checks."""
        # System resources
        self.add_health_check(SystemResourcesCheck())
        
        # Storage
        self.add_health_check(StorageCheck())
        
        # Database (if available)
        try:
            self.add_health_check(DatabaseConnectionCheck())
        except Exception:
            logger.debug("Database health check not available")
            
    def add_health_check(self, health_check: BaseHealthCheck):
        """Add a health check."""
        self.health_checks.append(health_check)
        logger.info(f"Added health check: {health_check.name}")
        
    def remove_health_check(self, name: str) -> bool:
        """Remove a health check by name."""
        for i, check in enumerate(self.health_checks):
            if check.name == name:
                del self.health_checks[i]
                logger.info(f"Removed health check: {name}")
                return True
        return False
        
    def run_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks."""
        results = {}
        
        for check in self.health_checks:
            try:
                result = check.check()
                results[check.name] = result
                
                # Log results
                if result.status == HealthStatus.CRITICAL:
                    logger.error(f"Health check CRITICAL: {check.name} - {result.message}")
                elif result.status == HealthStatus.WARNING:
                    logger.warning(f"Health check WARNING: {check.name} - {result.message}")
                else:
                    logger.debug(f"Health check OK: {check.name} - {result.message}")
                    
            except Exception as e:
                logger.error(f"Health check error for {check.name}: {e}")
                results[check.name] = HealthCheckResult(
                    component_name=check.name,
                    component_type=check.component_type,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}"
                )
                
        # Store in history
        self.health_history.append(results)
        if len(self.health_history) > self.max_history_size:
            self.health_history = self.health_history[-self.max_history_size:]
            
        # Check for alerts
        self._check_alerts(results)
        
        return results
        
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics."""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory
            memory = psutil.virtual_memory()
            
            # Disk
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)
            
            # Load average (Unix-like systems)
            try:
                load_avg = list(os.getloadavg())
            except (AttributeError, OSError):
                load_avg = [0.0, 0.0, 0.0]
                
            # Process info
            process_count = len(psutil.pids())
            
            # Network connections
            try:
                network_connections = len(psutil.net_connections())
            except (psutil.AccessDenied, OSError):
                network_connections = 0
                
            # Uptime
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                load_average=load_avg,
                uptime_seconds=uptime_seconds,
                process_count=process_count,
                network_connections=network_connections
            )
            
            # Store in history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history = self.metrics_history[-self.max_history_size:]
                
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            raise
            
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        health_results = self.run_health_checks()
        metrics = self.collect_system_metrics()
        
        # Determine overall status
        critical_count = sum(1 for r in health_results.values() if r.status == HealthStatus.CRITICAL)
        warning_count = sum(1 for r in health_results.values() if r.status == HealthStatus.WARNING)
        
        if critical_count > 0:
            overall_status = HealthStatus.CRITICAL
        elif warning_count > 0:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
            
        return {
            "overall_status": overall_status.name,
            "timestamp": datetime.utcnow().isoformat(),
            "health_checks": {name: result.to_dict() for name, result in health_results.items()},
            "system_metrics": metrics.to_dict(),
            "summary": {
                "total_checks": len(health_results),
                "healthy_checks": sum(1 for r in health_results.values() if r.status == HealthStatus.HEALTHY),
                "warning_checks": warning_count,
                "critical_checks": critical_count
            }
        }
        
    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous health monitoring."""
        if self.monitoring_active:
            logger.warning("Health monitoring already active")
            return
            
        self.monitoring_interval = interval_seconds
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info(f"Started health monitoring with {interval_seconds}s interval")
        
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        if not self.monitoring_active:
            return
            
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        logger.info("Stopped health monitoring")
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                health_status = self.get_overall_health()
                
                # Log overall status
                overall_status = health_status["overall_status"]
                if overall_status == "CRITICAL":
                    logger.error("System health CRITICAL")
                elif overall_status == "WARNING":
                    logger.warning("System health WARNING")
                else:
                    logger.debug("System health OK")
                    
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                
            # Wait for next interval
            for _ in range(self.monitoring_interval):
                if not self.monitoring_active:
                    break
                time.sleep(1)
                
    def add_alert_callback(self, callback: Callable[[str, HealthCheckResult], None]):
        """Add callback for health alerts."""
        self.alert_callbacks.append(callback)
        
    def _check_alerts(self, results: Dict[str, HealthCheckResult]):
        """Check for alert conditions."""
        for name, result in results.items():
            if result.status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
                for callback in self.alert_callbacks:
                    try:
                        callback(name, result)
                    except Exception as e:
                        logger.error(f"Error in alert callback: {e}")
                        
    def get_health_trend(self, component_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get health trend for a component."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        trend = []
        for health_snapshot in self.health_history:
            result = health_snapshot.get(component_name)
            if result and result.timestamp >= cutoff_time:
                trend.append({
                    "timestamp": result.timestamp.isoformat(),
                    "status": result.status.name,
                    "duration_ms": result.duration_ms
                })
                
        return trend
        
    def export_health_report(self, file_path: str = None) -> str:
        """Export health report to file."""
        report = {
            "report_timestamp": datetime.utcnow().isoformat(),
            "current_health": self.get_overall_health(),
            "recent_metrics": [m.to_dict() for m in self.metrics_history[-24:]],  # Last 24 metrics
            "health_checks_configured": [check.name for check in self.health_checks]
        }
        
        if file_path is None:
            file_path = f"health_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Health report exported to {file_path}")
        return file_path


# Global health monitor
_global_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor."""
    global _global_health_monitor
    if _global_health_monitor is None:
        _global_health_monitor = HealthMonitor()
    return _global_health_monitor


def health_check_decorator(component_name: str, component_type: ComponentType = ComponentType.CORE):
    """Decorator to automatically register function as health check."""
    def decorator(func):
        class DecoratorHealthCheck(BaseHealthCheck):
            def __init__(self):
                super().__init__(component_name, component_type)
                self.check_func = func
                
            def _perform_check(self):
                return self.check_func()
                
        # Register with global health monitor
        health_monitor = get_health_monitor()
        health_monitor.add_health_check(DecoratorHealthCheck())
        
        return func
    return decorator