"""Production-grade quantum ML monitoring and alerting system."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Callable, Set, Union
from enum import Enum
from dataclasses import dataclass, asdict
import asyncio
import time
import logging
import json
from datetime import datetime, timedelta
import numpy as np
from collections import deque, defaultdict
import threading
import queue
import warnings

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    
    CRITICAL = "critical"    # System down, data loss imminent
    HIGH = "high"           # Major feature broken, significant impact
    MEDIUM = "medium"       # Minor feature issues, moderate impact
    LOW = "low"            # Performance degradation, minimal impact
    INFO = "info"          # Informational, no action needed

class MetricType(Enum):
    """Types of metrics to monitor."""
    
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    SECURITY = "security"
    BUSINESS = "business"
    QUANTUM = "quantum"
    INFRASTRUCTURE = "infrastructure"

@dataclass
class MetricDataPoint:
    """Single metric data point."""
    
    timestamp: datetime
    metric_name: str
    value: float
    tags: Dict[str, str]
    unit: str = ""
    metadata: Dict[str, Any] = None

@dataclass
class Alert:
    """System alert."""
    
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    tags: Dict[str, str]
    is_resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    acknowledgment_timestamp: Optional[datetime] = None
    escalation_count: int = 0

@dataclass
class MonitoringThresholds:
    """Monitoring thresholds configuration."""
    
    # Performance thresholds
    max_execution_time: float = 300.0  # 5 minutes
    max_queue_time: float = 1800.0     # 30 minutes
    min_throughput: float = 1.0        # 1 job per minute
    max_error_rate: float = 0.05       # 5%
    max_memory_usage: float = 0.8      # 80% of available
    max_cpu_usage: float = 0.9         # 90% of available
    
    # Quantum-specific thresholds
    min_fidelity: float = 0.85
    max_gradient_variance: float = 1.0
    min_convergence_rate: float = 0.01
    max_decoherence_time: float = 100e-6  # 100 microseconds
    
    # Business metrics
    max_cost_per_job: float = 10.0
    min_success_rate: float = 0.95

class MetricsCollector:
    """Collects and aggregates metrics from various sources."""
    
    def __init__(self, collection_interval: float = 60.0):
        self.collection_interval = collection_interval
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.collectors: List[Callable] = []
        self._running = False
        self._collector_thread: Optional[threading.Thread] = None
    
    def add_collector(self, collector_func: Callable[[], Dict[str, Any]]):
        """Add a metrics collector function."""
        self.collectors.append(collector_func)
    
    def start_collection(self):
        """Start metrics collection in background thread."""
        if self._running:
            return
        
        self._running = True
        self._collector_thread = threading.Thread(target=self._collection_loop)
        self._collector_thread.daemon = True
        self._collector_thread.start()
        
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self._running = False
        if self._collector_thread:
            self._collector_thread.join(timeout=5.0)
        
        logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Main metrics collection loop."""
        while self._running:
            try:
                # Collect metrics from all collectors
                for collector in self.collectors:
                    try:
                        metrics = collector()
                        self._process_collected_metrics(metrics)
                    except Exception as e:
                        logger.error(f"Error in metrics collector: {e}")
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(min(self.collection_interval, 60))
    
    def _process_collected_metrics(self, metrics: Dict[str, Any]):
        """Process and store collected metrics."""
        timestamp = datetime.now()
        
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                data_point = MetricDataPoint(
                    timestamp=timestamp,
                    metric_name=metric_name,
                    value=float(value),
                    tags=metrics.get('tags', {}),
                    metadata=metrics.get('metadata', {})
                )
                
                self.metrics_buffer[metric_name].append(data_point)
    
    def get_metric_history(
        self,
        metric_name: str,
        duration: timedelta = timedelta(hours=1)
    ) -> List[MetricDataPoint]:
        """Get metric history for specified duration."""
        if metric_name not in self.metrics_buffer:
            return []
        
        cutoff_time = datetime.now() - duration
        return [
            point for point in self.metrics_buffer[metric_name]
            if point.timestamp >= cutoff_time
        ]
    
    def get_current_value(self, metric_name: str) -> Optional[float]:
        """Get current value for a metric."""
        if metric_name not in self.metrics_buffer or not self.metrics_buffer[metric_name]:
            return None
        
        return self.metrics_buffer[metric_name][-1].value

class AlertManager:
    """Manages alerting and escalation."""
    
    def __init__(self, thresholds: MonitoringThresholds):
        self.thresholds = thresholds
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.alert_rules: List[Dict[str, Any]] = []
        self.notification_channels: List[Callable] = []
        self._setup_default_alert_rules()
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules."""
        
        # Performance alerts
        self.alert_rules.extend([
            {
                'name': 'high_execution_time',
                'metric': 'execution_time',
                'threshold': self.thresholds.max_execution_time,
                'operator': '>',
                'severity': AlertSeverity.HIGH,
                'description': 'Quantum job execution time exceeded threshold'
            },
            {
                'name': 'low_throughput',
                'metric': 'jobs_per_minute',
                'threshold': self.thresholds.min_throughput,
                'operator': '<',
                'severity': AlertSeverity.MEDIUM,
                'description': 'System throughput below expected levels'
            },
            {
                'name': 'high_error_rate',
                'metric': 'error_rate',
                'threshold': self.thresholds.max_error_rate,
                'operator': '>',
                'severity': AlertSeverity.HIGH,
                'description': 'Job error rate exceeded threshold'
            },
            {
                'name': 'memory_usage_high',
                'metric': 'memory_usage_percent',
                'threshold': self.thresholds.max_memory_usage,
                'operator': '>',
                'severity': AlertSeverity.MEDIUM,
                'description': 'Memory usage approaching limits'
            },
            {
                'name': 'cpu_usage_critical',
                'metric': 'cpu_usage_percent',
                'threshold': self.thresholds.max_cpu_usage,
                'operator': '>',
                'severity': AlertSeverity.CRITICAL,
                'description': 'CPU usage critically high'
            }
        ])
        
        # Quantum-specific alerts
        self.alert_rules.extend([
            {
                'name': 'low_fidelity',
                'metric': 'quantum_fidelity',
                'threshold': self.thresholds.min_fidelity,
                'operator': '<',
                'severity': AlertSeverity.HIGH,
                'description': 'Quantum circuit fidelity below acceptable level'
            },
            {
                'name': 'high_gradient_variance',
                'metric': 'gradient_variance',
                'threshold': self.thresholds.max_gradient_variance,
                'operator': '>',
                'severity': AlertSeverity.MEDIUM,
                'description': 'Gradient variance indicating training instability'
            },
            {
                'name': 'decoherence_warning',
                'metric': 'coherence_time',
                'threshold': self.thresholds.max_decoherence_time,
                'operator': '<',
                'severity': AlertSeverity.HIGH,
                'description': 'Quantum coherence time below threshold'
            }
        ])
        
        # Business metrics alerts
        self.alert_rules.extend([
            {
                'name': 'high_cost_per_job',
                'metric': 'cost_per_job_usd',
                'threshold': self.thresholds.max_cost_per_job,
                'operator': '>',
                'severity': AlertSeverity.MEDIUM,
                'description': 'Job execution cost exceeding budget'
            },
            {
                'name': 'low_success_rate',
                'metric': 'success_rate',
                'threshold': self.thresholds.min_success_rate,
                'operator': '<',
                'severity': AlertSeverity.HIGH,
                'description': 'Job success rate below SLA requirements'
            }
        ])
    
    def add_notification_channel(self, channel_func: Callable[[Alert], None]):
        """Add notification channel for alerts."""
        self.notification_channels.append(channel_func)
    
    def evaluate_alerts(self, metrics: Dict[str, float]):
        """Evaluate alert rules against current metrics."""
        new_alerts = []
        resolved_alerts = []
        
        for rule in self.alert_rules:
            metric_name = rule['metric']
            if metric_name not in metrics:
                continue
            
            current_value = metrics[metric_name]
            threshold = rule['threshold']
            operator = rule['operator']
            
            # Check if alert condition is met
            alert_triggered = self._evaluate_condition(current_value, threshold, operator)
            alert_key = f"{rule['name']}_{metric_name}"
            
            if alert_triggered:
                if alert_key not in self.active_alerts:
                    # New alert
                    alert = Alert(
                        alert_id=self._generate_alert_id(),
                        severity=rule['severity'],
                        title=rule['name'].replace('_', ' ').title(),
                        description=rule['description'],
                        metric_name=metric_name,
                        current_value=current_value,
                        threshold_value=threshold,
                        timestamp=datetime.now(),
                        tags={'rule': rule['name']}
                    )
                    
                    self.active_alerts[alert_key] = alert
                    new_alerts.append(alert)
                    
                else:
                    # Update existing alert
                    self.active_alerts[alert_key].current_value = current_value
                    self.active_alerts[alert_key].escalation_count += 1
            
            else:
                if alert_key in self.active_alerts:
                    # Alert resolved
                    alert = self.active_alerts[alert_key]
                    alert.is_resolved = True
                    alert.resolution_timestamp = datetime.now()
                    
                    resolved_alerts.append(alert)
                    del self.active_alerts[alert_key]
        
        # Send notifications
        for alert in new_alerts:
            self._send_alert_notification(alert)
        
        for alert in resolved_alerts:
            self._send_resolution_notification(alert)
            self.alert_history.append(alert)
        
        return new_alerts, resolved_alerts
    
    def _evaluate_condition(self, value: float, threshold: float, operator: str) -> bool:
        """Evaluate alert condition."""
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return abs(value - threshold) < 1e-9
        elif operator == '!=':
            return abs(value - threshold) >= 1e-9
        else:
            logger.warning(f"Unknown operator: {operator}")
            return False
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _send_alert_notification(self, alert: Alert):
        """Send alert notification to all channels."""
        for channel in self.notification_channels:
            try:
                channel(alert)
            except Exception as e:
                logger.error(f"Failed to send alert notification: {e}")
    
    def _send_resolution_notification(self, alert: Alert):
        """Send alert resolution notification."""
        for channel in self.notification_channels:
            try:
                # Most channels can handle resolved alerts with is_resolved=True
                channel(alert)
            except Exception as e:
                logger.error(f"Failed to send resolution notification: {e}")

class QuantumMetricsCollector:
    """Specialized collector for quantum-specific metrics."""
    
    def __init__(self):
        self.circuit_executions = 0
        self.total_execution_time = 0.0
        self.fidelity_samples = deque(maxlen=100)
        self.gradient_variances = deque(maxlen=100)
        self.error_counts = defaultdict(int)
    
    def record_circuit_execution(
        self,
        execution_time: float,
        fidelity: float,
        gradient_variance: float = None,
        success: bool = True
    ):
        """Record quantum circuit execution metrics."""
        self.circuit_executions += 1
        self.total_execution_time += execution_time
        self.fidelity_samples.append(fidelity)
        
        if gradient_variance is not None:
            self.gradient_variances.append(gradient_variance)
        
        if not success:
            self.error_counts['execution_failed'] += 1
    
    def record_quantum_error(self, error_type: str):
        """Record quantum-specific error."""
        self.error_counts[error_type] += 1
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current quantum metrics."""
        metrics = {}
        
        if self.circuit_executions > 0:
            metrics['avg_execution_time'] = self.total_execution_time / self.circuit_executions
            metrics['total_executions'] = self.circuit_executions
        
        if self.fidelity_samples:
            metrics['quantum_fidelity'] = np.mean(list(self.fidelity_samples))
            metrics['fidelity_std'] = np.std(list(self.fidelity_samples))
        
        if self.gradient_variances:
            metrics['gradient_variance'] = np.mean(list(self.gradient_variances))
        
        # Calculate error rates
        total_operations = max(self.circuit_executions, 1)
        for error_type, count in self.error_counts.items():
            metrics[f'{error_type}_rate'] = count / total_operations
        
        # Calculate overall success rate
        failed_operations = sum(self.error_counts.values())
        metrics['success_rate'] = max(0, (total_operations - failed_operations) / total_operations)
        
        return metrics

class SystemResourceCollector:
    """Collects system resource metrics."""
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect system resource metrics."""
        try:
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)
            
            # Network metrics (if available)
            try:
                network = psutil.net_io_counters()
                network_sent_mb = network.bytes_sent / (1024**2)
                network_recv_mb = network.bytes_recv / (1024**2)
            except:
                network_sent_mb = 0
                network_recv_mb = 0
            
            return {
                'cpu_usage_percent': cpu_percent,
                'cpu_count': cpu_count,
                'memory_usage_percent': memory_percent,
                'memory_available_gb': memory_available_gb,
                'disk_usage_percent': disk_percent,
                'disk_free_gb': disk_free_gb,
                'network_sent_mb': network_sent_mb,
                'network_recv_mb': network_recv_mb
            }
        
        except ImportError:
            # psutil not available - return mock data
            logger.warning("psutil not available for system metrics")
            return {
                'cpu_usage_percent': 45.0,
                'memory_usage_percent': 62.0,
                'disk_usage_percent': 78.0
            }
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}

class ProductionMonitoringSystem:
    """Complete production monitoring system."""
    
    def __init__(self, thresholds: MonitoringThresholds = None):
        self.thresholds = thresholds or MonitoringThresholds()
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.thresholds)
        self.quantum_collector = QuantumMetricsCollector()
        self.system_collector = SystemResourceCollector()
        
        # Setup collectors
        self.metrics_collector.add_collector(self.quantum_collector.collect_metrics)
        self.metrics_collector.add_collector(self.system_collector.collect_metrics)
        
        # Setup default notification channels
        self._setup_notification_channels()
    
    def start_monitoring(self):
        """Start the monitoring system."""
        self.metrics_collector.start_collection()
        
        # Start alert evaluation loop
        self._start_alert_evaluation_loop()
        
        logger.info("Production monitoring system started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.metrics_collector.stop_collection()
        self._stop_alert_evaluation_loop()
        
        logger.info("Production monitoring system stopped")
    
    def _setup_notification_channels(self):
        """Setup notification channels."""
        # Console logging channel
        def console_notification(alert: Alert):
            if alert.is_resolved:
                logger.info(f"RESOLVED: {alert.title} - {alert.description}")
            else:
                level = logging.ERROR if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH] else logging.WARNING
                logger.log(level, f"ALERT [{alert.severity.value.upper()}]: {alert.title} - {alert.description} (Value: {alert.current_value}, Threshold: {alert.threshold_value})")
        
        self.alert_manager.add_notification_channel(console_notification)
    
    def _start_alert_evaluation_loop(self):
        """Start alert evaluation in background thread."""
        self._alert_evaluation_running = True
        self._alert_thread = threading.Thread(target=self._alert_evaluation_loop)
        self._alert_thread.daemon = True
        self._alert_thread.start()
    
    def _stop_alert_evaluation_loop(self):
        """Stop alert evaluation loop."""
        self._alert_evaluation_running = False
        if hasattr(self, '_alert_thread'):
            self._alert_thread.join(timeout=5.0)
    
    def _alert_evaluation_loop(self):
        """Main alert evaluation loop."""
        while self._alert_evaluation_running:
            try:
                # Get current metric values
                current_metrics = {}
                
                # Collect from quantum metrics
                quantum_metrics = self.quantum_collector.collect_metrics()
                current_metrics.update(quantum_metrics)
                
                # Collect from system metrics
                system_metrics = self.system_collector.collect_metrics()
                current_metrics.update(system_metrics)
                
                # Evaluate alerts
                self.alert_manager.evaluate_alerts(current_metrics)
                
                time.sleep(30)  # Evaluate alerts every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in alert evaluation loop: {e}")
                time.sleep(60)
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        
        # Get recent metrics
        recent_metrics = {}
        for metric_name in ['quantum_fidelity', 'execution_time', 'success_rate', 'cpu_usage_percent', 'memory_usage_percent']:
            history = self.metrics_collector.get_metric_history(metric_name, timedelta(hours=1))
            if history:
                recent_metrics[metric_name] = {
                    'current': history[-1].value,
                    'history': [{'timestamp': point.timestamp.isoformat(), 'value': point.value} for point in history[-20:]]
                }
        
        # Get active alerts
        active_alerts = [asdict(alert) for alert in self.alert_manager.active_alerts.values()]
        
        # Get system status
        system_status = "healthy"
        if any(alert.severity == AlertSeverity.CRITICAL for alert in self.alert_manager.active_alerts.values()):
            system_status = "critical"
        elif any(alert.severity == AlertSeverity.HIGH for alert in self.alert_manager.active_alerts.values()):
            system_status = "degraded"
        elif any(alert.severity == AlertSeverity.MEDIUM for alert in self.alert_manager.active_alerts.values()):
            system_status = "warning"
        
        return {
            'system_status': system_status,
            'metrics': recent_metrics,
            'active_alerts': active_alerts,
            'alert_count': len(self.alert_manager.active_alerts),
            'total_executions': self.quantum_collector.circuit_executions,
            'uptime_hours': 24,  # Mock uptime
            'timestamp': datetime.now().isoformat()
        }
    
    # Integration methods for quantum ML pipeline
    def record_training_metrics(
        self,
        execution_time: float,
        loss: float,
        accuracy: float,
        fidelity: float,
        gradient_variance: float = None
    ):
        """Record training-specific metrics."""
        self.quantum_collector.record_circuit_execution(
            execution_time=execution_time,
            fidelity=fidelity,
            gradient_variance=gradient_variance,
            success=True
        )
        
        # Record additional training metrics
        timestamp = datetime.now()
        training_metrics = [
            MetricDataPoint(timestamp, 'training_loss', loss, {}),
            MetricDataPoint(timestamp, 'training_accuracy', accuracy, {}),
            MetricDataPoint(timestamp, 'training_time', execution_time, {})
        ]
        
        for metric in training_metrics:
            self.metrics_collector.metrics_buffer[metric.metric_name].append(metric)
    
    def record_quantum_error(self, error_type: str, error_details: str = ""):
        """Record quantum-specific error for monitoring."""
        self.quantum_collector.record_quantum_error(error_type)
        logger.warning(f"Quantum error recorded: {error_type} - {error_details}")

# Export main classes
__all__ = [
    'AlertSeverity',
    'MetricType', 
    'MetricDataPoint',
    'Alert',
    'MonitoringThresholds',
    'MetricsCollector',
    'AlertManager',
    'QuantumMetricsCollector',
    'SystemResourceCollector',
    'ProductionMonitoringSystem'
]