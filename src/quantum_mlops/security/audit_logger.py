"""Comprehensive audit logging and security monitoring for quantum MLOps."""

import os
import json
import time
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum, auto
import threading
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class SecurityEventType(Enum):
    """Types of security events."""
    
    # Authentication events
    LOGIN_SUCCESS = auto()
    LOGIN_FAILED = auto()
    LOGOUT = auto()
    TOKEN_CREATED = auto()
    TOKEN_EXPIRED = auto()
    TOKEN_REVOKED = auto()
    
    # Authorization events
    ACCESS_GRANTED = auto()
    ACCESS_DENIED = auto()
    PERMISSION_CHANGED = auto()
    ROLE_ASSIGNED = auto()
    ROLE_REVOKED = auto()
    
    # Quantum events
    QUANTUM_JOB_START = auto()
    QUANTUM_JOB_COMPLETE = auto()
    QUANTUM_JOB_FAILED = auto()
    BACKEND_CONNECTED = auto()
    BACKEND_DISCONNECTED = auto()
    CIRCUIT_VALIDATED = auto()
    CIRCUIT_REJECTED = auto()
    
    # Credential events
    CREDENTIAL_CREATED = auto()
    CREDENTIAL_ACCESSED = auto()
    CREDENTIAL_UPDATED = auto()
    CREDENTIAL_DELETED = auto()
    CREDENTIAL_EXPIRED = auto()
    
    # Security events
    SECURITY_VIOLATION = auto()
    RATE_LIMIT_EXCEEDED = auto()
    SUSPICIOUS_ACTIVITY = auto()
    VULNERABILITY_DETECTED = auto()
    
    # Data events
    DATA_ACCESSED = auto()
    DATA_MODIFIED = auto()
    DATA_DELETED = auto()
    DATA_EXPORTED = auto()
    
    # System events
    SYSTEM_STARTUP = auto()
    SYSTEM_SHUTDOWN = auto()
    CONFIG_CHANGED = auto()
    ERROR_OCCURRED = auto()


@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    
    event_type: SecurityEventType
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: Optional[str] = None
    result: Optional[str] = None  # success, failure, error
    details: Dict[str, Any] = None
    risk_score: int = 0  # 0-10 risk score
    tags: List[str] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['event_type'] = self.event_type.name
        data['timestamp'] = self.timestamp.isoformat()
        return data
        
    def get_event_hash(self) -> str:
        """Get unique hash for event integrity."""
        event_str = f"{self.event_type.name}:{self.timestamp.isoformat()}:{self.user_id}:{self.action}"
        return hashlib.sha256(event_str.encode()).hexdigest()[:16]


class AuditLogStorage:
    """Storage backend for audit logs."""
    
    def __init__(self, log_directory: str = None, max_file_size: int = 10 * 1024 * 1024):
        """Initialize audit log storage."""
        self.log_directory = Path(log_directory or os.getenv('QUANTUM_AUDIT_LOG_DIR', 
                                                             '~/.quantum_mlops/audit_logs')).expanduser()
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size
        
        # Set secure permissions
        try:
            os.chmod(self.log_directory, 0o700)
        except OSError as e:
            logger.warning(f"Could not set secure permissions on audit log directory: {e}")
            
        self.current_log_file = None
        self.current_file_size = 0
        self._lock = threading.Lock()
        
    def _get_current_log_file(self) -> Path:
        """Get current log file path."""
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
        return self.log_directory / f"audit_{date_str}.jsonl"
        
    def _rotate_log_file_if_needed(self, log_file: Path) -> Path:
        """Rotate log file if it's too large."""
        if log_file.exists() and log_file.stat().st_size > self.max_file_size:
            # Create rotated filename
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            rotated_file = log_file.with_name(f"{log_file.stem}_{timestamp}.jsonl")
            log_file.rename(rotated_file)
            
            # Set secure permissions on rotated file
            try:
                os.chmod(rotated_file, 0o600)
            except OSError:
                pass
                
        return log_file
        
    def store_event(self, event: SecurityEvent) -> None:
        """Store security event."""
        with self._lock:
            log_file = self._get_current_log_file()
            log_file = self._rotate_log_file_if_needed(log_file)
            
            # Prepare log entry
            log_entry = {
                **event.to_dict(),
                'event_hash': event.get_event_hash(),
                'log_version': '1.0'
            }
            
            # Append to log file
            try:
                with open(log_file, 'a') as f:
                    json.dump(log_entry, f, separators=(',', ':'))
                    f.write('\n')
                    
                # Set secure permissions on new file
                if not hasattr(self, '_permissions_set'):
                    os.chmod(log_file, 0o600)
                    self._permissions_set = True
                    
            except Exception as e:
                logger.error(f"Failed to write audit log: {e}")
                
    def query_events(self, start_time: datetime = None, end_time: datetime = None,
                    event_types: List[SecurityEventType] = None,
                    user_id: str = None, limit: int = 1000) -> List[SecurityEvent]:
        """Query audit events."""
        events = []
        
        # Determine which log files to search
        if start_time is None:
            start_time = datetime.utcnow() - timedelta(days=30)
        if end_time is None:
            end_time = datetime.utcnow()
            
        # Search log files
        for log_file in self.log_directory.glob("audit_*.jsonl"):
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        if len(events) >= limit:
                            break
                            
                        try:
                            log_entry = json.loads(line.strip())
                            event_time = datetime.fromisoformat(log_entry['timestamp'])
                            
                            # Filter by time
                            if event_time < start_time or event_time > end_time:
                                continue
                                
                            # Filter by event type
                            if event_types:
                                event_type_name = log_entry['event_type']
                                if not any(et.name == event_type_name for et in event_types):
                                    continue
                                    
                            # Filter by user
                            if user_id and log_entry.get('user_id') != user_id:
                                continue
                                
                            # Convert back to SecurityEvent
                            event_data = log_entry.copy()
                            event_data['event_type'] = SecurityEventType[event_data['event_type']]
                            event_data['timestamp'] = datetime.fromisoformat(event_data['timestamp'])
                            
                            # Remove log-specific fields
                            event_data.pop('event_hash', None)
                            event_data.pop('log_version', None)
                            
                            events.append(SecurityEvent(**event_data))
                            
                        except (json.JSONDecodeError, KeyError, ValueError) as e:
                            logger.warning(f"Invalid log entry in {log_file}: {e}")
                            continue
                            
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.error(f"Error reading log file {log_file}: {e}")
                continue
                
        return sorted(events, key=lambda e: e.timestamp, reverse=True)[:limit]


class SecurityMonitor:
    """Real-time security monitoring and alerting."""
    
    def __init__(self, alert_thresholds: Dict[str, Any] = None):
        """Initialize security monitor."""
        self.alert_thresholds = alert_thresholds or {
            'failed_logins_per_hour': 5,
            'access_denied_per_hour': 10,
            'quantum_jobs_per_hour': 100,
            'high_risk_events_per_day': 20
        }
        
        self.event_counters = {}
        self.active_alerts = {}
        
    def process_event(self, event: SecurityEvent) -> List[str]:
        """Process event and generate alerts if needed."""
        alerts = []
        
        # Update counters
        self._update_counters(event)
        
        # Check thresholds
        if event.event_type == SecurityEventType.LOGIN_FAILED:
            if self._check_threshold('failed_logins_per_hour', 'hour'):
                alerts.append("Excessive failed login attempts detected")
                
        elif event.event_type == SecurityEventType.ACCESS_DENIED:
            if self._check_threshold('access_denied_per_hour', 'hour'):
                alerts.append("Excessive access denied events detected")
                
        elif event.event_type in [SecurityEventType.QUANTUM_JOB_START, SecurityEventType.QUANTUM_JOB_COMPLETE]:
            if self._check_threshold('quantum_jobs_per_hour', 'hour'):
                alerts.append("Excessive quantum job activity detected")
                
        # Check for high-risk events
        if event.risk_score >= 7:
            if self._check_threshold('high_risk_events_per_day', 'day'):
                alerts.append("High number of high-risk security events")
                
        # Pattern-based alerts
        pattern_alerts = self._check_patterns(event)
        alerts.extend(pattern_alerts)
        
        return alerts
        
    def _update_counters(self, event: SecurityEvent) -> None:
        """Update event counters."""
        now = datetime.utcnow()
        hour_key = now.strftime('%Y-%m-%d-%H')
        day_key = now.strftime('%Y-%m-%d')
        
        # Initialize counters if needed
        for key in [hour_key, day_key]:
            if key not in self.event_counters:
                self.event_counters[key] = {}
                
        # Update counters
        event_name = event.event_type.name
        self.event_counters[hour_key][event_name] = self.event_counters[hour_key].get(event_name, 0) + 1
        self.event_counters[day_key][event_name] = self.event_counters[day_key].get(event_name, 0) + 1
        
    def _check_threshold(self, threshold_name: str, period: str) -> bool:
        """Check if threshold is exceeded."""
        threshold = self.alert_thresholds.get(threshold_name, float('inf'))
        now = datetime.utcnow()
        
        if period == 'hour':
            period_key = now.strftime('%Y-%m-%d-%H')
        else:  # day
            period_key = now.strftime('%Y-%m-%d')
            
        period_counters = self.event_counters.get(period_key, {})
        
        # Map threshold names to event types
        event_mapping = {
            'failed_logins_per_hour': 'LOGIN_FAILED',
            'access_denied_per_hour': 'ACCESS_DENIED',
            'quantum_jobs_per_hour': ['QUANTUM_JOB_START', 'QUANTUM_JOB_COMPLETE'],
            'high_risk_events_per_day': 'HIGH_RISK'  # Special case
        }
        
        if threshold_name == 'high_risk_events_per_day':
            # Count all high-risk events
            total = sum(count for event_type, count in period_counters.items() 
                       if any(event_type in str(e) for e in period_counters.keys()))
            return total > threshold
        else:
            event_types = event_mapping.get(threshold_name, [])
            if isinstance(event_types, str):
                event_types = [event_types]
                
            total = sum(period_counters.get(event_type, 0) for event_type in event_types)
            return total > threshold
            
    def _check_patterns(self, event: SecurityEvent) -> List[str]:
        """Check for suspicious patterns."""
        alerts = []
        
        # Check for rapid successive failed attempts from same user
        if (event.event_type == SecurityEventType.LOGIN_FAILED and 
            event.user_id and hasattr(self, '_last_failed_login')):
            
            last_event = self._last_failed_login.get(event.user_id)
            if (last_event and 
                (event.timestamp - last_event).total_seconds() < 60):  # Within 1 minute
                alerts.append(f"Rapid failed login attempts for user {event.user_id}")
                
        self._last_failed_login = getattr(self, '_last_failed_login', {})
        if event.event_type == SecurityEventType.LOGIN_FAILED and event.user_id:
            self._last_failed_login[event.user_id] = event.timestamp
            
        return alerts


class AuditLogger:
    """Main audit logging system."""
    
    def __init__(self, storage: AuditLogStorage = None, monitor: SecurityMonitor = None,
                 async_logging: bool = True):
        """Initialize audit logger."""
        self.storage = storage or AuditLogStorage()
        self.monitor = monitor or SecurityMonitor()
        self.async_logging = async_logging
        
        if self.async_logging:
            self._event_queue = Queue()
            self._worker_thread = threading.Thread(target=self._log_worker, daemon=True)
            self._worker_thread.start()
            self._shutdown = False
            
    def _log_worker(self) -> None:
        """Background worker for async logging."""
        while not self._shutdown:
            try:
                event = self._event_queue.get(timeout=1.0)
                if event is None:  # Shutdown signal
                    break
                    
                # Store event
                self.storage.store_event(event)
                
                # Process with monitor
                alerts = self.monitor.process_event(event)
                if alerts:
                    for alert in alerts:
                        logger.warning(f"Security Alert: {alert}")
                        
                self._event_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audit log worker: {e}")
                
    def log_event(self, event: SecurityEvent) -> None:
        """Log security event."""
        if self.async_logging:
            self._event_queue.put(event)
        else:
            self.storage.store_event(event)
            alerts = self.monitor.process_event(event)
            if alerts:
                for alert in alerts:
                    logger.warning(f"Security Alert: {alert}")
                    
    def log_authentication(self, event_type: SecurityEventType, user_id: str,
                          success: bool = True, details: Dict[str, Any] = None,
                          ip_address: str = None, user_agent: str = None) -> None:
        """Log authentication event."""
        event = SecurityEvent(
            event_type=event_type,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            result="success" if success else "failure",
            details=details or {},
            risk_score=1 if success else 3,
            tags=["authentication"]
        )
        self.log_event(event)
        
    def log_authorization(self, event_type: SecurityEventType, user_id: str,
                         resource_type: str, resource_id: str = None,
                         action: str = None, granted: bool = True,
                         details: Dict[str, Any] = None) -> None:
        """Log authorization event."""
        event = SecurityEvent(
            event_type=event_type,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            result="granted" if granted else "denied",
            details=details or {},
            risk_score=1 if granted else 4,
            tags=["authorization"]
        )
        self.log_event(event)
        
    def log_quantum_job_start(self, user_id: str, backend: str, job_id: str,
                             circuits_count: int, estimated_cost: float,
                             details: Dict[str, Any] = None) -> None:
        """Log quantum job start."""
        event = SecurityEvent(
            event_type=SecurityEventType.QUANTUM_JOB_START,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            resource_type="quantum_job",
            resource_id=job_id,
            action="execute",
            result="started",
            details={
                "backend": backend,
                "circuits_count": circuits_count,
                "estimated_cost": estimated_cost,
                **(details or {})
            },
            risk_score=2 if estimated_cost > 10 else 1,
            tags=["quantum", "job"]
        )
        self.log_event(event)
        
    def log_quantum_job_complete(self, user_id: str, job_id: str, success: bool,
                                execution_time: float, actual_cost: float = None,
                                error: str = None, details: Dict[str, Any] = None) -> None:
        """Log quantum job completion."""
        event = SecurityEvent(
            event_type=SecurityEventType.QUANTUM_JOB_COMPLETE if success else SecurityEventType.QUANTUM_JOB_FAILED,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            resource_type="quantum_job",
            resource_id=job_id,
            action="execute",
            result="success" if success else "failure",
            details={
                "execution_time": execution_time,
                "actual_cost": actual_cost,
                "error": error,
                **(details or {})
            },
            risk_score=1 if success else 3,
            tags=["quantum", "job"]
        )
        self.log_event(event)
        
    def log_credential_access(self, user_id: str, credential_name: str,
                             action: str, success: bool = True,
                             details: Dict[str, Any] = None) -> None:
        """Log credential access."""
        event_type = {
            "create": SecurityEventType.CREDENTIAL_CREATED,
            "read": SecurityEventType.CREDENTIAL_ACCESSED,
            "update": SecurityEventType.CREDENTIAL_UPDATED,
            "delete": SecurityEventType.CREDENTIAL_DELETED
        }.get(action, SecurityEventType.CREDENTIAL_ACCESSED)
        
        event = SecurityEvent(
            event_type=event_type,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            resource_type="credential",
            resource_id=credential_name,
            action=action,
            result="success" if success else "failure",
            details=details or {},
            risk_score=5 if action in ["create", "update", "delete"] else 2,
            tags=["credential", "security"]
        )
        self.log_event(event)
        
    def log_security_violation(self, user_id: str, violation_type: str,
                              description: str, risk_score: int = 5,
                              details: Dict[str, Any] = None) -> None:
        """Log security violation."""
        event = SecurityEvent(
            event_type=SecurityEventType.SECURITY_VIOLATION,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            action=violation_type,
            result="violation",
            details={
                "description": description,
                **(details or {})
            },
            risk_score=risk_score,
            tags=["security", "violation", violation_type]
        )
        self.log_event(event)
        
    def query_events(self, **kwargs) -> List[SecurityEvent]:
        """Query audit events."""
        return self.storage.query_events(**kwargs)
        
    def generate_security_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate security report."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        events = self.query_events(start_time=start_time, end_time=end_time, limit=10000)
        
        # Analyze events
        report = {
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "days": days
            },
            "summary": {
                "total_events": len(events),
                "unique_users": len(set(e.user_id for e in events if e.user_id)),
                "high_risk_events": len([e for e in events if e.risk_score >= 7]),
                "failed_events": len([e for e in events if e.result == "failure"])
            },
            "event_types": {},
            "top_users": {},
            "risk_analysis": {
                "average_risk_score": 0,
                "high_risk_users": [],
                "suspicious_patterns": []
            }
        }
        
        # Event type breakdown
        for event in events:
            event_type = event.event_type.name
            report["event_types"][event_type] = report["event_types"].get(event_type, 0) + 1
            
        # User activity
        user_activity = {}
        user_risk_scores = {}
        
        for event in events:
            if event.user_id:
                user_activity[event.user_id] = user_activity.get(event.user_id, 0) + 1
                user_risk_scores[event.user_id] = user_risk_scores.get(event.user_id, [])
                user_risk_scores[event.user_id].append(event.risk_score)
                
        # Top users by activity
        report["top_users"] = dict(sorted(user_activity.items(), 
                                        key=lambda x: x[1], reverse=True)[:10])
        
        # Risk analysis
        if events:
            report["risk_analysis"]["average_risk_score"] = sum(e.risk_score for e in events) / len(events)
            
        # High-risk users
        for user_id, scores in user_risk_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score >= 5 or max(scores) >= 8:
                report["risk_analysis"]["high_risk_users"].append({
                    "user_id": user_id,
                    "average_risk_score": avg_score,
                    "max_risk_score": max(scores),
                    "event_count": len(scores)
                })
                
        return report
        
    def shutdown(self) -> None:
        """Shutdown audit logger."""
        if self.async_logging:
            self._shutdown = True
            self._event_queue.put(None)  # Shutdown signal
            self._worker_thread.join(timeout=5.0)


# Global audit logger
_global_audit_logger: Optional[AuditLogger] = None

def get_audit_logger() -> AuditLogger:
    """Get global audit logger."""
    global _global_audit_logger
    if _global_audit_logger is None:
        _global_audit_logger = AuditLogger()
    return _global_audit_logger