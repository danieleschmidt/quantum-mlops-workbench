"""Security configuration management for quantum MLOps workbench."""

import os
import json
import yaml
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    
    # Authentication settings
    jwt_secret_key: str
    jwt_expiry_hours: int = 1
    refresh_token_days: int = 30
    password_min_length: int = 8
    
    # Authorization settings
    default_role: str = "user"
    admin_approval_required: bool = False
    role_inheritance_enabled: bool = True
    
    # Encryption settings
    encryption_algorithm: str = "fernet"
    key_rotation_days: int = 90
    data_encryption_enabled: bool = True
    transit_encryption_enabled: bool = True
    
    # Quantum security settings
    max_qubits_per_user: int = 50
    max_jobs_per_hour: int = 100
    max_cost_per_day: float = 100.0
    circuit_validation_enabled: bool = True
    parameter_sanitization_enabled: bool = True
    
    # Audit settings
    audit_logging_enabled: bool = True
    audit_log_retention_days: int = 365
    security_monitoring_enabled: bool = True
    alert_thresholds: Dict[str, int] = None
    
    # Network security settings
    ssl_required: bool = True
    min_tls_version: str = "1.2"
    allowed_origins: List[str] = None
    rate_limiting_enabled: bool = True
    
    # Compliance settings
    gdpr_compliance: bool = False
    hipaa_compliance: bool = False
    soc2_compliance: bool = False
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                "failed_logins_per_hour": 5,
                "access_denied_per_hour": 10,
                "quantum_jobs_per_hour": 50
            }
        if self.allowed_origins is None:
            self.allowed_origins = ["http://localhost:3000", "http://localhost:8000"]


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration."""
    
    name: str
    security_level: str  # development, staging, production
    debug_enabled: bool = False
    logging_level: str = "INFO"
    database_encryption: bool = True
    external_services: Dict[str, Dict[str, str]] = None
    
    def __post_init__(self):
        if self.external_services is None:
            self.external_services = {}


class SecurityConfigManager:
    """Manages security configuration across environments."""
    
    def __init__(self, config_path: str = None, environment: str = None):
        """Initialize security configuration manager."""
        self.config_path = Path(config_path or os.getenv('QUANTUM_SECURITY_CONFIG', 
                                                        '~/.quantum_mlops/security_config.yaml')).expanduser()
        self.environment = environment or os.getenv('QUANTUM_ENVIRONMENT', 'development')
        
        # Create config directory
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set secure permissions
        try:
            os.chmod(self.config_path.parent, 0o700)
            if self.config_path.exists():
                os.chmod(self.config_path, 0o600)
        except OSError as e:
            logger.warning(f"Could not set secure permissions: {e}")
            
        self._configs: Dict[str, SecurityConfig] = {}
        self._env_configs: Dict[str, EnvironmentConfig] = {}
        self._load_configurations()
        
    def _load_configurations(self) -> None:
        """Load configurations from file."""
        if not self.config_path.exists():
            self._create_default_configurations()
            return
            
        try:
            with open(self.config_path, 'r') as f:
                data = yaml.safe_load(f) or {}
                
            # Load security configs
            security_configs = data.get('security', {})
            for env_name, config_data in security_configs.items():
                self._configs[env_name] = SecurityConfig(**config_data)
                
            # Load environment configs
            env_configs = data.get('environments', {})
            for env_name, config_data in env_configs.items():
                self._env_configs[env_name] = EnvironmentConfig(**config_data)
                
        except Exception as e:
            logger.error(f"Failed to load security configuration: {e}")
            self._create_default_configurations()
            
    def _create_default_configurations(self) -> None:
        """Create default security configurations."""
        # Generate secure JWT secret
        import secrets
        jwt_secret = secrets.token_urlsafe(32)
        
        # Development configuration
        dev_config = SecurityConfig(
            jwt_secret_key=jwt_secret,
            jwt_expiry_hours=8,  # Longer for development
            max_qubits_per_user=20,
            max_jobs_per_hour=200,
            max_cost_per_day=50.0,
            ssl_required=False,  # Disabled for development
            audit_log_retention_days=30
        )
        
        # Staging configuration
        staging_config = SecurityConfig(
            jwt_secret_key=jwt_secret,
            jwt_expiry_hours=2,
            max_qubits_per_user=30,
            max_jobs_per_hour=150,
            max_cost_per_day=75.0,
            ssl_required=True,
            audit_log_retention_days=90
        )
        
        # Production configuration
        prod_config = SecurityConfig(
            jwt_secret_key=jwt_secret,
            jwt_expiry_hours=1,
            admin_approval_required=True,
            max_qubits_per_user=50,
            max_jobs_per_hour=100,
            max_cost_per_day=100.0,
            ssl_required=True,
            gdpr_compliance=True,
            soc2_compliance=True,
            audit_log_retention_days=365
        )
        
        self._configs = {
            'development': dev_config,
            'staging': staging_config,
            'production': prod_config
        }
        
        # Environment configurations
        self._env_configs = {
            'development': EnvironmentConfig(
                name='development',
                security_level='development',
                debug_enabled=True,
                logging_level='DEBUG',
                database_encryption=False
            ),
            'staging': EnvironmentConfig(
                name='staging',
                security_level='staging',
                debug_enabled=False,
                logging_level='INFO',
                database_encryption=True
            ),
            'production': EnvironmentConfig(
                name='production',
                security_level='production',
                debug_enabled=False,
                logging_level='WARNING',
                database_encryption=True
            )
        }
        
        self._save_configurations()
        
    def _save_configurations(self) -> None:
        """Save configurations to file."""
        try:
            data = {
                'security': {name: asdict(config) for name, config in self._configs.items()},
                'environments': {name: asdict(config) for name, config in self._env_configs.items()},
                'metadata': {
                    'created_at': datetime.utcnow().isoformat(),
                    'version': '1.0'
                }
            }
            
            # Write atomically
            temp_path = self.config_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
                
            os.chmod(temp_path, 0o600)
            temp_path.replace(self.config_path)
            
        except Exception as e:
            logger.error(f"Failed to save security configuration: {e}")
            
    def get_security_config(self, environment: str = None) -> SecurityConfig:
        """Get security configuration for environment."""
        env = environment or self.environment
        
        if env not in self._configs:
            logger.warning(f"No security config for environment {env}, using development")
            env = 'development'
            
        return self._configs[env]
        
    def get_environment_config(self, environment: str = None) -> EnvironmentConfig:
        """Get environment configuration."""
        env = environment or self.environment
        
        if env not in self._env_configs:
            logger.warning(f"No environment config for {env}, using development")
            env = 'development'
            
        return self._env_configs[env]
        
    def update_security_config(self, environment: str,
                              config_updates: Dict[str, Any]) -> None:
        """Update security configuration."""
        if environment not in self._configs:
            raise ValueError(f"Unknown environment: {environment}")
            
        config = self._configs[environment]
        
        # Update configuration fields
        for key, value in config_updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"Unknown config field: {key}")
                
        self._save_configurations()
        logger.info(f"Updated security config for {environment}")
        
    def validate_configuration(self, environment: str = None) -> Dict[str, Any]:
        """Validate security configuration."""
        env = environment or self.environment
        config = self.get_security_config(env)
        env_config = self.get_environment_config(env)
        
        issues = []
        warnings = []
        
        # Validate JWT settings
        if not config.jwt_secret_key or len(config.jwt_secret_key) < 32:
            issues.append("JWT secret key is too short or missing")
            
        if config.jwt_expiry_hours > 24:
            warnings.append("JWT expiry time is very long")
            
        # Validate quantum settings
        if config.max_qubits_per_user > 100:
            warnings.append("High qubit limit may impact performance")
            
        if config.max_cost_per_day > 1000:
            warnings.append("High cost limit may lead to unexpected charges")
            
        # Validate security settings for production
        if env_config.security_level == 'production':
            if not config.ssl_required:
                issues.append("SSL is required for production environment")
                
            if config.jwt_expiry_hours > 2:
                warnings.append("JWT expiry time should be shorter in production")
                
            if env_config.debug_enabled:
                issues.append("Debug mode should be disabled in production")
                
        # Validate compliance settings
        if config.gdpr_compliance and config.audit_log_retention_days < 365:
            warnings.append("GDPR compliance may require longer audit log retention")
            
        return {
            "environment": env,
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "security_level": env_config.security_level
        }
        
    def generate_environment_variables(self, environment: str = None) -> Dict[str, str]:
        """Generate environment variables from configuration."""
        env = environment or self.environment
        config = self.get_security_config(env)
        env_config = self.get_environment_config(env)
        
        env_vars = {
            # Authentication
            "JWT_SECRET_KEY": config.jwt_secret_key,
            "JWT_EXPIRY_HOURS": str(config.jwt_expiry_hours),
            "PASSWORD_MIN_LENGTH": str(config.password_min_length),
            
            # Authorization
            "DEFAULT_ROLE": config.default_role,
            "ADMIN_APPROVAL_REQUIRED": str(config.admin_approval_required).lower(),
            
            # Quantum limits
            "MAX_QUBITS_PER_USER": str(config.max_qubits_per_user),
            "MAX_JOBS_PER_HOUR": str(config.max_jobs_per_hour),
            "MAX_COST_PER_DAY": str(config.max_cost_per_day),
            
            # Security features
            "CIRCUIT_VALIDATION_ENABLED": str(config.circuit_validation_enabled).lower(),
            "AUDIT_LOGGING_ENABLED": str(config.audit_logging_enabled).lower(),
            "SSL_REQUIRED": str(config.ssl_required).lower(),
            
            # Environment
            "QUANTUM_ENVIRONMENT": env,
            "DEBUG": str(env_config.debug_enabled).lower(),
            "LOG_LEVEL": env_config.logging_level
        }
        
        return env_vars
        
    def export_configuration(self, output_file: Path, format: str = "yaml") -> None:
        """Export configuration to file."""
        data = {
            'security': {name: asdict(config) for name, config in self._configs.items()},
            'environments': {name: asdict(config) for name, config in self._env_configs.items()}
        }
        
        with open(output_file, 'w') as f:
            if format.lower() == "yaml":
                yaml.dump(data, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        logger.info(f"Configuration exported to {output_file}")
        
    def import_configuration(self, input_file: Path) -> None:
        """Import configuration from file."""
        with open(input_file, 'r') as f:
            if input_file.suffix.lower() in ['.yml', '.yaml']:
                data = yaml.safe_load(f)
            elif input_file.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {input_file.suffix}")
                
        # Import security configs
        security_configs = data.get('security', {})
        for env_name, config_data in security_configs.items():
            self._configs[env_name] = SecurityConfig(**config_data)
            
        # Import environment configs
        env_configs = data.get('environments', {})
        for env_name, config_data in env_configs.items():
            self._env_configs[env_name] = EnvironmentConfig(**config_data)
            
        self._save_configurations()
        logger.info(f"Configuration imported from {input_file}")
        
    def get_compliance_settings(self, environment: str = None) -> Dict[str, bool]:
        """Get compliance settings for environment."""
        config = self.get_security_config(environment)
        
        return {
            "gdpr_compliance": config.gdpr_compliance,
            "hipaa_compliance": config.hipaa_compliance,
            "soc2_compliance": config.soc2_compliance
        }
        
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate security configuration report."""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "environments": {},
            "overall_security_level": "unknown",
            "recommendations": []
        }
        
        security_levels = []
        
        for env_name in self._configs.keys():
            validation = self.validate_configuration(env_name)
            config = self.get_security_config(env_name)
            env_config = self.get_environment_config(env_name)
            
            report["environments"][env_name] = {
                "validation": validation,
                "security_features": {
                    "ssl_required": config.ssl_required,
                    "audit_logging": config.audit_logging_enabled,
                    "circuit_validation": config.circuit_validation_enabled,
                    "data_encryption": config.data_encryption_enabled,
                    "admin_approval": config.admin_approval_required
                },
                "compliance": self.get_compliance_settings(env_name)
            }
            
            security_levels.append(env_config.security_level)
            
        # Determine overall security level
        if "production" in security_levels:
            report["overall_security_level"] = "production"
        elif "staging" in security_levels:
            report["overall_security_level"] = "staging"
        else:
            report["overall_security_level"] = "development"
            
        # Generate recommendations
        for env_name, env_data in report["environments"].items():
            if env_data["validation"]["issues"]:
                report["recommendations"].append(f"Fix security issues in {env_name} environment")
                
        return report


# Global security config manager
_global_config_manager: Optional[SecurityConfigManager] = None

def get_security_config_manager() -> SecurityConfigManager:
    """Get global security configuration manager."""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = SecurityConfigManager()
    return _global_config_manager