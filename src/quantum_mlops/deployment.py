"""Deployment and production utilities for quantum MLOps workbench."""

import os
import yaml
import json
import subprocess
import shutil
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import logging
import tempfile

from .exceptions import QuantumMLOpsException, ErrorCategory, ErrorSeverity
from .logging_config import get_logger
from .health import HealthMonitor

logger = get_logger("deployment")


@dataclass
class DeploymentConfig:
    """Configuration for deployment."""
    
    # Application settings
    app_name: str = "quantum-mlops"
    app_version: str = "0.1.0"
    environment: str = "production"  # development, staging, production
    
    # Container settings
    container_registry: str = "docker.io"
    container_image: str = "quantum-mlops"
    container_tag: str = "latest"
    
    # Resource requirements
    cpu_request: str = "500m"
    cpu_limit: str = "2"
    memory_request: str = "1Gi"
    memory_limit: str = "4Gi"
    storage_size: str = "10Gi"
    
    # Scaling settings
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    
    # Database settings
    database_url: str = ""
    database_max_connections: int = 100
    
    # Quantum backend settings
    quantum_backends: List[str] = field(default_factory=lambda: ["simulator"])
    quantum_timeout: int = 300
    
    # Security settings
    enable_tls: bool = True
    tls_cert_path: str = "/etc/certs/tls.crt"
    tls_key_path: str = "/etc/certs/tls.key"
    
    # Monitoring settings
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_path: str = "/health"
    
    # Custom environment variables
    env_vars: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class DockerImageBuilder:
    """Builder for Docker images."""
    
    def __init__(self, project_root: Path):
        """Initialize Docker image builder."""
        self.project_root = project_root
        
    def generate_dockerfile(self, config: DeploymentConfig) -> str:
        """Generate Dockerfile content."""
        dockerfile_content = f"""# Multi-stage build for quantum MLOps workbench
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    gcc \\
    g++ \\
    gfortran \\
    libopenblas-dev \\
    liblapack-dev \\
    pkg-config \\
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt requirements-lock.txt ./
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements-lock.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    libopenblas0 \\
    liblapack3 \\
    libgomp1 \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN groupadd -r quantum && useradd -r -g quantum -d /app -s /bin/bash quantum

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=quantum:quantum src/ ./src/
COPY --chown=quantum:quantum pyproject.toml ./
COPY --chown=quantum:quantum README.md ./

# Install application
RUN pip install -e .

# Create directories
RUN mkdir -p /app/data /app/logs /app/models && \\
    chown -R quantum:quantum /app

# Switch to non-root user
USER quantum

# Set environment variables
ENV PYTHONPATH="/app/src"
ENV QUANTUM_MLOPS_CONFIG_PATH="/app/config"
ENV QUANTUM_MLOPS_DATA_PATH="/app/data"
ENV QUANTUM_MLOPS_LOG_PATH="/app/logs"
ENV QUANTUM_MLOPS_MODEL_PATH="/app/models"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000{config.health_check_path} || exit 1

# Expose ports
EXPOSE 8000 {config.metrics_port}

# Default command
CMD ["python", "-m", "quantum_mlops.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
"""
        return dockerfile_content
        
    def generate_dockerignore(self) -> str:
        """Generate .dockerignore content."""
        return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Git
.git/
.gitignore

# Documentation
docs/_build/

# Temporary files
*.tmp
*.temp

# Local data
data/
logs/
models/
*.db
*.sqlite

# CI/CD
.github/
.gitlab-ci.yml

# Development
docker-compose.dev.yml
Dockerfile.dev
"""

    def build_image(self, config: DeploymentConfig, build_args: Dict[str, str] = None,
                   no_cache: bool = False) -> bool:
        """Build Docker image."""
        try:
            # Generate Dockerfile
            dockerfile_content = self.generate_dockerfile(config)
            dockerfile_path = self.project_root / "Dockerfile"
            
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
                
            # Generate .dockerignore
            dockerignore_content = self.generate_dockerignore()
            dockerignore_path = self.project_root / ".dockerignore"
            
            with open(dockerignore_path, 'w') as f:
                f.write(dockerignore_content)
                
            # Build command
            image_tag = f"{config.container_registry}/{config.container_image}:{config.container_tag}"
            cmd = ["docker", "build", "-t", image_tag, "."]
            
            if no_cache:
                cmd.append("--no-cache")
                
            if build_args:
                for key, value in build_args.items():
                    cmd.extend(["--build-arg", f"{key}={value}"])
                    
            # Execute build
            logger.info(f"Building Docker image: {image_tag}")
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully built Docker image: {image_tag}")
                return True
            else:
                logger.error(f"Docker build failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Docker build error: {e}")
            return False
            
    def push_image(self, config: DeploymentConfig) -> bool:
        """Push Docker image to registry."""
        try:
            image_tag = f"{config.container_registry}/{config.container_image}:{config.container_tag}"
            
            cmd = ["docker", "push", image_tag]
            
            logger.info(f"Pushing Docker image: {image_tag}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully pushed Docker image: {image_tag}")
                return True
            else:
                logger.error(f"Docker push failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Docker push error: {e}")
            return False


class KubernetesDeployer:
    """Kubernetes deployment generator and manager."""
    
    def __init__(self, output_dir: Path = None):
        """Initialize Kubernetes deployer."""
        self.output_dir = output_dir or Path("./k8s")
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_namespace(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Generate Kubernetes namespace manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": f"{config.app_name}-{config.environment}",
                "labels": {
                    "app": config.app_name,
                    "environment": config.environment
                }
            }
        }
        
    def generate_configmap(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Generate ConfigMap manifest."""
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{config.app_name}-config",
                "namespace": f"{config.app_name}-{config.environment}"
            },
            "data": {
                "quantum_backends": ",".join(config.quantum_backends),
                "quantum_timeout": str(config.quantum_timeout),
                "database_max_connections": str(config.database_max_connections),
                "metrics_enabled": str(config.enable_metrics).lower(),
                "metrics_port": str(config.metrics_port),
                **config.env_vars
            }
        }
        
    def generate_secret(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Generate Secret manifest."""
        import base64
        
        # Encode sensitive data
        secret_data = {}
        if config.database_url:
            secret_data["database_url"] = base64.b64encode(config.database_url.encode()).decode()
            
        return {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": f"{config.app_name}-secrets",
                "namespace": f"{config.app_name}-{config.environment}"
            },
            "type": "Opaque",
            "data": secret_data
        }
        
    def generate_deployment(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Generate Deployment manifest."""
        image = f"{config.container_registry}/{config.container_image}:{config.container_tag}"
        
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{config.app_name}-deployment",
                "namespace": f"{config.app_name}-{config.environment}",
                "labels": {
                    "app": config.app_name,
                    "environment": config.environment,
                    "version": config.app_version
                }
            },
            "spec": {
                "replicas": config.min_replicas,
                "selector": {
                    "matchLabels": {
                        "app": config.app_name,
                        "environment": config.environment
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": config.app_name,
                            "environment": config.environment,
                            "version": config.app_version
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": config.app_name,
                            "image": image,
                            "imagePullPolicy": "Always",
                            "ports": [
                                {
                                    "containerPort": 8000,
                                    "name": "http"
                                },
                                {
                                    "containerPort": config.metrics_port,
                                    "name": "metrics"
                                }
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": config.cpu_request,
                                    "memory": config.memory_request
                                },
                                "limits": {
                                    "cpu": config.cpu_limit,
                                    "memory": config.memory_limit
                                }
                            },
                            "env": [
                                {
                                    "name": "ENVIRONMENT",
                                    "value": config.environment
                                },
                                {
                                    "name": "DATABASE_URL",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": f"{config.app_name}-secrets",
                                            "key": "database_url"
                                        }
                                    }
                                }
                            ],
                            "envFrom": [
                                {
                                    "configMapRef": {
                                        "name": f"{config.app_name}-config"
                                    }
                                }
                            ],
                            "volumeMounts": [
                                {
                                    "name": "data-storage",
                                    "mountPath": "/app/data"
                                },
                                {
                                    "name": "model-storage",
                                    "mountPath": "/app/models"
                                }
                            ],
                            "livenessProbe": {
                                "httpGet": {
                                    "path": config.health_check_path,
                                    "port": 8000
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": config.health_check_path,
                                    "port": 8000
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }],
                        "volumes": [
                            {
                                "name": "data-storage",
                                "persistentVolumeClaim": {
                                    "claimName": f"{config.app_name}-data-pvc"
                                }
                            },
                            {
                                "name": "model-storage",
                                "persistentVolumeClaim": {
                                    "claimName": f"{config.app_name}-model-pvc"
                                }
                            }
                        ]
                    }
                }
            }
        }
        
        # Add TLS configuration if enabled
        if config.enable_tls:
            tls_mount = {
                "name": "tls-certs",
                "mountPath": "/etc/certs",
                "readOnly": True
            }
            deployment["spec"]["template"]["spec"]["containers"][0]["volumeMounts"].append(tls_mount)
            
            tls_volume = {
                "name": "tls-certs",
                "secret": {
                    "secretName": f"{config.app_name}-tls"
                }
            }
            deployment["spec"]["template"]["spec"]["volumes"].append(tls_volume)
            
        return deployment
        
    def generate_service(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Generate Service manifest."""
        service_ports = [
            {
                "name": "http",
                "port": 80,
                "targetPort": 8000,
                "protocol": "TCP"
            },
            {
                "name": "metrics",
                "port": config.metrics_port,
                "targetPort": config.metrics_port,
                "protocol": "TCP"
            }
        ]
        
        if config.enable_tls:
            service_ports.append({
                "name": "https",
                "port": 443,
                "targetPort": 8443,
                "protocol": "TCP"
            })
            
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{config.app_name}-service",
                "namespace": f"{config.app_name}-{config.environment}",
                "labels": {
                    "app": config.app_name,
                    "environment": config.environment
                }
            },
            "spec": {
                "selector": {
                    "app": config.app_name,
                    "environment": config.environment
                },
                "ports": service_ports,
                "type": "ClusterIP"
            }
        }
        
    def generate_hpa(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Generate Horizontal Pod Autoscaler manifest."""
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{config.app_name}-hpa",
                "namespace": f"{config.app_name}-{config.environment}"
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"{config.app_name}-deployment"
                },
                "minReplicas": config.min_replicas,
                "maxReplicas": config.max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": config.target_cpu_utilization
                            }
                        }
                    }
                ]
            }
        }
        
    def generate_pvc(self, config: DeploymentConfig, name: str, size: str) -> Dict[str, Any]:
        """Generate PersistentVolumeClaim manifest."""
        return {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": f"{config.app_name}-{name}-pvc",
                "namespace": f"{config.app_name}-{config.environment}"
            },
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "resources": {
                    "requests": {
                        "storage": size
                    }
                }
            }
        }
        
    def generate_ingress(self, config: DeploymentConfig, hostname: str) -> Dict[str, Any]:
        """Generate Ingress manifest."""
        ingress = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": f"{config.app_name}-ingress",
                "namespace": f"{config.app_name}-{config.environment}",
                "annotations": {
                    "nginx.ingress.kubernetes.io/rewrite-target": "/",
                    "nginx.ingress.kubernetes.io/ssl-redirect": "true" if config.enable_tls else "false"
                }
            },
            "spec": {
                "rules": [
                    {
                        "host": hostname,
                        "http": {
                            "paths": [
                                {
                                    "path": "/",
                                    "pathType": "Prefix",
                                    "backend": {
                                        "service": {
                                            "name": f"{config.app_name}-service",
                                            "port": {
                                                "number": 80
                                            }
                                        }
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }
        
        if config.enable_tls:
            ingress["spec"]["tls"] = [
                {
                    "hosts": [hostname],
                    "secretName": f"{config.app_name}-tls"
                }
            ]
            
        return ingress
        
    def generate_all_manifests(self, config: DeploymentConfig, hostname: str = None) -> Dict[str, Any]:
        """Generate all Kubernetes manifests."""
        manifests = {
            "namespace": self.generate_namespace(config),
            "configmap": self.generate_configmap(config),
            "secret": self.generate_secret(config),
            "deployment": self.generate_deployment(config),
            "service": self.generate_service(config),
            "hpa": self.generate_hpa(config),
            "data_pvc": self.generate_pvc(config, "data", config.storage_size),
            "model_pvc": self.generate_pvc(config, "model", config.storage_size)
        }
        
        if hostname:
            manifests["ingress"] = self.generate_ingress(config, hostname)
            
        return manifests
        
    def write_manifests(self, manifests: Dict[str, Any], config: DeploymentConfig):
        """Write manifests to YAML files."""
        for name, manifest in manifests.items():
            file_path = self.output_dir / f"{name}.yaml"
            
            with open(file_path, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False, indent=2)
                
            logger.info(f"Generated Kubernetes manifest: {file_path}")
            
    def deploy_to_cluster(self, config: DeploymentConfig, kubectl_context: str = None) -> bool:
        """Deploy to Kubernetes cluster."""
        try:
            kubectl_cmd = ["kubectl"]
            
            if kubectl_context:
                kubectl_cmd.extend(["--context", kubectl_context])
                
            # Apply namespace first
            namespace_file = self.output_dir / "namespace.yaml"
            if namespace_file.exists():
                cmd = kubectl_cmd + ["apply", "-f", str(namespace_file)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"Failed to create namespace: {result.stderr}")
                    return False
                    
            # Apply all other manifests
            cmd = kubectl_cmd + ["apply", "-f", str(self.output_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Successfully deployed to Kubernetes cluster")
                return True
            else:
                logger.error(f"Kubernetes deployment failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Kubernetes deployment error: {e}")
            return False


class ProductionManager:
    """Manager for production deployment operations."""
    
    def __init__(self, project_root: Path):
        """Initialize production manager."""
        self.project_root = project_root
        self.docker_builder = DockerImageBuilder(project_root)
        self.k8s_deployer = KubernetesDeployer()
        
    def create_deployment_package(self, config: DeploymentConfig, 
                                 hostname: str = None) -> Path:
        """Create complete deployment package."""
        try:
            # Create deployment directory
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            package_dir = self.project_root / f"deployment_{config.environment}_{timestamp}"
            package_dir.mkdir(exist_ok=True)
            
            # Generate Docker files
            dockerfile_content = self.docker_builder.generate_dockerfile(config)
            with open(package_dir / "Dockerfile", 'w') as f:
                f.write(dockerfile_content)
                
            dockerignore_content = self.docker_builder.generate_dockerignore()
            with open(package_dir / ".dockerignore", 'w') as f:
                f.write(dockerignore_content)
                
            # Generate Kubernetes manifests
            k8s_dir = package_dir / "k8s"
            k8s_deployer = KubernetesDeployer(k8s_dir)
            manifests = k8s_deployer.generate_all_manifests(config, hostname)
            k8s_deployer.write_manifests(manifests, config)
            
            # Generate deployment script
            deploy_script = self._generate_deployment_script(config)
            script_path = package_dir / "deploy.sh"
            with open(script_path, 'w') as f:
                f.write(deploy_script)
            script_path.chmod(0o755)
            
            # Generate configuration file
            config_path = package_dir / "deployment_config.json"
            with open(config_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
                
            # Generate README
            readme_content = self._generate_deployment_readme(config)
            with open(package_dir / "README.md", 'w') as f:
                f.write(readme_content)
                
            logger.info(f"Created deployment package: {package_dir}")
            return package_dir
            
        except Exception as e:
            logger.error(f"Failed to create deployment package: {e}")
            raise
            
    def _generate_deployment_script(self, config: DeploymentConfig) -> str:
        """Generate deployment script."""
        return f"""#!/bin/bash
set -e

echo "Deploying {config.app_name} version {config.app_version}"
echo "Environment: {config.environment}"
echo "Timestamp: $(date)"

# Build Docker image
echo "Building Docker image..."
docker build -t {config.container_registry}/{config.container_image}:{config.container_tag} .

# Push Docker image
echo "Pushing Docker image..."
docker push {config.container_registry}/{config.container_image}:{config.container_tag}

# Deploy to Kubernetes
echo "Deploying to Kubernetes..."
kubectl apply -f k8s/

# Wait for deployment to be ready
echo "Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/{config.app_name}-deployment \\
    -n {config.app_name}-{config.environment}

echo "Deployment completed successfully!"

# Show deployment status
echo "Deployment status:"
kubectl get pods -n {config.app_name}-{config.environment}
kubectl get svc -n {config.app_name}-{config.environment}

echo "Health check:"
kubectl exec -n {config.app_name}-{config.environment} \\
    deployment/{config.app_name}-deployment -- \\
    curl -f http://localhost:8000{config.health_check_path} || echo "Health check failed"
"""

    def _generate_deployment_readme(self, config: DeploymentConfig) -> str:
        """Generate deployment README."""
        return f"""# {config.app_name.title()} Deployment Package

## Environment: {config.environment}
## Version: {config.app_version}

### Prerequisites

1. Docker installed and configured
2. kubectl installed and configured with cluster access
3. Access to container registry: {config.container_registry}

### Deployment Instructions

1. **Build and Push Docker Image**
   ```bash
   docker build -t {config.container_registry}/{config.container_image}:{config.container_tag} .
   docker push {config.container_registry}/{config.container_image}:{config.container_tag}
   ```

2. **Deploy to Kubernetes**
   ```bash
   kubectl apply -f k8s/
   ```

3. **Automated Deployment**
   ```bash
   ./deploy.sh
   ```

### Configuration

The deployment is configured with the following settings:

- **Resources**: {config.cpu_request} CPU, {config.memory_request} Memory
- **Scaling**: {config.min_replicas}-{config.max_replicas} replicas
- **Storage**: {config.storage_size} persistent storage
- **Quantum Backends**: {', '.join(config.quantum_backends)}

### Health Checks

- **Health Check Endpoint**: {config.health_check_path}
- **Metrics Port**: {config.metrics_port}

### Troubleshooting

1. **Check Pod Status**
   ```bash
   kubectl get pods -n {config.app_name}-{config.environment}
   ```

2. **View Logs**
   ```bash
   kubectl logs -n {config.app_name}-{config.environment} deployment/{config.app_name}-deployment
   ```

3. **Access Pod Shell**
   ```bash
   kubectl exec -it -n {config.app_name}-{config.environment} deployment/{config.app_name}-deployment -- /bin/bash
   ```

4. **Port Forward for Local Access**
   ```bash
   kubectl port-forward -n {config.app_name}-{config.environment} svc/{config.app_name}-service 8080:80
   ```

### Scaling

To scale the deployment:
```bash
kubectl scale deployment/{config.app_name}-deployment --replicas=5 -n {config.app_name}-{config.environment}
```

### Updates

To update the deployment:
1. Build new Docker image with new tag
2. Update the image tag in k8s/deployment.yaml
3. Apply the updated manifest:
   ```bash
   kubectl apply -f k8s/deployment.yaml
   ```

### Monitoring

- Metrics are available at port {config.metrics_port}
- Health checks at {config.health_check_path}
- Use Prometheus/Grafana for comprehensive monitoring
"""

    def perform_production_checks(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Perform production readiness checks."""
        checks = {
            "docker_available": self._check_docker(),
            "kubectl_available": self._check_kubectl(),
            "container_registry_access": self._check_registry_access(config),
            "kubernetes_cluster_access": self._check_cluster_access(),
            "required_secrets": self._check_required_secrets(config),
            "resource_requirements": self._check_resource_requirements(config)
        }
        
        all_passed = all(checks.values())
        
        return {
            "ready_for_production": all_passed,
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True)
            return result.returncode == 0
        except:
            return False
            
    def _check_kubectl(self) -> bool:
        """Check if kubectl is available."""
        try:
            result = subprocess.run(["kubectl", "version", "--client"], capture_output=True)
            return result.returncode == 0
        except:
            return False
            
    def _check_registry_access(self, config: DeploymentConfig) -> bool:
        """Check container registry access."""
        try:
            # Try to pull a small public image to test registry access
            result = subprocess.run(["docker", "pull", "hello-world"], capture_output=True)
            return result.returncode == 0
        except:
            return False
            
    def _check_cluster_access(self) -> bool:
        """Check Kubernetes cluster access."""
        try:
            result = subprocess.run(["kubectl", "cluster-info"], capture_output=True)
            return result.returncode == 0
        except:
            return False
            
    def _check_required_secrets(self, config: DeploymentConfig) -> bool:
        """Check if required secrets are configured."""
        required_vars = ["DATABASE_URL"] if config.database_url else []
        
        for var in required_vars:
            if not os.getenv(var):
                return False
                
        return True
        
    def _check_resource_requirements(self, config: DeploymentConfig) -> bool:
        """Check resource requirements."""
        # This is a simplified check - in production you'd want to verify
        # cluster capacity, quotas, etc.
        return True


def create_production_deployment(project_root: Path, environment: str = "production",
                               hostname: str = None, **config_overrides) -> Path:
    """Create a production deployment package."""
    
    # Create deployment configuration
    config = DeploymentConfig(environment=environment, **config_overrides)
    
    # Initialize production manager
    manager = ProductionManager(project_root)
    
    # Perform production readiness checks
    checks = manager.perform_production_checks(config)
    if not checks["ready_for_production"]:
        failed_checks = [name for name, result in checks["checks"].items() if not result]
        raise QuantumMLOpsException(
            f"Production readiness checks failed: {', '.join(failed_checks)}",
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            suggestions=[
                "Install required tools (Docker, kubectl)",
                "Configure cluster access",
                "Set required environment variables"
            ]
        )
        
    # Create deployment package
    package_path = manager.create_deployment_package(config, hostname)
    
    logger.info(f"Production deployment package created: {package_path}")
    return package_path