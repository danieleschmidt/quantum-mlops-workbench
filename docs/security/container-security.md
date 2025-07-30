# Container Security Scanning and Hardening

## Overview

Container security is critical for quantum MLOps workbenches due to the sensitive nature of quantum algorithms, valuable intellectual property, and access to expensive quantum hardware resources. This document provides comprehensive guidance on container security scanning, hardening, and best practices specific to quantum computing environments.

## 🎯 Container Security Objectives

### Core Security Goals
- **Image Vulnerability Management**: Identify and remediate container vulnerabilities
- **Runtime Security**: Secure container execution environment
- **Supply Chain Security**: Ensure trusted base images and dependencies
- **Compliance**: Meet security standards and regulatory requirements

### Quantum-Specific Considerations
- **Algorithm Protection**: Secure proprietary quantum algorithms and circuits
- **Credential Security**: Protect quantum cloud provider credentials
- **Resource Isolation**: Prevent unauthorized access to quantum hardware
- **Data Confidentiality**: Secure quantum training data and results

## 🛠️ Container Security Scanning Tools

### 1. Trivy (Aqua Security)
Comprehensive vulnerability scanner supporting multiple formats and registries.

#### Installation
```bash
# Install Trivy
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

# Or using package manager
sudo apt-get install wget apt-transport-https gnupg lsb-release
wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
sudo apt-get update
sudo apt-get install trivy
```

#### Basic Usage
```bash
# Scan container image
trivy image quantum-mlops:latest

# Scan with specific severity levels
trivy image --severity HIGH,CRITICAL quantum-mlops:latest

# Generate JSON report
trivy image --format json --output trivy-report.json quantum-mlops:latest

# Scan filesystem
trivy fs --security-checks vuln,config .

# Scan Kubernetes manifests
trivy k8s --report summary cluster
```

#### Configuration
```yaml
# .trivyignore - Ignore specific vulnerabilities
CVE-2023-12345  # False positive for quantum library
CVE-2023-67890  # Accepted risk with mitigation

# trivy.yaml - Trivy configuration
format: json
output: trivy-results.json
severity:
  - HIGH
  - CRITICAL
ignore-unfixed: true
skip-dirs:
  - /tmp
  - /var/cache
```

### 2. Grype (Anchore)
Fast vulnerability scanner with comprehensive database coverage.

#### Installation and Usage
```bash
# Install Grype
curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin

# Scan container image
grype quantum-mlops:latest

# Scan with JSON output
grype quantum-mlops:latest -o json --file grype-report.json

# Scan specific directory
grype dir:.

# Compare with previous scan
grype quantum-mlops:latest --fail-on high
```

### 3. Clair (Red Hat)
Scalable container vulnerability analysis service.

#### Docker Compose Setup
```yaml
# docker-compose.clair.yml
version: '3.8'
services:
  clair:
    image: quay.io/coreos/clair:v4.7.4
    ports:
      - "6060:6060"
      - "6061:6061"
    environment:
      CLAIR_CONF: /etc/clair/config.yaml
    volumes:
      - ./clair-config.yaml:/etc/clair/config.yaml
    
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: clair
      POSTGRES_USER: clair
      POSTGRES_PASSWORD: clair
    volumes:
      - clair-db:/var/lib/postgresql/data

volumes:
  clair-db:
```

### 4. Snyk Container
Commercial solution with excellent developer integration.

```bash
# Install Snyk CLI
npm install -g snyk

# Authenticate
snyk auth

# Scan container
snyk container test quantum-mlops:latest

# Monitor for new vulnerabilities
snyk container monitor quantum-mlops:latest --project-name=quantum-mlops

# Scan Dockerfile
snyk iac test Dockerfile
```

## 🔧 CI/CD Integration

### GitHub Actions Workflow
```yaml
# .github/workflows/container-security.yml
name: Container Security Scanning

on:
  push:
    branches: [main, develop]
    paths: ['Dockerfile*', 'docker-compose*.yml']
  pull_request:
    paths: ['Dockerfile*', 'docker-compose*.yml']
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  container-security-scan:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      security-events: write
      packages: read
    
    steps:
    - name: Checkout Repository
      uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Build Container Image
      uses: docker/build-push-action@v5
      with:
        context: .
        load: true
        tags: quantum-mlops:ci-scan
        cache-from: type=gha
        cache-to: type=gha,mode=max
    
    - name: Run Trivy Vulnerability Scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: quantum-mlops:ci-scan
        format: 'sarif'
        output: 'trivy-results.sarif'
        severity: 'MEDIUM,HIGH,CRITICAL'
    
    - name: Upload Trivy SARIF to GitHub Security
      uses: github/codeql-action/upload-sarif@v3
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'
    
    - name: Run Trivy JSON Report
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: quantum-mlops:ci-scan
        format: 'json'
        output: 'trivy-report.json'
        severity: 'MEDIUM,HIGH,CRITICAL'
    
    - name: Run Grype Scanner
      run: |
        curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin
        grype quantum-mlops:ci-scan -o json --file grype-report.json
    
    - name: Analyze Container Security Results
      run: |
        python scripts/security/analyze_container_results.py \
          --trivy-report trivy-report.json \
          --grype-report grype-report.json \
          --threshold-critical 0 \
          --threshold-high 5 \
          --output security-analysis.json
    
    - name: Container Configuration Security Scan
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'config'
        scan-ref: '.'
        format: 'json'
        output: 'trivy-config-report.json'
    
    - name: Dockerfile Security Scan
      run: |
        docker run --rm -v "$(pwd)":/project \
          hadolint/hadolint:latest hadolint /project/Dockerfile \
          --format json > hadolint-report.json || true
    
    - name: Generate Security Report
      run: |
        python scripts/security/generate_container_security_report.py \
          --trivy-vuln trivy-report.json \
          --trivy-config trivy-config-report.json \
          --grype grype-report.json \
          --hadolint hadolint-report.json \
          --output container-security-report.html
    
    - name: Upload Security Reports
      uses: actions/upload-artifact@v4
      with:
        name: container-security-reports
        path: |
          trivy-*.json
          grype-report.json
          hadolint-report.json
          security-analysis.json
          container-security-report.html
        retention-days: 30
    
    - name: Fail Build on Critical Vulnerabilities
      run: |
        python scripts/security/check_security_threshold.py \
          --analysis-file security-analysis.json \
          --fail-on-critical true \
          --max-high-vulnerabilities 5
```

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: dockerfile-lint
        name: Dockerfile Lint
        entry: hadolint
        language: docker_image
        files: Dockerfile.*
        args: ['--config', '.hadolint.yaml']
      
      - id: container-structure-test
        name: Container Structure Test
        entry: container-structure-test
        language: docker_image
        files: Dockerfile.*
        args: ['test', '--image', 'quantum-mlops:test', '--config', 'container-structure-test.yaml']
```

## 🏗️ Secure Container Image Building

### Multi-stage Dockerfile Security
```dockerfile
# Dockerfile - Security-hardened quantum MLOps container
# Build stage
FROM python:3.11-slim as builder

# Security: Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Security: Create non-root user for build
RUN groupadd -g 1000 builder && \
    useradd -u 1000 -g builder -m builder

USER builder
WORKDIR /build

# Security: Copy only necessary files
COPY --chown=builder:builder requirements*.txt ./
COPY --chown=builder:builder pyproject.toml ./
COPY --chown=builder:builder src/ ./src/

# Security: Install dependencies with pinned versions
RUN pip install --user --no-cache-dir --require-hashes -r requirements.txt

# Production stage
FROM python:3.11-slim

# Security: Install security updates only
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Security: Create non-root user
RUN groupadd -g 1000 quantum && \
    useradd -u 1000 -g quantum -m -s /bin/bash quantum

# Security: Set up secure directories
RUN mkdir -p /app /app/data /app/logs && \
    chown -R quantum:quantum /app

# Security: Copy built artifacts from builder stage
COPY --from=builder --chown=quantum:quantum /home/builder/.local /home/quantum/.local
COPY --chown=quantum:quantum src/ /app/src/

# Security: Set PATH for user-installed packages
ENV PATH="/home/quantum/.local/bin:$PATH"
ENV PYTHONPATH="/app/src"

# Security: Switch to non-root user
USER quantum
WORKDIR /app

# Security: Set secure defaults
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV QUANTUM_SECURE_MODE=true

# Security: Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import quantum_mlops; print('healthy')" || exit 1

# Security: Use exec form for CMD
CMD ["python", "-m", "quantum_mlops.cli"]

# Security: Add labels for tracking
LABEL org.opencontainers.image.title="Quantum MLOps Workbench"
LABEL org.opencontainers.image.description="Secure quantum machine learning environment"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.authors="security@quantum-mlops.example.com"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.documentation="https://quantum-mlops.readthedocs.io"
LABEL org.opencontainers.image.source="https://github.com/example/quantum-mlops-workbench"
```

### Distroless Container (Ultra-secure)
```dockerfile
# Dockerfile.distroless - Ultra-secure distroless container
FROM python:3.11-slim as builder

# Build stage (same as above)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements*.txt pyproject.toml ./
COPY src/ ./src/

RUN pip install --target=/install --no-cache-dir -r requirements.txt
RUN pip install --target=/install --no-cache-dir .

# Production stage with distroless
FROM gcr.io/distroless/python3-debian12

# Copy application and dependencies
COPY --from=builder /install /usr/local/lib/python3.11/site-packages
COPY --from=builder /build/src /app

# Set environment
ENV PYTHONPATH="/usr/local/lib/python3.11/site-packages:/app"
ENV QUANTUM_SECURE_MODE=true

# Use numeric UID for security
USER 1000

WORKDIR /app

# Entry point
ENTRYPOINT ["python", "-m", "quantum_mlops.cli"]
```

## 🔍 Container Security Analysis Scripts

### Security Analysis Script
```python
#!/usr/bin/env python3
"""
Container security analysis and reporting
"""
import json
import sys
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class SecurityThreshold:
    critical: int = 0
    high: int = 5
    medium: int = 20
    low: int = 50

class ContainerSecurityAnalyzer:
    def __init__(self, trivy_report: str, grype_report: str = None):
        self.trivy_data = self._load_json(trivy_report)
        self.grype_data = self._load_json(grype_report) if grype_report else None
        self.results = {
            'summary': {},
            'vulnerabilities': [],
            'recommendations': [],
            'compliance': {}
        }
    
    def analyze(self, threshold: SecurityThreshold = None) -> Dict[str, Any]:
        """Perform comprehensive security analysis"""
        threshold = threshold or SecurityThreshold()
        
        # Analyze vulnerabilities
        self._analyze_vulnerabilities()
        
        # Check security thresholds
        self._check_thresholds(threshold)
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Assess compliance
        self._assess_compliance()
        
        return self.results
    
    def _load_json(self, file_path: str) -> Dict:
        """Load JSON report file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _analyze_vulnerabilities(self):
        """Analyze vulnerability data from Trivy report"""
        severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        vulnerabilities = []
        
        # Process Trivy results
        for result in self.trivy_data.get('Results', []):
            for vuln in result.get('Vulnerabilities', []):
                severity = vuln.get('Severity', 'UNKNOWN')
                if severity in severity_counts:
                    severity_counts[severity] += 1
                
                vulnerabilities.append({
                    'id': vuln.get('VulnerabilityID'),
                    'severity': severity,
                    'package': vuln.get('PkgName'),
                    'version': vuln.get('InstalledVersion'),
                    'fixed_version': vuln.get('FixedVersion'),
                    'title': vuln.get('Title'),
                    'description': vuln.get('Description', '')[:200] + '...'
                })
        
        self.results['summary'] = {
            'total_vulnerabilities': sum(severity_counts.values()),
            'severity_breakdown': severity_counts,
            'scan_time': self.trivy_data.get('CreatedAt'),
            'image_size': self._get_image_size()
        }
        
        # Sort vulnerabilities by severity
        severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        self.results['vulnerabilities'] = sorted(
            vulnerabilities,
            key=lambda x: severity_order.get(x['severity'], 4)
        )
    
    def _check_thresholds(self, threshold: SecurityThreshold):
        """Check if vulnerabilities exceed security thresholds"""
        counts = self.results['summary']['severity_breakdown']
        
        violations = []
        if counts['CRITICAL'] > threshold.critical:
            violations.append(f"Critical vulnerabilities: {counts['CRITICAL']} > {threshold.critical}")
        
        if counts['HIGH'] > threshold.high:
            violations.append(f"High vulnerabilities: {counts['HIGH']} > {threshold.high}")
        
        if counts['MEDIUM'] > threshold.medium:
            violations.append(f"Medium vulnerabilities: {counts['MEDIUM']} > {threshold.medium}")
        
        self.results['threshold_violations'] = violations
        self.results['passes_threshold'] = len(violations) == 0
    
    def _generate_recommendations(self):
        """Generate security recommendations"""
        recommendations = []
        
        # Base image recommendations
        if any('debian' in str(result) for result in self.trivy_data.get('Results', [])):
            recommendations.append({
                'category': 'Base Image',
                'priority': 'HIGH',
                'recommendation': 'Consider using a distroless or minimal base image',
                'rationale': 'Reduces attack surface and number of vulnerabilities'
            })
        
        # Package update recommendations
        fixable_vulns = [v for v in self.results['vulnerabilities'] if v.get('fixed_version')]
        if fixable_vulns:
            recommendations.append({
                'category': 'Package Updates',
                'priority': 'HIGH',
                'recommendation': f'Update {len(fixable_vulns)} packages with available fixes',
                'rationale': 'Eliminates known vulnerabilities with available patches'
            })
        
        # Quantum-specific security recommendations
        quantum_packages = [v for v in self.results['vulnerabilities'] 
                          if any(pkg in v.get('package', '').lower() 
                               for pkg in ['pennylane', 'qiskit', 'cirq'])]
        
        if quantum_packages:
            recommendations.append({
                'category': 'Quantum Security',
                'priority': 'MEDIUM',
                'recommendation': 'Review quantum framework vulnerabilities carefully',
                'rationale': 'Quantum packages may expose sensitive algorithms or credentials'
            })
        
        self.results['recommendations'] = recommendations
    
    def _assess_compliance(self):
        """Assess compliance with security standards"""
        compliance = {
            'cis_docker_benchmark': self._check_cis_compliance(),
            'nist_guidelines': self._check_nist_compliance(),
            'quantum_security': self._check_quantum_security()
        }
        
        self.results['compliance'] = compliance
    
    def _check_cis_compliance(self) -> Dict[str, Any]:
        """Check CIS Docker Benchmark compliance"""
        # This would implement actual CIS benchmark checks
        return {
            'score': 85,  # Example score
            'passed_checks': 17,
            'total_checks': 20,
            'failed_checks': ['4.1', '4.5', '5.7']  # Example failed checks
        }
    
    def _check_nist_compliance(self) -> Dict[str, Any]:
        """Check NIST container security compliance"""
        return {
            'framework_version': 'NIST SP 800-190',
            'compliance_level': 'Moderate',
            'areas_for_improvement': [
                'Runtime security monitoring',
                'Container image signing'
            ]
        }
    
    def _check_quantum_security(self) -> Dict[str, Any]:
        """Check quantum-specific security compliance"""
        return {
            'post_quantum_ready': True,
            'quantum_credential_security': 'Compliant',
            'algorithm_protection': 'Review needed',
            'hardware_access_controls': 'Implemented'
        }
    
    def _get_image_size(self) -> str:
        """Extract image size from metadata"""
        metadata = self.trivy_data.get('Metadata', {})
        return metadata.get('ImageConfig', {}).get('Size', 'Unknown')
    
    def generate_report(self, format_type: str = 'json') -> str:
        """Generate security report in specified format"""
        if format_type == 'json':
            return json.dumps(self.results, indent=2)
        elif format_type == 'html':
            return self._generate_html_report()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _generate_html_report(self) -> str:
        """Generate HTML security report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Container Security Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .critical { color: #d32f2f; }
                .high { color: #f57c00; }
                .medium { color: #fbc02d; }
                .low { color: #388e3c; }
                .summary { background: #f5f5f5; padding: 15px; border-radius: 5px; }
                .vulnerability { border-left: 4px solid #ccc; padding: 10px; margin: 10px 0; }
                .recommendation { background: #e3f2fd; padding: 10px; margin: 10px 0; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>Container Security Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total Vulnerabilities:</strong> {total_vulnerabilities}</p>
                <p><strong>Critical:</strong> <span class="critical">{critical}</span></p>
                <p><strong>High:</strong> <span class="high">{high}</span></p>
                <p><strong>Medium:</strong> <span class="medium">{medium}</span></p>
                <p><strong>Low:</strong> <span class="low">{low}</span></p>
            </div>
            
            <h2>Recommendations</h2>
            {recommendations_html}
            
            <h2>Top Vulnerabilities</h2>
            {vulnerabilities_html}
        </body>
        </html>
        """
        
        # Generate recommendations HTML
        recommendations_html = ""
        for rec in self.results['recommendations'][:5]:  # Top 5 recommendations
            recommendations_html += f"""
            <div class="recommendation">
                <strong>{rec['category']} ({rec['priority']})</strong><br>
                {rec['recommendation']}<br>
                <em>{rec['rationale']}</em>
            </div>
            """
        
        # Generate vulnerabilities HTML
        vulnerabilities_html = ""
        for vuln in self.results['vulnerabilities'][:10]:  # Top 10 vulnerabilities
            severity_class = vuln['severity'].lower()
            vulnerabilities_html += f"""
            <div class="vulnerability">
                <strong class="{severity_class}">{vuln['severity']}</strong> - {vuln['id']}<br>
                <strong>Package:</strong> {vuln['package']} ({vuln['version']})<br>
                <strong>Title:</strong> {vuln['title']}<br>
                {vuln['description']}
            </div>
            """
        
        return html_template.format(
            total_vulnerabilities=self.results['summary']['total_vulnerabilities'],
            critical=self.results['summary']['severity_breakdown']['CRITICAL'],
            high=self.results['summary']['severity_breakdown']['HIGH'],
            medium=self.results['summary']['severity_breakdown']['MEDIUM'],
            low=self.results['summary']['severity_breakdown']['LOW'],
            recommendations_html=recommendations_html,
            vulnerabilities_html=vulnerabilities_html
        )

# CLI usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze container security reports')
    parser.add_argument('--trivy-report', required=True, help='Trivy JSON report file')
    parser.add_argument('--grype-report', help='Grype JSON report file')
    parser.add_argument('--threshold-critical', type=int, default=0, help='Critical vulnerability threshold')
    parser.add_argument('--threshold-high', type=int, default=5, help='High vulnerability threshold')
    parser.add_argument('--output', default='security-analysis.json', help='Output file')
    parser.add_argument('--format', choices=['json', 'html'], default='json', help='Output format')
    
    args = parser.parse_args()
    
    threshold = SecurityThreshold(
        critical=args.threshold_critical,
        high=args.threshold_high
    )
    
    analyzer = ContainerSecurityAnalyzer(args.trivy_report, args.grype_report)
    results = analyzer.analyze(threshold)
    
    # Write results
    report = analyzer.generate_report(args.format)
    with open(args.output, 'w') as f:
        f.write(report)
    
    # Exit with appropriate code
    if not results['passes_threshold']:
        print("❌ Security threshold violations detected!")
        for violation in results['threshold_violations']:
            print(f"  - {violation}")
        sys.exit(1)
    else:
        print("✅ All security thresholds passed!")
        sys.exit(0)
```

## 🛡️ Runtime Container Security

### Security Policies with OPA/Gatekeeper
```yaml
# kubernetes/security-policies/container-security-policy.yaml
apiVersion: templates.gatekeeper.sh/v1beta1
kind: ConstraintTemplate
metadata:
  name: quantumcontainersecurity
spec:
  crd:
    spec:
      names:
        kind: QuantumContainerSecurity
      validation:
        openAPIV3Schema:
          type: object
          properties:
            allowedRegistries:
              type: array
              items:
                type: string
            requiredSecurityContext:
              type: object
            quantumSecurityChecks:
              type: boolean
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package quantumcontainersecurity
        
        violation[{"msg": msg}] {
          container := input.review.object.spec.containers[_]
          not starts_with(container.image, input.parameters.allowedRegistries[_])
          msg := sprintf("Container image '%v' is not from allowed registry", [container.image])
        }
        
        violation[{"msg": msg}] {
          container := input.review.object.spec.containers[_]
          container.securityContext.runAsRoot == true
          msg := "Containers must not run as root"
        }
        
        violation[{"msg": msg}] {
          input.parameters.quantumSecurityChecks == true
          container := input.review.object.spec.containers[_]
          not container.env
          quantum_frameworks := ["pennylane", "qiskit", "cirq"]
          contains(container.image, quantum_frameworks[_])
          msg := "Quantum containers must have security environment variables configured"
        }

---
apiVersion: templates.gatekeeper.sh/v1beta1
kind: QuantumContainerSecurity
metadata:
  name: quantum-container-security
spec:
  match:
    - apiGroups: ["apps"]
      kinds: ["Deployment"]
  parameters:
    allowedRegistries:
      - "ghcr.io/quantum-mlops/"
      - "quay.io/quantum-mlops/"
    requiredSecurityContext:
      runAsNonRoot: true
      readOnlyRootFilesystem: true
    quantumSecurityChecks: true
```

### Pod Security Standards
```yaml
# kubernetes/security/pod-security-policy.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: quantum-mlops
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantum-mlops-workbench
  namespace: quantum-mlops
spec:
  template:
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault
      
      containers:
      - name: quantum-mlops
        image: ghcr.io/quantum-mlops/workbench:latest
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          runAsUser: 1000
          capabilities:
            drop:
            - ALL
        
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        
        env:
        - name: QUANTUM_SECURE_MODE
          value: "true"
        - name: PYTHONUNBUFFERED
          value: "1"
        
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: quantum-data
          mountPath: /app/data
          readOnly: false
      
      volumes:
      - name: tmp
        emptyDir: {}
      - name: quantum-data
        persistentVolumeClaim:
          claimName: quantum-data
```

## 📊 Container Security Monitoring

### Monitoring Configuration
```yaml
# monitoring/container-security-monitoring.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: container-security-alerts
data:
  alert-rules.yml: |
    groups:
    - name: container_security
      rules:
      - alert: ContainerVulnerabilityHigh
        expr: trivy_vulnerabilities{severity="HIGH"} > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High vulnerability count in container"
          description: "Container {{ $labels.image }} has {{ $value }} high severity vulnerabilities"
      
      - alert: ContainerRunningAsRoot
        expr: container_spec_security_context_run_as_root == 1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Container running as root"
          description: "Container {{ $labels.container }} is running as root user"
      
      - alert: QuantumCredentialExposure
        expr: |
          increase(log_entries{
            level="ERROR",
            message=~".*quantum.*credential.*|.*API.*key.*"
          }[5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Potential quantum credential exposure"
          description: "Detected potential quantum credential exposure in logs"
```

### Security Metrics Collection
```python
#!/usr/bin/env python3
"""
Container security metrics collector
"""
import json
import time
from datetime import datetime
from typing import Dict, List
import requests
from prometheus_client import Gauge, Counter, start_http_server

class ContainerSecurityMetrics:
    def __init__(self):
        # Prometheus metrics
        self.vulnerability_gauge = Gauge(
            'trivy_vulnerabilities_total',
            'Total number of vulnerabilities by severity',
            ['image', 'severity']
        )
        
        self.security_scan_counter = Counter(
            'security_scans_total',
            'Total number of security scans performed',
            ['scanner', 'status']
        )
        
        self.quantum_security_gauge = Gauge(
            'quantum_security_score',
            'Quantum-specific security score',
            ['component']
        )
    
    def collect_trivy_metrics(self, report_file: str):
        """Collect metrics from Trivy report"""
        try:
            with open(report_file, 'r') as f:
                report = json.load(f)
            
            # Extract image name
            image_name = report.get('ArtifactName', 'unknown')
            
            # Count vulnerabilities by severity
            severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            
            for result in report.get('Results', []):
                for vuln in result.get('Vulnerabilities', []):
                    severity = vuln.get('Severity', 'UNKNOWN')
                    if severity in severity_counts:
                        severity_counts[severity] += 1
            
            # Update Prometheus metrics
            for severity, count in severity_counts.items():
                self.vulnerability_gauge.labels(
                    image=image_name,
                    severity=severity
                ).set(count)
            
            self.security_scan_counter.labels(
                scanner='trivy',
                status='success'
            ).inc()
            
        except Exception as e:
            print(f"Error collecting Trivy metrics: {e}")
            self.security_scan_counter.labels(
                scanner='trivy',
                status='error'
            ).inc()
    
    def calculate_quantum_security_score(self, components: List[str]) -> float:
        """Calculate quantum-specific security score"""
        # This would implement actual quantum security scoring logic
        base_score = 100.0
        
        # Deduct points for quantum-related vulnerabilities
        quantum_components = [c for c in components if any(
            qf in c.lower() for qf in ['pennylane', 'qiskit', 'cirq']
        )]
        
        # Example scoring logic
        if len(quantum_components) > 10:
            base_score -= 10  # More components = higher risk
        
        return max(0.0, base_score)
    
    def start_metrics_server(self, port: int = 8000):
        """Start Prometheus metrics server"""
        start_http_server(port)
        print(f"Metrics server started on port {port}")
        
        while True:
            # Collect metrics periodically
            self.collect_trivy_metrics('trivy-report.json')
            time.sleep(300)  # 5 minutes

if __name__ == "__main__":
    metrics = ContainerSecurityMetrics()
    metrics.start_metrics_server()
```

## 🎯 Best Practices Summary

### 1. Container Image Security
- **Use minimal base images**: Prefer distroless or Alpine-based images
- **Multi-stage builds**: Separate build and runtime environments
- **Regular updates**: Keep base images and dependencies current
- **Image signing**: Use cosign or similar tools for image attestation

### 2. Vulnerability Management
- **Automated scanning**: Integrate security scanning in CI/CD
- **Continuous monitoring**: Monitor running containers for new vulnerabilities
- **Patch management**: Maintain update schedules for critical vulnerabilities
- **Threshold enforcement**: Fail builds on critical/high severity issues

### 3. Runtime Security
- **Non-root users**: Always run containers as non-root
- **Read-only filesystems**: Use read-only root filesystems where possible
- **Security contexts**: Implement proper Kubernetes security contexts
- **Network policies**: Restrict container network access

### 4. Quantum-Specific Security
- **Credential management**: Securely handle quantum cloud provider credentials
- **Algorithm protection**: Protect proprietary quantum algorithms
- **Hardware access control**: Implement proper quantum hardware access controls
- **Data encryption**: Encrypt quantum training data and results

### 5. Compliance and Auditing
- **Security policies**: Implement and enforce security policies
- **Audit logging**: Maintain comprehensive audit logs
- **Compliance checking**: Regular compliance assessments
- **Documentation**: Keep security documentation current

## 📚 Additional Resources

- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [NIST Container Security Guide](https://csrc.nist.gov/publications/detail/sp/800-190/final)
- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/)
- [OWASP Container Security](https://owasp.org/www-project-docker-top-10/)
- [Trivy Documentation](https://aquasecurity.github.io/trivy/)
- [Grype Documentation](https://github.com/anchore/grype)
- [Container Structure Tests](https://github.com/GoogleContainerTools/container-structure-test)