# Software Bill of Materials (SBOM) Configuration

## Overview

A Software Bill of Materials (SBOM) is a comprehensive inventory of all software components, dependencies, and metadata used in the quantum MLOps workbench. This document provides detailed configuration and best practices for SBOM generation, management, and integration into CI/CD pipelines.

## 🎯 Why SBOMs Matter for Quantum ML

### Traditional Benefits
- **Supply Chain Security**: Track all dependencies and their vulnerabilities
- **Compliance**: Meet regulatory requirements (Executive Order 14028, NTIA)
- **Risk Management**: Understand and mitigate third-party component risks
- **Incident Response**: Quickly identify affected components during security incidents

### Quantum-Specific Benefits
- **Quantum Framework Tracking**: Monitor PennyLane, Qiskit, Cirq dependencies
- **Hardware Driver Dependencies**: Track quantum hardware SDK components
- **Cryptographic Component Tracking**: Essential for post-quantum readiness
- **Research Reproducibility**: Enable exact environment reconstruction

## 📋 SBOM Standards and Formats

### Supported Formats

#### 1. SPDX (Software Package Data Exchange)
- **Standard**: ISO/IEC 5962:2021
- **Format**: JSON, YAML, RDF, Tag-Value
- **Use Case**: Legal compliance, license tracking

```json
{
  "spdxVersion": "SPDX-2.3",
  "creationInfo": {
    "created": "2025-01-15T10:30:00Z",
    "creators": ["Tool: quantum-mlops-sbom-generator"]
  },
  "name": "quantum-mlops-workbench",
  "packages": [
    {
      "SPDXID": "SPDXRef-Package-pennylane",
      "name": "pennylane",
      "downloadLocation": "https://pypi.org/project/PennyLane/0.34.0/",
      "filesAnalyzed": false,
      "licenseConcluded": "Apache-2.0",
      "copyrightText": "Copyright (c) 2018-2024 Xanadu Quantum Technologies Inc."
    }
  ]
}
```

#### 2. CycloneDX
- **Standard**: OWASP CycloneDX v1.5
- **Format**: JSON, XML
- **Use Case**: Vulnerability management, security analysis

```json
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.5",
  "serialNumber": "urn:uuid:quantum-mlops-2025-01-15",
  "version": 1,
  "metadata": {
    "timestamp": "2025-01-15T10:30:00Z",
    "tools": [
      {
        "vendor": "quantum-mlops",
        "name": "sbom-generator",
        "version": "1.0.0"
      }
    ]
  },
  "components": [
    {
      "type": "library",
      "bom-ref": "pennylane@0.34.0",
      "name": "pennylane",
      "version": "0.34.0",
      "licenses": [
        {
          "license": {
            "name": "Apache-2.0"
          }
        }
      ],
      "hashes": [
        {
          "alg": "SHA-256",
          "content": "abc123..."
        }
      ]
    }
  ]
}
```

## 🛠️ SBOM Generation Tools

### 1. CycloneDX Python
```bash
# Install CycloneDX Python tool
pip install cyclonedx-bom

# Generate SBOM
cyclonedx-py -o sbom-cyclonedx.json \
  --format json \
  --schema-version 1.5 \
  --include-dev \
  --include-optional \
  .
```

### 2. Syft (Anchore)
```bash
# Install Syft
curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin

# Generate SPDX SBOM
syft . -o spdx-json=sbom-syft.spdx.json

# Generate CycloneDX SBOM
syft . -o cyclonedx-json=sbom-syft-cyclonedx.json
```

### 3. Custom Quantum SBOM Generator
```python
#!/usr/bin/env python3
"""
Custom SBOM generator with quantum-specific enhancements
"""
import json
import pkg_resources
from datetime import datetime
import hashlib
import subprocess
from typing import Dict, List, Any

class QuantumSBOMGenerator:
    def __init__(self):
        self.quantum_frameworks = [
            'pennylane', 'qiskit', 'cirq', 'amazon-braket-sdk',
            'qiskit-ibm-runtime', 'qiskit-machine-learning'
        ]
    
    def generate_sbom(self, format_type: str = 'spdx-json') -> Dict[str, Any]:
        """Generate SBOM with quantum-specific metadata"""
        if format_type == 'spdx-json':
            return self._generate_spdx()
        elif format_type == 'cyclonedx-json':
            return self._generate_cyclonedx()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _generate_spdx(self) -> Dict[str, Any]:
        """Generate SPDX format SBOM"""
        packages = []
        
        for dist in pkg_resources.working_set:
            package_info = {
                "SPDXID": f"SPDXRef-Package-{dist.project_name}",
                "name": dist.project_name,
                "version": dist.version,
                "downloadLocation": f"https://pypi.org/project/{dist.project_name}/{dist.version}/",
                "filesAnalyzed": False,
                "copyrightText": "NOASSERTION"
            }
            
            # Add quantum-specific metadata
            if dist.project_name.lower() in self.quantum_frameworks:
                package_info["annotations"] = [{
                    "annotationType": "OTHER",
                    "annotator": "quantum-mlops-sbom-generator",
                    "annotationDate": datetime.utcnow().isoformat() + "Z",
                    "annotationComment": f"Quantum computing framework: {dist.project_name}"
                }]
            
            # Get license information
            try:
                metadata = dist.get_metadata('METADATA')
                if 'License:' in metadata:
                    license_line = next(line for line in metadata.split('\n') if line.startswith('License:'))
                    package_info["licenseConcluded"] = license_line.split(':', 1)[1].strip()
            except:
                package_info["licenseConcluded"] = "NOASSERTION"
            
            packages.append(package_info)
        
        return {
            "spdxVersion": "SPDX-2.3",
            "creationInfo": {
                "created": datetime.utcnow().isoformat() + "Z",
                "creators": ["Tool: quantum-mlops-sbom-generator"],
                "licenseListVersion": "3.21"
            },
            "name": "quantum-mlops-workbench",
            "SPDXID": "SPDXRef-DOCUMENT",
            "documentNamespace": f"https://quantum-mlops.example.com/sbom/{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "packages": packages
        }
    
    def _generate_cyclonedx(self) -> Dict[str, Any]:
        """Generate CycloneDX format SBOM"""
        components = []
        
        for dist in pkg_resources.working_set:
            component = {
                "type": "library",
                "bom-ref": f"{dist.project_name}@{dist.version}",
                "name": dist.project_name,
                "version": dist.version,
                "scope": "required"
            }
            
            # Add quantum-specific properties
            if dist.project_name.lower() in self.quantum_frameworks:
                component["properties"] = [{
                    "name": "quantum:framework",
                    "value": "true"
                }, {
                    "name": "quantum:category",
                    "value": self._get_quantum_category(dist.project_name)
                }]
            
            components.append(component)
        
        return {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "serialNumber": f"urn:uuid:quantum-mlops-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            "version": 1,
            "metadata": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "tools": [{
                    "vendor": "quantum-mlops",
                    "name": "quantum-sbom-generator",
                    "version": "1.0.0"
                }],
                "component": {
                    "type": "application",
                    "name": "quantum-mlops-workbench",
                    "version": "0.1.0"
                }
            },
            "components": components
        }
    
    def _get_quantum_category(self, package_name: str) -> str:
        """Categorize quantum packages"""
        categories = {
            'pennylane': 'quantum-ml-framework',
            'qiskit': 'quantum-computing-sdk',
            'cirq': 'quantum-computing-sdk',
            'amazon-braket-sdk': 'cloud-quantum-service',
            'qiskit-ibm-runtime': 'cloud-quantum-service',
            'qiskit-machine-learning': 'quantum-ml-library'
        }
        return categories.get(package_name.lower(), 'quantum-related')

# Usage example
if __name__ == "__main__":
    generator = QuantumSBOMGenerator()
    
    # Generate SPDX SBOM
    spdx_sbom = generator.generate_sbom('spdx-json')
    with open('quantum-sbom-spdx.json', 'w') as f:
        json.dump(spdx_sbom, f, indent=2)
    
    # Generate CycloneDX SBOM
    cyclonedx_sbom = generator.generate_sbom('cyclonedx-json')
    with open('quantum-sbom-cyclonedx.json', 'w') as f:
        json.dump(cyclonedx_sbom, f, indent=2)
    
    print("Quantum SBOM generation completed!")
```

## 🔧 CI/CD Integration

### GitHub Actions Workflow
```yaml
# .github/workflows/sbom-generation.yml
name: SBOM Generation

on:
  push:
    branches: [main]
    paths: ['requirements*.txt', 'pyproject.toml']
  release:
    types: [published]
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday

jobs:
  generate-sbom:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    
    - name: Install Dependencies
      run: |
        pip install -e ".[dev,all]"
        pip install cyclonedx-bom
    
    - name: Generate SBOM with CycloneDX
      run: |
        cyclonedx-py -o sbom-cyclonedx.json \
          --format json \
          --schema-version 1.5 \
          --include-dev \
          --include-optional
    
    - name: Generate SBOM with Syft
      uses: anchore/sbom-action@v0
      with:
        path: .
        format: spdx-json
        artifact-name: sbom-syft.spdx.json
    
    - name: Generate Quantum-Enhanced SBOM
      run: |
        python scripts/security/generate_quantum_sbom.py \
          --output quantum-sbom-enhanced.json \
          --format cyclonedx \
          --include-quantum-metadata
    
    - name: Validate SBOM
      run: |
        python scripts/security/validate_sbom.py \
          --sbom-file sbom-cyclonedx.json \
          --check-completeness \
          --check-quantum-components
    
    - name: Sign SBOM with Cosign
      uses: sigstore/cosign-installer@v3
    
    - name: Sign SBOM
      run: |
        cosign sign-blob \
          --bundle sbom-cyclonedx.json.bundle \
          sbom-cyclonedx.json
    
    - name: Upload SBOM Artifacts
      uses: actions/upload-artifact@v4
      with:
        name: sbom-artifacts
        path: |
          sbom-*.json
          *.bundle
        retention-days: 90
    
    - name: Publish SBOM to Registry
      if: github.event_name == 'release'
      run: |
        # Upload to SBOM registry or artifact store
        curl -X POST \
          -H "Authorization: Bearer ${{ secrets.SBOM_REGISTRY_TOKEN }}" \
          -H "Content-Type: application/json" \
          --data @sbom-cyclonedx.json \
          "${{ secrets.SBOM_REGISTRY_URL }}/quantum-mlops-workbench/${{ github.sha }}"
```

## 🔍 SBOM Validation and Quality Checks

### Validation Script
```python
#!/usr/bin/env python3
"""
SBOM validation and quality assessment script
"""
import json
import sys
from typing import Dict, List, Any, Tuple

class SBOMValidator:
    def __init__(self, sbom_file: str):
        with open(sbom_file, 'r') as f:
            self.sbom = json.load(f)
        self.issues = []
        self.warnings = []
    
    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """Validate SBOM completeness and quality"""
        self._check_required_fields()
        self._check_license_information()
        self._check_version_information()
        self._check_quantum_components()
        self._check_vulnerability_correlation()
        
        return len(self.issues) == 0, self.issues, self.warnings
    
    def _check_required_fields(self):
        """Check for required SBOM fields"""
        if 'bomFormat' in self.sbom:  # CycloneDX
            required_fields = ['bomFormat', 'specVersion', 'version', 'components']
        else:  # SPDX
            required_fields = ['spdxVersion', 'creationInfo', 'name', 'packages']
        
        for field in required_fields:
            if field not in self.sbom:
                self.issues.append(f"Missing required field: {field}")
    
    def _check_license_information(self):
        """Check license information completeness"""
        components = self.sbom.get('components', self.sbom.get('packages', []))
        unlicensed_count = 0
        
        for component in components:
            has_license = False
            
            # Check CycloneDX license format
            if 'licenses' in component and component['licenses']:
                has_license = True
            
            # Check SPDX license format
            if 'licenseConcluded' in component and component['licenseConcluded'] != 'NOASSERTION':
                has_license = True
            
            if not has_license:
                unlicensed_count += 1
        
        if unlicensed_count > 0:
            self.warnings.append(f"{unlicensed_count} components missing license information")
    
    def _check_quantum_components(self):
        """Check quantum-specific component tracking"""
        quantum_frameworks = ['pennylane', 'qiskit', 'cirq', 'braket']
        components = self.sbom.get('components', self.sbom.get('packages', []))
        
        quantum_components = []
        for component in components:
            name = component.get('name', '').lower()
            if any(qf in name for qf in quantum_frameworks):
                quantum_components.append(name)
        
        if not quantum_components:
            self.warnings.append("No quantum computing frameworks detected in SBOM")
        else:
            print(f"Found quantum components: {', '.join(quantum_components)}")
    
    def _check_version_information(self):
        """Check version information completeness"""
        components = self.sbom.get('components', self.sbom.get('packages', []))
        missing_versions = 0
        
        for component in components:
            if not component.get('version'):
                missing_versions += 1
        
        if missing_versions > 0:
            self.issues.append(f"{missing_versions} components missing version information")
    
    def _check_vulnerability_correlation(self):
        """Check if components can be correlated with vulnerability databases"""
        components = self.sbom.get('components', self.sbom.get('packages', []))
        uncorrelatable = 0
        
        for component in components:
            has_purl = 'purl' in component
            has_cpe = any('cpe' in prop.get('name', '') for prop in component.get('properties', []))
            
            if not (has_purl or has_cpe):
                uncorrelatable += 1
        
        if uncorrelatable > 0:
            self.warnings.append(f"{uncorrelatable} components may be difficult to correlate with vulnerability databases")

# Usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_sbom.py <sbom-file>")
        sys.exit(1)
    
    validator = SBOMValidator(sys.argv[1])
    is_valid, issues, warnings = validator.validate()
    
    if issues:
        print("SBOM Validation Issues:")
        for issue in issues:
            print(f"  ❌ {issue}")
    
    if warnings:
        print("SBOM Validation Warnings:")
        for warning in warnings:
            print(f"  ⚠️ {warning}")
    
    if is_valid:
        print("✅ SBOM validation passed!")
        sys.exit(0)
    else:
        print("❌ SBOM validation failed!")
        sys.exit(1)
```

## 📊 SBOM Analysis and Reporting

### Vulnerability Analysis
```python
#!/usr/bin/env python3
"""
SBOM vulnerability analysis using OSV database
"""
import json
import requests
from typing import Dict, List, Any

class SBOMVulnerabilityAnalyzer:
    def __init__(self, sbom_file: str):
        with open(sbom_file, 'r') as f:
            self.sbom = json.load(f)
        self.osv_api_url = "https://api.osv.dev/v1/query"
    
    def analyze_vulnerabilities(self) -> Dict[str, Any]:
        """Analyze SBOM components for known vulnerabilities"""
        components = self.sbom.get('components', self.sbom.get('packages', []))
        vulnerabilities = {}
        
        for component in components:
            name = component.get('name')
            version = component.get('version')
            
            if name and version:
                vulns = self._query_osv(name, version)
                if vulns:
                    vulnerabilities[f"{name}@{version}"] = vulns
        
        return {
            'total_components': len(components),
            'vulnerable_components': len(vulnerabilities),
            'vulnerabilities': vulnerabilities,
            'summary': self._generate_summary(vulnerabilities)
        }
    
    def _query_osv(self, package_name: str, version: str) -> List[Dict]:
        """Query OSV database for vulnerabilities"""
        query = {
            "package": {
                "name": package_name,
                "ecosystem": "PyPI"
            },
            "version": version
        }
        
        try:
            response = requests.post(self.osv_api_url, json=query, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('vulns', [])
        except requests.RequestException:
            pass
        
        return []
    
    def _generate_summary(self, vulnerabilities: Dict) -> Dict[str, int]:
        """Generate vulnerability summary"""
        severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNKNOWN': 0}
        
        for component_vulns in vulnerabilities.values():
            for vuln in component_vulns:
                severity = vuln.get('severity', [{}])
                if severity:
                    severity_level = severity[0].get('score', 'UNKNOWN')
                    if severity_level in severity_counts:
                        severity_counts[severity_level] += 1
                    else:
                        severity_counts['UNKNOWN'] += 1
        
        return severity_counts

# Usage
if __name__ == "__main__":
    analyzer = SBOMVulnerabilityAnalyzer('sbom-cyclonedx.json')
    results = analyzer.analyze_vulnerabilities()
    
    print(json.dumps(results, indent=2))
```

## 🔒 SBOM Security and Signing

### Digital Signing with Cosign
```bash
# Install cosign
curl -O -L "https://github.com/sigstore/cosign/releases/latest/download/cosign-linux-amd64"
sudo mv cosign-linux-amd64 /usr/local/bin/cosign
sudo chmod +x /usr/local/bin/cosign

# Generate key pair (for private signing)
cosign generate-key-pair

# Sign SBOM
cosign sign-blob --key cosign.key sbom-cyclonedx.json

# Verify signature
cosign verify-blob --key cosign.pub --signature sbom-cyclonedx.json.sig sbom-cyclonedx.json
```

### Keyless Signing (Recommended)
```bash
# Sign with keyless signing (uses OIDC)
cosign sign-blob --bundle sbom-cyclonedx.json.bundle sbom-cyclonedx.json

# Verify keyless signature
cosign verify-blob --bundle sbom-cyclonedx.json.bundle sbom-cyclonedx.json
```

## 📈 SBOM Monitoring and Alerting

### Monitoring Configuration
```yaml
# monitoring/sbom-alerts.yml
sbom_monitoring:
  schedules:
    generation: "0 6 * * *"  # Daily at 6 AM
    vulnerability_scan: "0 */6 * * *"  # Every 6 hours
    
  alerts:
    critical_vulnerabilities:
      threshold: 1
      channel: "#security-alerts"
    
    sbom_generation_failure:
      retry_count: 3
      escalation: security-team
    
    license_compliance:
      forbidden_licenses: ["GPL-3.0", "AGPL-3.0"]
      notification: legal-team
    
    quantum_component_updates:
      frameworks: ["pennylane", "qiskit", "cirq"]
      notification: quantum-team
```

## 🔄 SBOM Lifecycle Management

### Retention Policy
```yaml
# SBOM retention configuration
sbom_retention:
  development_builds: 30 days
  release_builds: 2 years
  security_incidents: 7 years
  
  storage_locations:
    - artifact_registry
    - backup_storage
    - compliance_archive
```

### Update Automation
```python
#!/usr/bin/env python3
"""
Automated SBOM update detection and notification
"""
import json
import hashlib
from typing import Dict, Set

class SBOMUpdateDetector:
    def __init__(self, current_sbom: str, previous_sbom: str):
        self.current = self._load_sbom(current_sbom)
        self.previous = self._load_sbom(previous_sbom)
    
    def detect_changes(self) -> Dict[str, Set[str]]:
        """Detect changes between SBOM versions"""
        current_components = self._extract_components(self.current)
        previous_components = self._extract_components(self.previous)
        
        return {
            'added': current_components - previous_components,
            'removed': previous_components - current_components,
            'modified': self._detect_modified_components()
        }
    
    def _load_sbom(self, file_path: str) -> Dict:
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def _extract_components(self, sbom: Dict) -> Set[str]:
        components = sbom.get('components', sbom.get('packages', []))
        return {f"{comp.get('name')}@{comp.get('version')}" for comp in components}
    
    def _detect_modified_components(self) -> Set[str]:
        # Implementation for detecting component modifications
        # (version changes, metadata changes, etc.)
        pass

# Integration with notification systems
def notify_sbom_changes(changes: Dict[str, Set[str]]):
    """Send notifications for SBOM changes"""
    if changes['added']:
        print(f"New components added: {', '.join(changes['added'])}")
    
    if changes['removed']:
        print(f"Components removed: {', '.join(changes['removed'])}")
    
    if changes['modified']:
        print(f"Components modified: {', '.join(changes['modified'])}")
```

## 🎯 Best Practices

### 1. SBOM Generation
- **Automate Generation**: Include SBOM generation in CI/CD pipelines
- **Multiple Formats**: Generate both SPDX and CycloneDX formats
- **Quantum Metadata**: Include quantum-specific component categorization
- **Regular Updates**: Generate SBOMs for every build and release

### 2. SBOM Quality
- **Completeness**: Ensure all components are captured
- **Accuracy**: Validate version and license information
- **Metadata**: Include comprehensive component metadata
- **Signing**: Digitally sign SBOMs for integrity verification

### 3. SBOM Usage
- **Vulnerability Management**: Regularly scan SBOMs for vulnerabilities
- **License Compliance**: Monitor license compatibility
- **Supply Chain Risk**: Assess third-party component risks
- **Incident Response**: Use SBOMs for rapid impact assessment

### 4. Integration
- **Security Tools**: Integrate with vulnerability scanners
- **Compliance Systems**: Feed into compliance reporting
- **Monitoring**: Set up alerts for SBOM changes
- **Documentation**: Maintain SBOM documentation and procedures

## 📚 Additional Resources

- [NTIA SBOM Minimum Elements](https://www.ntia.doc.gov/files/ntia/publications/sbom_minimum_elements_report.pdf)
- [SPDX Specification](https://spdx.github.io/spdx-spec/)
- [CycloneDX Specification](https://cyclonedx.org/specification/overview/)
- [CISA SBOM Tools and Resources](https://www.cisa.gov/sbom)
- [Executive Order 14028](https://www.whitehouse.gov/briefing-room/presidential-actions/2021/05/12/executive-order-on-improving-the-nations-cybersecurity/)