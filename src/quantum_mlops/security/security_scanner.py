"""Security scanning and vulnerability detection for quantum MLOps."""

import os
import re
import json
import subprocess
import tempfile
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class SecurityFinding:
    """Security vulnerability or issue finding."""
    
    severity: str  # critical, high, medium, low, info
    category: str  # code_quality, security, quantum_specific, dependency
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    recommendation: Optional[str] = None
    cwe_id: Optional[str] = None  # Common Weakness Enumeration ID
    confidence: str = "medium"  # high, medium, low
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return asdict(self)


@dataclass
class ScanResult:
    """Result of security scan."""
    
    scan_type: str
    timestamp: datetime
    findings: List[SecurityFinding]
    summary: Dict[str, int]
    duration: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "scan_type": self.scan_type,
            "timestamp": self.timestamp.isoformat(),
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "duration": self.duration,
            "metadata": self.metadata
        }


class CodeSecurityScanner:
    """Static code analysis for security issues."""
    
    def __init__(self):
        """Initialize code security scanner."""
        self.patterns = self._load_security_patterns()
        
    def _load_security_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load security patterns for scanning."""
        return {
            "hardcoded_secrets": [
                {
                    "pattern": r"(?i)(password|passwd|pwd|secret|token|key|api[_-]?key)\s*[:=]\s*['\"]([^'\"]{8,})['\"]",
                    "severity": "high",
                    "title": "Hardcoded Secret",
                    "description": "Hardcoded secret or password found in code",
                    "cwe_id": "CWE-798"
                },
                {
                    "pattern": r"(?i)(aws_access_key_id|aws_secret_access_key|ibm_quantum_token)\s*[:=]\s*['\"]([^'\"]+)['\"]",
                    "severity": "critical", 
                    "title": "Hardcoded Cloud Credentials",
                    "description": "Hardcoded cloud provider credentials found",
                    "cwe_id": "CWE-798"
                }
            ],
            "sql_injection": [
                {
                    "pattern": r"(?i)(select|insert|update|delete|drop|union|exec)\s+.*\+.*['\"]",
                    "severity": "high",
                    "title": "Potential SQL Injection",
                    "description": "String concatenation in SQL query may be vulnerable to injection",
                    "cwe_id": "CWE-89"
                }
            ],
            "command_injection": [
                {
                    "pattern": r"(subprocess\.|os\.system|os\.popen|eval\(|exec\().*\+",
                    "severity": "high",
                    "title": "Potential Command Injection",
                    "description": "Dynamic command execution may be vulnerable to injection",
                    "cwe_id": "CWE-77"
                }
            ],
            "path_traversal": [
                {
                    "pattern": r"(open\(|file\(|Path\().*\+.*['\"].*\.\.[/\\]",
                    "severity": "medium",
                    "title": "Potential Path Traversal",
                    "description": "File path construction may be vulnerable to directory traversal",
                    "cwe_id": "CWE-22"
                }
            ],
            "quantum_specific": [
                {
                    "pattern": r"(?i)(quantum[_-]?circuit|quantum[_-]?parameter).*log.*\(",
                    "severity": "medium",
                    "title": "Quantum Information Leakage",
                    "description": "Quantum circuit or parameter data may be logged inappropriately",
                    "cwe_id": "CWE-532"
                },
                {
                    "pattern": r"(?i)(backend[_-]?token|device[_-]?arn).*print\(",
                    "severity": "medium",
                    "title": "Backend Credential Exposure",
                    "description": "Quantum backend credentials may be exposed in output",
                    "cwe_id": "CWE-532"
                }
            ],
            "weak_crypto": [
                {
                    "pattern": r"(?i)(md5|sha1)\s*\(",
                    "severity": "medium",
                    "title": "Weak Cryptographic Hash",
                    "description": "Use of weak cryptographic hash function",
                    "cwe_id": "CWE-327"
                },
                {
                    "pattern": r"(?i)random\.random\(\)",
                    "severity": "low",
                    "title": "Weak Random Number Generation",
                    "description": "Use of weak random number generator for security purposes",
                    "cwe_id": "CWE-338"
                }
            ]
        }
        
    def scan_file(self, file_path: Path) -> List[SecurityFinding]:
        """Scan single file for security issues."""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            lines = content.split('\n')
            
            for category, patterns in self.patterns.items():
                for pattern_info in patterns:
                    pattern = pattern_info["pattern"]
                    
                    for line_num, line in enumerate(lines, 1):
                        matches = re.finditer(pattern, line)
                        
                        for match in matches:
                            # Skip if in comment
                            if line.strip().startswith('#'):
                                continue
                                
                            finding = SecurityFinding(
                                severity=pattern_info["severity"],
                                category=category,
                                title=pattern_info["title"],
                                description=pattern_info["description"],
                                file_path=str(file_path),
                                line_number=line_num,
                                code_snippet=line.strip(),
                                cwe_id=pattern_info.get("cwe_id"),
                                confidence="medium"
                            )
                            findings.append(finding)
                            
        except Exception as e:
            logger.warning(f"Error scanning file {file_path}: {e}")
            
        return findings
        
    def scan_directory(self, directory: Path, extensions: Set[str] = None) -> ScanResult:
        """Scan directory for security issues."""
        start_time = datetime.utcnow()
        
        if extensions is None:
            extensions = {'.py', '.yml', '.yaml', '.json', '.toml', '.cfg', '.ini'}
            
        findings = []
        files_scanned = 0
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix in extensions:
                file_findings = self.scan_file(file_path)
                findings.extend(file_findings)
                files_scanned += 1
                
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Generate summary
        summary = {
            "total_findings": len(findings),
            "files_scanned": files_scanned,
            "critical": len([f for f in findings if f.severity == "critical"]),
            "high": len([f for f in findings if f.severity == "high"]),
            "medium": len([f for f in findings if f.severity == "medium"]),
            "low": len([f for f in findings if f.severity == "low"]),
            "info": len([f for f in findings if f.severity == "info"])
        }
        
        return ScanResult(
            scan_type="code_security",
            timestamp=start_time,
            findings=findings,
            summary=summary,
            duration=duration,
            metadata={"directory": str(directory)}
        )


class DependencyScanner:
    """Scanner for dependency vulnerabilities."""
    
    def __init__(self):
        """Initialize dependency scanner."""
        self.known_vulnerabilities = self._load_vulnerability_database()
        
    def _load_vulnerability_database(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load known vulnerability database."""
        # In production, this would load from a real vulnerability database
        return {
            "qiskit": [
                {
                    "version_range": "<0.25.0",
                    "severity": "medium",
                    "cve_id": "CVE-2021-EXAMPLE",
                    "description": "Example vulnerability in older Qiskit versions"
                }
            ],
            "numpy": [
                {
                    "version_range": "<1.21.0",
                    "severity": "low",
                    "cve_id": "CVE-2021-NUMPY",
                    "description": "Buffer overflow in older NumPy versions"
                }
            ]
        }
        
    def scan_requirements(self, requirements_file: Path) -> ScanResult:
        """Scan requirements file for vulnerable dependencies."""
        start_time = datetime.utcnow()
        findings = []
        
        try:
            with open(requirements_file, 'r') as f:
                lines = f.readlines()
                
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                # Parse dependency
                if '==' in line:
                    package, version = line.split('==', 1)
                    package = package.strip()
                    version = version.strip()
                    
                    # Check for known vulnerabilities
                    if package in self.known_vulnerabilities:
                        for vuln in self.known_vulnerabilities[package]:
                            if self._version_in_range(version, vuln["version_range"]):
                                finding = SecurityFinding(
                                    severity=vuln["severity"],
                                    category="dependency",
                                    title=f"Vulnerable Dependency: {package}",
                                    description=vuln["description"],
                                    file_path=str(requirements_file),
                                    line_number=line_num,
                                    code_snippet=line,
                                    cwe_id=vuln.get("cve_id"),
                                    confidence="high"
                                )
                                findings.append(finding)
                                
        except Exception as e:
            logger.error(f"Error scanning requirements file: {e}")
            
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        summary = {
            "total_findings": len(findings),
            "critical": len([f for f in findings if f.severity == "critical"]),
            "high": len([f for f in findings if f.severity == "high"]),
            "medium": len([f for f in findings if f.severity == "medium"]),
            "low": len([f for f in findings if f.severity == "low"])
        }
        
        return ScanResult(
            scan_type="dependency_scan",
            timestamp=start_time,
            findings=findings,
            summary=summary,
            duration=duration,
            metadata={"requirements_file": str(requirements_file)}
        )
        
    def _version_in_range(self, version: str, version_range: str) -> bool:
        """Check if version is in vulnerable range."""
        # Simplified version comparison - in production use packaging library
        if version_range.startswith('<'):
            target = version_range[1:]
            return version < target
        elif version_range.startswith('<='):
            target = version_range[2:]
            return version <= target
        elif version_range.startswith('>'):
            target = version_range[1:]
            return version > target
        elif version_range.startswith('>='):
            target = version_range[2:]
            return version >= target
        else:
            return version == version_range


class QuantumSecurityScanner:
    """Scanner for quantum-specific security issues."""
    
    def __init__(self):
        """Initialize quantum security scanner."""
        self.quantum_patterns = self._load_quantum_patterns()
        
    def _load_quantum_patterns(self) -> List[Dict[str, Any]]:
        """Load quantum-specific security patterns."""
        return [
            {
                "pattern": r"(?i)circuit.*parameters.*log",
                "severity": "medium",
                "title": "Quantum Parameter Logging",
                "description": "Quantum circuit parameters may be logged, potentially exposing sensitive information",
                "recommendation": "Avoid logging quantum parameters or use secure logging"
            },
            {
                "pattern": r"(?i)(ibm_quantum_token|aws_access_key).*=.*['\"][^'\"]{10,}['\"]",
                "severity": "critical",
                "title": "Hardcoded Quantum Backend Credentials",
                "description": "Quantum backend credentials are hardcoded in source code",
                "recommendation": "Use environment variables or secure credential storage"
            },
            {
                "pattern": r"backend\.execute.*user_input",
                "severity": "high",
                "title": "Unvalidated Quantum Circuit Execution",
                "description": "Quantum circuit execution with unvalidated user input",
                "recommendation": "Validate and sanitize quantum circuits before execution"
            },
            {
                "pattern": r"(?i)noise_model.*=.*None",
                "severity": "low",
                "title": "Disabled Noise Modeling",
                "description": "Noise modeling is disabled, may not reflect real quantum hardware",
                "recommendation": "Consider enabling noise modeling for realistic simulations"
            }
        ]
        
    def scan_quantum_code(self, file_path: Path) -> List[SecurityFinding]:
        """Scan file for quantum-specific security issues."""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            lines = content.split('\n')
            
            for pattern_info in self.quantum_patterns:
                pattern = pattern_info["pattern"]
                
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line):
                        finding = SecurityFinding(
                            severity=pattern_info["severity"],
                            category="quantum_specific",
                            title=pattern_info["title"],
                            description=pattern_info["description"],
                            file_path=str(file_path),
                            line_number=line_num,
                            code_snippet=line.strip(),
                            recommendation=pattern_info.get("recommendation"),
                            confidence="medium"
                        )
                        findings.append(finding)
                        
        except Exception as e:
            logger.warning(f"Error scanning quantum code {file_path}: {e}")
            
        return findings
        
    def scan_quantum_circuits(self, circuits: List[Dict[str, Any]]) -> List[SecurityFinding]:
        """Scan quantum circuits for security issues."""
        findings = []
        
        for i, circuit in enumerate(circuits):
            # Check for suspicious patterns
            n_qubits = circuit.get('n_qubits', 0)
            gates = circuit.get('gates', [])
            
            # Check for excessive resource usage
            if n_qubits > 50:
                findings.append(SecurityFinding(
                    severity="medium",
                    category="quantum_specific",
                    title="High Qubit Count Circuit",
                    description=f"Circuit uses {n_qubits} qubits, may indicate resource abuse",
                    recommendation="Verify legitimate need for high qubit count"
                ))
                
            if len(gates) > 10000:
                findings.append(SecurityFinding(
                    severity="medium",
                    category="quantum_specific", 
                    title="Deep Circuit",
                    description=f"Circuit has {len(gates)} gates, may be excessively deep",
                    recommendation="Consider circuit optimization or verify legitimate complexity"
                ))
                
            # Check for suspicious gate patterns
            gate_types = [gate.get('type', '').lower() for gate in gates]
            unique_gates = set(gate_types)
            
            if len(unique_gates) == 1 and len(gate_types) > 100:
                findings.append(SecurityFinding(
                    severity="low",
                    category="quantum_specific",
                    title="Repetitive Gate Pattern",
                    description="Circuit contains highly repetitive gate pattern",
                    recommendation="Verify circuit correctness and optimization"
                ))
                
        return findings


class ConfigurationScanner:
    """Scanner for configuration security issues."""
    
    def __init__(self):
        """Initialize configuration scanner."""
        self.config_patterns = self._load_config_patterns()
        
    def _load_config_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load configuration security patterns."""
        return {
            "insecure_settings": [
                {
                    "pattern": r"(?i)debug\s*[:=]\s*true",
                    "severity": "medium",
                    "title": "Debug Mode Enabled",
                    "description": "Debug mode is enabled in configuration"
                },
                {
                    "pattern": r"(?i)ssl[_-]?verify\s*[:=]\s*false",
                    "severity": "high",
                    "title": "SSL Verification Disabled",
                    "description": "SSL certificate verification is disabled"
                }
            ],
            "default_credentials": [
                {
                    "pattern": r"(?i)(password|secret)\s*[:=]\s*['\"]?(admin|password|secret|default)['\"]?",
                    "severity": "high",
                    "title": "Default Credentials",
                    "description": "Default or weak credentials found in configuration"
                }
            ]
        }
        
    def scan_config_file(self, config_file: Path) -> List[SecurityFinding]:
        """Scan configuration file for security issues."""
        findings = []
        
        try:
            with open(config_file, 'r') as f:
                content = f.read()
                
            lines = content.split('\n')
            
            for category, patterns in self.config_patterns.items():
                for pattern_info in patterns:
                    pattern = pattern_info["pattern"]
                    
                    for line_num, line in enumerate(lines, 1):
                        if re.search(pattern, line):
                            finding = SecurityFinding(
                                severity=pattern_info["severity"],
                                category="configuration",
                                title=pattern_info["title"],
                                description=pattern_info["description"],
                                file_path=str(config_file),
                                line_number=line_num,
                                code_snippet=line.strip(),
                                confidence="high"
                            )
                            findings.append(finding)
                            
        except Exception as e:
            logger.warning(f"Error scanning config file {config_file}: {e}")
            
        return findings


class SecurityScanner:
    """Main security scanner orchestrator."""
    
    def __init__(self):
        """Initialize security scanner."""
        self.code_scanner = CodeSecurityScanner()
        self.dependency_scanner = DependencyScanner()
        self.quantum_scanner = QuantumSecurityScanner()
        self.config_scanner = ConfigurationScanner()
        
    def scan_project(self, project_path: Path) -> Dict[str, ScanResult]:
        """Perform comprehensive security scan of project."""
        results = {}
        
        # Code security scan
        logger.info("Running code security scan...")
        results["code_security"] = self.code_scanner.scan_directory(project_path)
        
        # Dependency scan
        requirements_files = list(project_path.glob("**/requirements*.txt"))
        if requirements_files:
            logger.info("Running dependency scan...")
            results["dependencies"] = self.dependency_scanner.scan_requirements(requirements_files[0])
            
        # Configuration scan
        config_files = []
        for pattern in ["**/*.yml", "**/*.yaml", "**/*.toml", "**/*.cfg", "**/*.ini"]:
            config_files.extend(project_path.glob(pattern))
            
        if config_files:
            logger.info("Running configuration scan...")
            config_findings = []
            for config_file in config_files[:10]:  # Limit to avoid too many files
                config_findings.extend(self.config_scanner.scan_config_file(config_file))
                
            results["configuration"] = ScanResult(
                scan_type="configuration",
                timestamp=datetime.utcnow(),
                findings=config_findings,
                summary={"total_findings": len(config_findings)},
                duration=0.0
            )
            
        return results
        
    def generate_report(self, scan_results: Dict[str, ScanResult], 
                       output_file: Path = None) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        report = {
            "scan_timestamp": datetime.utcnow().isoformat(),
            "scan_types": list(scan_results.keys()),
            "overall_summary": {
                "total_findings": 0,
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "info": 0
            },
            "scan_results": {},
            "recommendations": []
        }
        
        # Aggregate results
        for scan_type, result in scan_results.items():
            report["scan_results"][scan_type] = result.to_dict()
            
            # Update overall summary
            report["overall_summary"]["total_findings"] += result.summary.get("total_findings", 0)
            for severity in ["critical", "high", "medium", "low", "info"]:
                report["overall_summary"][severity] += result.summary.get(severity, 0)
                
        # Generate recommendations
        if report["overall_summary"]["critical"] > 0:
            report["recommendations"].append("Address critical security findings immediately")
        if report["overall_summary"]["high"] > 0:
            report["recommendations"].append("Review and fix high severity issues")
        if report["overall_summary"]["medium"] > 5:
            report["recommendations"].append("Consider addressing medium severity issues")
            
        # Save report if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
                
        return report


# Global security scanner
_global_security_scanner: Optional[SecurityScanner] = None

def get_security_scanner() -> SecurityScanner:
    """Get global security scanner."""" 
    global _global_security_scanner
    if _global_security_scanner is None:
        _global_security_scanner = SecurityScanner()
    return _global_security_scanner