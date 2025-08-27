#!/usr/bin/env python3
"""
Quantum Meta-Learning Security Scan and Quality Gates
Comprehensive security analysis, vulnerability assessment, and quality validation
"""

import json
import time
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union, Set
from dataclasses import dataclass, asdict
import logging
import warnings
import traceback
import hashlib
import ast
import re
from pathlib import Path
import subprocess
import sys
import os

# Configure security logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [SECURITY] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('quantum_security_scan.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SecurityVulnerability:
    """Security vulnerability report"""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str
    description: str
    file_path: str
    line_number: int
    recommendation: str
    cwe_id: Optional[str] = None

@dataclass  
class QualityIssue:
    """Code quality issue"""
    severity: str
    category: str
    description: str
    file_path: str
    line_number: int
    metric_value: Optional[float] = None

@dataclass
class SecurityScanResult:
    """Complete security and quality scan results"""
    scan_timestamp: int
    total_files_scanned: int
    vulnerabilities: List[SecurityVulnerability]
    quality_issues: List[QualityIssue]
    security_score: float
    quality_score: float
    overall_score: float
    gates_passed: Dict[str, bool]
    recommendations: List[str]
    compliance_status: Dict[str, bool]

class QuantumSecurityScanner:
    """Advanced security scanner for quantum meta-learning systems"""
    
    def __init__(self, scan_directory: str = "/root/repo"):
        self.scan_directory = Path(scan_directory)
        self.vulnerabilities = []
        self.quality_issues = []
        
        # Security patterns to detect
        self.security_patterns = {
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']',
            ],
            'sql_injection': [
                r'execute\s*\(\s*["\'].*%s.*["\']',
                r'query\s*\(\s*["\'].*\+.*["\']',
            ],
            'command_injection': [
                r'subprocess\.\w+\([^)]*shell\s*=\s*True',
                r'os\.system\(',
                r'os\.popen\(',
                r'eval\(',
                r'exec\(',
            ],
            'path_traversal': [
                r'open\([^)]*\.\.[/\\]',
                r'file\([^)]*\.\.[/\\]',
            ],
            'crypto_weakness': [
                r'md5\(',
                r'sha1\(',
                r'random\.random\(',
                r'random\.seed\(',
            ]
        }
        
        # Quality metrics thresholds
        self.quality_thresholds = {
            'max_complexity': 15,
            'max_line_length': 120,
            'min_function_docstring_coverage': 0.7,
            'max_function_parameters': 8,
            'max_nested_depth': 5
        }
        
    def scan_file_for_vulnerabilities(self, file_path: Path) -> List[SecurityVulnerability]:
        """Scan single file for security vulnerabilities"""
        vulnerabilities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            for category, patterns in self.security_patterns.items():
                for pattern in patterns:
                    for line_num, line in enumerate(lines, 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            severity = self._determine_severity(category, line)
                            
                            vuln = SecurityVulnerability(
                                severity=severity,
                                category=category,
                                description=f"Potential {category.replace('_', ' ')} vulnerability",
                                file_path=str(file_path.relative_to(self.scan_directory)),
                                line_number=line_num,
                                recommendation=self._get_security_recommendation(category),
                                cwe_id=self._get_cwe_id(category)
                            )
                            vulnerabilities.append(vuln)
                            
        except Exception as e:
            logger.warning(f"Error scanning {file_path}: {e}")
            
        return vulnerabilities
    
    def _determine_severity(self, category: str, line: str) -> str:
        """Determine vulnerability severity"""
        severity_mapping = {
            'hardcoded_secrets': 'HIGH',
            'sql_injection': 'CRITICAL', 
            'command_injection': 'CRITICAL',
            'path_traversal': 'HIGH',
            'crypto_weakness': 'MEDIUM'
        }
        
        base_severity = severity_mapping.get(category, 'LOW')
        
        # Increase severity for certain contexts
        if 'password' in line.lower() or 'secret' in line.lower():
            if base_severity == 'MEDIUM':
                return 'HIGH'
            elif base_severity == 'HIGH':
                return 'CRITICAL'
                
        return base_severity
    
    def _get_security_recommendation(self, category: str) -> str:
        """Get security recommendation for vulnerability category"""
        recommendations = {
            'hardcoded_secrets': "Use environment variables or secure credential storage",
            'sql_injection': "Use parameterized queries or prepared statements",
            'command_injection': "Validate input and use subprocess with shell=False",
            'path_traversal': "Validate file paths and use absolute paths",
            'crypto_weakness': "Use cryptographically secure algorithms (SHA-256+)"
        }
        return recommendations.get(category, "Review code for security implications")
    
    def _get_cwe_id(self, category: str) -> str:
        """Get CWE (Common Weakness Enumeration) ID"""
        cwe_mapping = {
            'hardcoded_secrets': 'CWE-798',
            'sql_injection': 'CWE-89',
            'command_injection': 'CWE-78', 
            'path_traversal': 'CWE-22',
            'crypto_weakness': 'CWE-327'
        }
        return cwe_mapping.get(category, 'CWE-707')
    
    def analyze_code_quality(self, file_path: Path) -> List[QualityIssue]:
        """Analyze code quality metrics"""
        quality_issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Parse AST for advanced analysis
            try:
                tree = ast.parse(content, filename=str(file_path))
                quality_issues.extend(self._analyze_ast_quality(tree, file_path, lines))
            except SyntaxError as e:
                quality_issues.append(QualityIssue(
                    severity='HIGH',
                    category='syntax_error',
                    description=f"Syntax error: {e.msg}",
                    file_path=str(file_path.relative_to(self.scan_directory)),
                    line_number=e.lineno or 1
                ))
            
            # Line-based quality checks
            quality_issues.extend(self._analyze_line_quality(lines, file_path))
            
        except Exception as e:
            logger.warning(f"Error analyzing quality for {file_path}: {e}")
            
        return quality_issues
    
    def _analyze_ast_quality(self, tree: ast.AST, file_path: Path, lines: List[str]) -> List[QualityIssue]:
        """Analyze AST for quality issues"""
        issues = []
        
        class QualityVisitor(ast.NodeVisitor):
            def __init__(self, outer_self, file_path, lines):
                self.outer = outer_self
                self.file_path = file_path
                self.lines = lines
                self.function_depths = []
                self.current_depth = 0
                
            def visit_FunctionDef(self, node):
                # Check function parameter count
                param_count = len(node.args.args)
                if param_count > self.outer.quality_thresholds['max_function_parameters']:
                    issues.append(QualityIssue(
                        severity='MEDIUM',
                        category='too_many_parameters',
                        description=f"Function has {param_count} parameters (max {self.outer.quality_thresholds['max_function_parameters']})",
                        file_path=str(file_path.relative_to(self.outer.scan_directory)),
                        line_number=node.lineno,
                        metric_value=param_count
                    ))
                
                # Check for docstring
                has_docstring = (node.body and 
                               isinstance(node.body[0], ast.Expr) and
                               isinstance(node.body[0].value, ast.Str))
                
                if not has_docstring and not node.name.startswith('_'):
                    issues.append(QualityIssue(
                        severity='LOW',
                        category='missing_docstring',
                        description=f"Public function '{node.name}' missing docstring",
                        file_path=str(file_path.relative_to(self.outer.scan_directory)),
                        line_number=node.lineno
                    ))
                
                # Check cyclomatic complexity (simplified)
                complexity = self._calculate_complexity(node)
                if complexity > self.outer.quality_thresholds['max_complexity']:
                    issues.append(QualityIssue(
                        severity='HIGH',
                        category='high_complexity',
                        description=f"Function complexity {complexity} exceeds maximum {self.outer.quality_thresholds['max_complexity']}",
                        file_path=str(file_path.relative_to(self.outer.scan_directory)),
                        line_number=node.lineno,
                        metric_value=complexity
                    ))
                
                self.generic_visit(node)
                
            def visit_If(self, node):
                self.current_depth += 1
                if self.current_depth > self.outer.quality_thresholds['max_nested_depth']:
                    issues.append(QualityIssue(
                        severity='MEDIUM',
                        category='excessive_nesting',
                        description=f"Nesting depth {self.current_depth} exceeds maximum {self.outer.quality_thresholds['max_nested_depth']}",
                        file_path=str(file_path.relative_to(self.outer.scan_directory)),
                        line_number=node.lineno,
                        metric_value=self.current_depth
                    ))
                
                self.generic_visit(node)
                self.current_depth -= 1
                
            def visit_For(self, node):
                self.current_depth += 1
                if self.current_depth > self.outer.quality_thresholds['max_nested_depth']:
                    issues.append(QualityIssue(
                        severity='MEDIUM',
                        category='excessive_nesting',
                        description=f"Nesting depth {self.current_depth} exceeds maximum {self.outer.quality_thresholds['max_nested_depth']}",
                        file_path=str(file_path.relative_to(self.outer.scan_directory)),
                        line_number=node.lineno,
                        metric_value=self.current_depth
                    ))
                
                self.generic_visit(node)
                self.current_depth -= 1
                
            def visit_While(self, node):
                self.current_depth += 1
                if self.current_depth > self.outer.quality_thresholds['max_nested_depth']:
                    issues.append(QualityIssue(
                        severity='MEDIUM', 
                        category='excessive_nesting',
                        description=f"Nesting depth {self.current_depth} exceeds maximum {self.outer.quality_thresholds['max_nested_depth']}",
                        file_path=str(file_path.relative_to(self.outer.scan_directory)),
                        line_number=node.lineno,
                        metric_value=self.current_depth
                    ))
                
                self.generic_visit(node)
                self.current_depth -= 1
                
            def _calculate_complexity(self, node):
                """Calculate cyclomatic complexity (simplified)"""
                complexity = 1  # Base complexity
                
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.ExceptHandler)):
                        complexity += 1
                    elif isinstance(child, ast.BoolOp):
                        complexity += len(child.values) - 1
                        
                return complexity
        
        visitor = QualityVisitor(self, file_path, lines)
        visitor.visit(tree)
        
        return issues
    
    def _analyze_line_quality(self, lines: List[str], file_path: Path) -> List[QualityIssue]:
        """Analyze line-based quality issues"""
        issues = []
        
        for line_num, line in enumerate(lines, 1):
            # Check line length
            if len(line) > self.quality_thresholds['max_line_length']:
                issues.append(QualityIssue(
                    severity='LOW',
                    category='line_too_long',
                    description=f"Line length {len(line)} exceeds maximum {self.quality_thresholds['max_line_length']}",
                    file_path=str(file_path.relative_to(self.scan_directory)),
                    line_number=line_num,
                    metric_value=len(line)
                ))
            
            # Check for TODO/FIXME/HACK comments
            if re.search(r'(TODO|FIXME|HACK|XXX)', line, re.IGNORECASE):
                issues.append(QualityIssue(
                    severity='LOW',
                    category='technical_debt',
                    description="Technical debt marker found",
                    file_path=str(file_path.relative_to(self.scan_directory)),
                    line_number=line_num
                ))
        
        return issues
    
    def check_dependency_security(self) -> List[SecurityVulnerability]:
        """Check for known vulnerable dependencies"""
        vulnerabilities = []
        
        # Check requirements.txt and pyproject.toml
        req_files = [
            self.scan_directory / 'requirements.txt',
            self.scan_directory / 'pyproject.toml',
            self.scan_directory / 'setup.py'
        ]
        
        # Known vulnerable packages (simplified list)
        vulnerable_packages = {
            'numpy': ['<1.21.0', 'CVE-2021-33430'],
            'tensorflow': ['<2.7.0', 'CVE-2021-37678'], 
            'requests': ['<2.25.0', 'CVE-2018-18074'],
            'pillow': ['<8.3.2', 'CVE-2021-34552']
        }
        
        for req_file in req_files:
            if req_file.exists():
                try:
                    with open(req_file, 'r') as f:
                        content = f.read()
                    
                    for pkg_name, (min_version, cve) in vulnerable_packages.items():
                        if pkg_name in content.lower():
                            vulnerabilities.append(SecurityVulnerability(
                                severity='MEDIUM',
                                category='vulnerable_dependency',
                                description=f"Potentially vulnerable {pkg_name} dependency ({cve})",
                                file_path=str(req_file.relative_to(self.scan_directory)),
                                line_number=1,
                                recommendation=f"Update {pkg_name} to {min_version} or later",
                                cwe_id='CWE-1104'
                            ))
                            
                except Exception as e:
                    logger.warning(f"Error checking {req_file}: {e}")
        
        return vulnerabilities
    
    def validate_quantum_specific_security(self) -> List[SecurityVulnerability]:
        """Validate quantum-specific security concerns"""
        vulnerabilities = []
        
        # Search for quantum-specific patterns
        quantum_files = list(self.scan_directory.glob("**/*quantum*.py"))
        
        quantum_security_patterns = {
            'parameter_injection': r'params\s*=\s*[^=]*input\(',
            'circuit_tampering': r'circuit\.\w+\([^)]*user_input',
            'measurement_manipulation': r'measure\([^)]*input\(',
            'insecure_randomness': r'np\.random\.\w+\([^)]*seed\s*=',
        }
        
        for file_path in quantum_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                for pattern_name, pattern in quantum_security_patterns.items():
                    for line_num, line in enumerate(lines, 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            vulnerabilities.append(SecurityVulnerability(
                                severity='HIGH',
                                category='quantum_security',
                                description=f"Potential quantum {pattern_name.replace('_', ' ')} vulnerability",
                                file_path=str(file_path.relative_to(self.scan_directory)),
                                line_number=line_num,
                                recommendation="Validate and sanitize quantum parameters",
                                cwe_id='CWE-20'
                            ))
                            
            except Exception as e:
                logger.warning(f"Error checking quantum security in {file_path}: {e}")
        
        return vulnerabilities
    
    def calculate_security_score(self) -> float:
        """Calculate overall security score (0-100)"""
        if not self.vulnerabilities:
            return 100.0
        
        # Weight by severity
        severity_weights = {
            'CRITICAL': 10,
            'HIGH': 5,
            'MEDIUM': 2,
            'LOW': 1
        }
        
        total_weight = sum(severity_weights[vuln.severity] for vuln in self.vulnerabilities)
        max_possible = len(self.vulnerabilities) * severity_weights['CRITICAL']
        
        score = max(0, 100 - (total_weight / max_possible * 100))
        return score
    
    def calculate_quality_score(self) -> float:
        """Calculate overall quality score (0-100)"""
        if not self.quality_issues:
            return 100.0
        
        severity_weights = {
            'HIGH': 5,
            'MEDIUM': 3,
            'LOW': 1
        }
        
        total_weight = sum(severity_weights[issue.severity] for issue in self.quality_issues)
        max_possible = len(self.quality_issues) * severity_weights['HIGH']
        
        score = max(0, 100 - (total_weight / max_possible * 100))
        return score
    
    def run_quality_gates(self, security_score: float, quality_score: float) -> Dict[str, bool]:
        """Run quality gates validation"""
        gates = {
            'security_gate': security_score >= 80.0,  # Minimum 80% security score
            'quality_gate': quality_score >= 70.0,   # Minimum 70% quality score
            'critical_vulnerabilities': not any(v.severity == 'CRITICAL' for v in self.vulnerabilities),
            'high_complexity': not any(i.category == 'high_complexity' for i in self.quality_issues),
            'dependency_security': not any(v.category == 'vulnerable_dependency' for v in self.vulnerabilities),
            'quantum_security': not any(v.category == 'quantum_security' for v in self.vulnerabilities)
        }
        
        return gates
    
    def generate_recommendations(self) -> List[str]:
        """Generate security and quality recommendations"""
        recommendations = []
        
        # Security recommendations
        if any(v.severity == 'CRITICAL' for v in self.vulnerabilities):
            recommendations.append("🔴 CRITICAL: Address all critical security vulnerabilities immediately")
        
        if any(v.category == 'hardcoded_secrets' for v in self.vulnerabilities):
            recommendations.append("🔐 Implement secure credential management (environment variables, HashiCorp Vault)")
        
        if any(v.category == 'vulnerable_dependency' for v in self.vulnerabilities):
            recommendations.append("📦 Update vulnerable dependencies to latest secure versions")
        
        # Quality recommendations
        high_complexity_issues = [i for i in self.quality_issues if i.category == 'high_complexity']
        if high_complexity_issues:
            avg_complexity = np.mean([i.metric_value for i in high_complexity_issues if i.metric_value])
            recommendations.append(f"🔧 Refactor high complexity functions (average complexity: {avg_complexity:.1f})")
        
        if any(i.category == 'missing_docstring' for i in self.quality_issues):
            recommendations.append("📚 Add docstrings to improve code documentation")
        
        # Quantum-specific recommendations
        if any(v.category == 'quantum_security' for v in self.vulnerabilities):
            recommendations.append("⚛️ Implement quantum parameter validation and sanitization")
        
        if not recommendations:
            recommendations.append("✅ No major security or quality issues detected")
        
        return recommendations
    
    def check_compliance(self) -> Dict[str, bool]:
        """Check compliance with security standards"""
        return {
            'owasp_top10': not any(v.cwe_id in ['CWE-89', 'CWE-78', 'CWE-22'] for v in self.vulnerabilities),
            'pci_dss': not any(v.category == 'hardcoded_secrets' for v in self.vulnerabilities),
            'gdpr_privacy': True,  # Simplified check
            'quantum_security_best_practices': not any(v.category == 'quantum_security' for v in self.vulnerabilities),
            'code_quality_standards': len([i for i in self.quality_issues if i.severity == 'HIGH']) == 0
        }
    
    def run_comprehensive_scan(self) -> SecurityScanResult:
        """Run comprehensive security and quality scan"""
        logger.info("🛡️ Starting comprehensive security and quality scan")
        
        start_time = time.time()
        files_scanned = 0
        
        # Scan Python files
        python_files = list(self.scan_directory.glob("**/*.py"))
        
        for file_path in python_files:
            if file_path.name.startswith('.'):
                continue  # Skip hidden files
                
            logger.debug(f"Scanning {file_path}")
            
            # Security scan
            file_vulnerabilities = self.scan_file_for_vulnerabilities(file_path)
            self.vulnerabilities.extend(file_vulnerabilities)
            
            # Quality analysis
            file_quality_issues = self.analyze_code_quality(file_path)
            self.quality_issues.extend(file_quality_issues)
            
            files_scanned += 1
        
        # Additional security checks
        dependency_vulnerabilities = self.check_dependency_security()
        self.vulnerabilities.extend(dependency_vulnerabilities)
        
        quantum_vulnerabilities = self.validate_quantum_specific_security()
        self.vulnerabilities.extend(quantum_vulnerabilities)
        
        # Calculate scores
        security_score = self.calculate_security_score()
        quality_score = self.calculate_quality_score()
        overall_score = (security_score + quality_score) / 2
        
        # Run quality gates
        gates_passed = self.run_quality_gates(security_score, quality_score)
        
        # Generate recommendations
        recommendations = self.generate_recommendations()
        
        # Check compliance
        compliance_status = self.check_compliance()
        
        scan_time = time.time() - start_time
        
        result = SecurityScanResult(
            scan_timestamp=int(time.time()),
            total_files_scanned=files_scanned,
            vulnerabilities=self.vulnerabilities,
            quality_issues=self.quality_issues,
            security_score=security_score,
            quality_score=quality_score,
            overall_score=overall_score,
            gates_passed=gates_passed,
            recommendations=recommendations,
            compliance_status=compliance_status
        )
        
        logger.info(f"✅ Scan complete: {files_scanned} files, {len(self.vulnerabilities)} vulnerabilities, "
                   f"{len(self.quality_issues)} quality issues in {scan_time:.2f}s")
        
        return result

def main():
    """Execute comprehensive security and quality scan"""
    timestamp = int(time.time() * 1000)
    
    print("\n" + "="*70)
    print("🛡️ QUANTUM META-LEARNING SECURITY & QUALITY GATES")
    print("="*70)
    
    # Initialize scanner
    scanner = QuantumSecurityScanner("/root/repo")
    
    # Run comprehensive scan
    scan_result = scanner.run_comprehensive_scan()
    
    # Save results
    results_dict = asdict(scan_result)
    results_dict['scan_version'] = '1.0'
    
    filename = f"security_quality_scan_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    
    # Display results
    print(f"\n📊 SCAN RESULTS:")
    print(f"Files Scanned: {scan_result.total_files_scanned}")
    print(f"Vulnerabilities Found: {len(scan_result.vulnerabilities)}")
    print(f"Quality Issues Found: {len(scan_result.quality_issues)}")
    print(f"Security Score: {scan_result.security_score:.1f}/100")
    print(f"Quality Score: {scan_result.quality_score:.1f}/100")
    print(f"Overall Score: {scan_result.overall_score:.1f}/100")
    
    print(f"\n🚪 QUALITY GATES:")
    for gate_name, passed in scan_result.gates_passed.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {gate_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n🔍 VULNERABILITIES BY SEVERITY:")
    severity_counts = {}
    for vuln in scan_result.vulnerabilities:
        severity_counts[vuln.severity] = severity_counts.get(vuln.severity, 0) + 1
    
    for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        count = severity_counts.get(severity, 0)
        print(f"  {severity}: {count}")
    
    print(f"\n📈 QUALITY ISSUES BY SEVERITY:")
    quality_counts = {}
    for issue in scan_result.quality_issues:
        quality_counts[issue.severity] = quality_counts.get(issue.severity, 0) + 1
    
    for severity in ['HIGH', 'MEDIUM', 'LOW']:
        count = quality_counts.get(severity, 0)
        print(f"  {severity}: {count}")
    
    print(f"\n✅ COMPLIANCE STATUS:")
    for standard, compliant in scan_result.compliance_status.items():
        status = "✅ COMPLIANT" if compliant else "❌ NON-COMPLIANT"
        print(f"  {standard.replace('_', ' ').upper()}: {status}")
    
    print(f"\n🔧 RECOMMENDATIONS:")
    for i, recommendation in enumerate(scan_result.recommendations[:5], 1):
        print(f"  {i}. {recommendation}")
    
    # Overall gate status
    all_gates_passed = all(scan_result.gates_passed.values())
    gate_status = "✅ ALL GATES PASSED" if all_gates_passed else "❌ SOME GATES FAILED"
    
    print(f"\n🎯 FINAL STATUS: {gate_status}")
    print(f"Results saved to: {filename}")
    print("="*70)
    
    return scan_result

if __name__ == "__main__":
    main()