"""Quantum-specific security validation and circuit sanitization."""

import re
import json
import hashlib
import numpy as np
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class SecurityIssue:
    """Quantum security issue."""
    
    severity: str  # critical, high, medium, low
    issue_type: str
    description: str
    recommendation: str
    affected_component: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CircuitSecurityReport:
    """Security report for quantum circuit."""
    
    circuit_id: str
    is_secure: bool
    issues: List[SecurityIssue]
    risk_score: int  # 0-10
    sanitized_circuit: Optional[Dict[str, Any]] = None
    
    def get_max_severity(self) -> str:
        """Get maximum severity level from issues."""
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        max_severity = "low"
        max_value = 0
        
        for issue in self.issues:
            value = severity_order.get(issue.severity, 0)
            if value > max_value:
                max_value = value
                max_severity = issue.severity
                
        return max_severity


class QuantumSecurityPolicy:
    """Security policy for quantum operations."""
    
    def __init__(self, policy_config: Dict[str, Any] = None):
        """Initialize quantum security policy."""
        self.config = policy_config or self._default_policy()
        
    def _default_policy(self) -> Dict[str, Any]:
        """Default security policy configuration."""
        return {
            "max_qubits": 100,
            "max_gates": 50000,
            "max_circuit_depth": 1000,
            "allowed_gate_types": {
                "h", "hadamard", "x", "pauli_x", "y", "pauli_y", "z", "pauli_z",
                "s", "sdg", "t", "tdg", "rx", "ry", "rz", "u1", "u2", "u3",
                "cnot", "cx", "cy", "cz", "ch", "crx", "cry", "crz",
                "swap", "iswap", "cswap", "ccx", "toffoli",
                "measure", "measurement", "barrier", "reset"
            },
            "forbidden_patterns": [
                "excessive_entanglement",
                "parameter_extraction",
                "hardware_fingerprinting",
                "noise_exploitation"
            ],
            "parameter_constraints": {
                "angle_range": [-4 * np.pi, 4 * np.pi],
                "max_parameter_count": 10000,
                "precision_limit": 1e-12
            },
            "resource_limits": {
                "max_measurement_ratio": 0.5,  # Max measurements per qubit
                "max_barrier_count": 1000,
                "max_reset_count": 100
            }
        }
        
    def is_gate_allowed(self, gate_type: str) -> bool:
        """Check if gate type is allowed."""
        return gate_type.lower() in self.config["allowed_gate_types"]
        
    def validate_resource_usage(self, circuit: Dict[str, Any]) -> List[SecurityIssue]:
        """Validate circuit resource usage against policy."""
        issues = []
        n_qubits = circuit.get("n_qubits", 0)
        gates = circuit.get("gates", [])
        
        # Check qubit count
        if n_qubits > self.config["max_qubits"]:
            issues.append(SecurityIssue(
                severity="high",
                issue_type="resource_limit",
                description=f"Circuit uses {n_qubits} qubits, exceeds limit of {self.config['max_qubits']}",
                recommendation="Reduce qubit count or request higher limits",
                affected_component="circuit"
            ))
            
        # Check gate count
        if len(gates) > self.config["max_gates"]:
            issues.append(SecurityIssue(
                severity="medium",
                issue_type="resource_limit",
                description=f"Circuit has {len(gates)} gates, exceeds limit of {self.config['max_gates']}",
                recommendation="Optimize circuit or split into smaller circuits",
                affected_component="gates"
            ))
            
        return issues


class CircuitAnalyzer:
    """Analyzes quantum circuits for security issues."""
    
    def __init__(self, policy: QuantumSecurityPolicy = None):
        """Initialize circuit analyzer."""
        self.policy = policy or QuantumSecurityPolicy()
        
    def analyze_circuit_structure(self, circuit: Dict[str, Any]) -> List[SecurityIssue]:
        """Analyze circuit structure for security issues."""
        issues = []
        gates = circuit.get("gates", [])
        n_qubits = circuit.get("n_qubits", 0)
        
        if not gates:
            return issues
            
        # Analyze gate patterns
        gate_types = [gate.get("type", "").lower() for gate in gates]
        gate_counts = {}
        
        for gate_type in gate_types:
            gate_counts[gate_type] = gate_counts.get(gate_type, 0) + 1
            
        # Check for excessive repetition
        for gate_type, count in gate_counts.items():
            if count > len(gates) * 0.8:
                issues.append(SecurityIssue(
                    severity="medium",
                    issue_type="suspicious_pattern",
                    description=f"Gate '{gate_type}' appears {count} times ({count/len(gates)*100:.1f}% of circuit)",
                    recommendation="Verify circuit correctness and consider optimization",
                    affected_component="gates",
                    metadata={"gate_type": gate_type, "count": count}
                ))
                
        # Check measurement patterns
        measurement_count = sum(1 for gt in gate_types if "measure" in gt)
        if measurement_count > n_qubits * self.policy.config["resource_limits"]["max_measurement_ratio"]:
            issues.append(SecurityIssue(
                severity="medium",
                issue_type="measurement_abuse",
                description=f"Excessive measurements: {measurement_count} for {n_qubits} qubits",
                recommendation="Reduce unnecessary measurements",
                affected_component="measurements"
            ))
            
        # Check for forbidden gate types
        for gate in gates:
            gate_type = gate.get("type", "").lower()
            if not self.policy.is_gate_allowed(gate_type):
                issues.append(SecurityIssue(
                    severity="high",
                    issue_type="forbidden_gate",
                    description=f"Forbidden gate type: {gate_type}",
                    recommendation="Remove forbidden gates or request permission",
                    affected_component="gates",
                    metadata={"gate_type": gate_type}
                ))
                
        return issues
        
    def analyze_entanglement_patterns(self, circuit: Dict[str, Any]) -> List[SecurityIssue]:
        """Analyze entanglement patterns for potential security issues."""
        issues = []
        gates = circuit.get("gates", [])
        n_qubits = circuit.get("n_qubits", 0)
        
        # Track two-qubit gates for entanglement analysis
        two_qubit_gates = []
        for gate in gates:
            gate_type = gate.get("type", "").lower()
            if gate_type in ["cnot", "cx", "cy", "cz", "swap", "iswap"]:
                wires = gate.get("wires", gate.get("qubits", []))
                if len(wires) >= 2:
                    two_qubit_gates.append((wires[0], wires[1]))
                    
        # Check for excessive entanglement
        if len(two_qubit_gates) > n_qubits * 10:  # Heuristic threshold
            issues.append(SecurityIssue(
                severity="medium",
                issue_type="excessive_entanglement",
                description=f"High number of two-qubit gates: {len(two_qubit_gates)}",
                recommendation="Verify need for extensive entanglement",
                affected_component="entanglement"
            ))
            
        # Check for star-pattern entanglement (potential fingerprinting)
        qubit_connections = {}
        for q1, q2 in two_qubit_gates:
            qubit_connections[q1] = qubit_connections.get(q1, 0) + 1
            qubit_connections[q2] = qubit_connections.get(q2, 0) + 1
            
        # Look for highly connected qubits
        for qubit, connections in qubit_connections.items():
            if connections > n_qubits * 0.5:  # Connected to >50% of qubits
                issues.append(SecurityIssue(
                    severity="low",
                    issue_type="hub_qubit_pattern",
                    description=f"Qubit {qubit} highly connected ({connections} connections)",
                    recommendation="Verify circuit design, may indicate fingerprinting attempt",
                    affected_component="connectivity",
                    metadata={"qubit": qubit, "connections": connections}
                ))
                
        return issues
        
    def analyze_parameter_security(self, circuit: Dict[str, Any]) -> List[SecurityIssue]:
        """Analyze circuit parameters for security issues."""
        issues = []
        gates = circuit.get("gates", [])
        constraints = self.policy.config["parameter_constraints"]
        
        parameter_count = 0
        suspicious_values = []
        
        for gate in gates:
            # Check various parameter fields
            for param_field in ["angle", "angles", "parameters", "rotation"]:
                if param_field in gate:
                    param_value = gate[param_field]
                    
                    if isinstance(param_value, (int, float)):
                        parameter_count += 1
                        
                        # Check parameter range
                        if (param_value < constraints["angle_range"][0] or 
                            param_value > constraints["angle_range"][1]):
                            issues.append(SecurityIssue(
                                severity="medium",
                                issue_type="parameter_range",
                                description=f"Parameter {param_value} outside safe range",
                                recommendation="Ensure parameter values are within expected range",
                                affected_component="parameters"
                            ))
                            
                        # Check for suspicious precision
                        if abs(param_value) < constraints["precision_limit"]:
                            suspicious_values.append(param_value)
                            
                    elif isinstance(param_value, list):
                        parameter_count += len(param_value)
                        
        # Check total parameter count
        if parameter_count > constraints["max_parameter_count"]:
            issues.append(SecurityIssue(
                severity="medium",
                issue_type="parameter_count",
                description=f"Too many parameters: {parameter_count}",
                recommendation="Reduce parameter count or optimize circuit",
                affected_component="parameters"
            ))
            
        # Check for suspicious precision patterns
        if len(suspicious_values) > 10:
            issues.append(SecurityIssue(
                severity="low",
                issue_type="precision_pattern",
                description=f"Many extremely small parameters detected: {len(suspicious_values)}",
                recommendation="Verify parameter values are intentional",
                affected_component="parameters"
            ))
            
        return issues


class ParameterSanitizer:
    """Sanitizes quantum circuit parameters."""
    
    def __init__(self, precision_digits: int = 10):
        """Initialize parameter sanitizer."""
        self.precision_digits = precision_digits
        
    def sanitize_parameter(self, param: Union[float, int, complex]) -> Union[float, int, complex]:
        """Sanitize individual parameter."""
        if isinstance(param, complex):
            # Round complex parameters
            real_part = round(param.real, self.precision_digits)
            imag_part = round(param.imag, self.precision_digits)
            return complex(real_part, imag_part)
        elif isinstance(param, float):
            # Round float parameters
            return round(param, self.precision_digits)
        else:
            return param
            
    def sanitize_parameters(self, parameters: Any) -> Any:
        """Sanitize parameter collection."""
        if isinstance(parameters, (list, tuple)):
            return [self.sanitize_parameter(p) for p in parameters]
        elif isinstance(parameters, dict):
            return {k: self.sanitize_parameters(v) for k, v in parameters.items()}
        else:
            return self.sanitize_parameter(parameters)


class CircuitSanitizer:
    """Sanitizes quantum circuits for security."""
    
    def __init__(self, policy: QuantumSecurityPolicy = None):
        """Initialize circuit sanitizer."""
        self.policy = policy or QuantumSecurityPolicy()
        self.param_sanitizer = ParameterSanitizer()
        
    def sanitize_circuit(self, circuit: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Sanitize quantum circuit."""
        sanitized = circuit.copy()
        modifications = []
        
        # Sanitize gates
        if "gates" in sanitized:
            sanitized_gates = []
            
            for i, gate in enumerate(sanitized["gates"]):
                sanitized_gate = self._sanitize_gate(gate)
                
                # Skip forbidden gates
                gate_type = gate.get("type", "").lower()
                if self.policy.is_gate_allowed(gate_type):
                    sanitized_gates.append(sanitized_gate)
                else:
                    modifications.append(f"Removed forbidden gate {gate_type} at position {i}")
                    
            sanitized["gates"] = sanitized_gates
            
        # Limit qubit count
        max_qubits = self.policy.config["max_qubits"]
        if sanitized.get("n_qubits", 0) > max_qubits:
            sanitized["n_qubits"] = max_qubits
            modifications.append(f"Limited qubit count to {max_qubits}")
            
        # Limit gate count
        max_gates = self.policy.config["max_gates"]
        if len(sanitized.get("gates", [])) > max_gates:
            sanitized["gates"] = sanitized["gates"][:max_gates]
            modifications.append(f"Limited gate count to {max_gates}")
            
        return sanitized, modifications
        
    def _sanitize_gate(self, gate: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize individual gate."""
        sanitized_gate = gate.copy()
        
        # Sanitize parameters
        for param_field in ["angle", "angles", "parameters", "rotation"]:
            if param_field in sanitized_gate:
                sanitized_gate[param_field] = self.param_sanitizer.sanitize_parameters(
                    sanitized_gate[param_field]
                )
                
        # Ensure qubit indices are valid
        for wire_field in ["wires", "qubit", "qubits"]:
            if wire_field in sanitized_gate:
                wires = sanitized_gate[wire_field]
                if isinstance(wires, int):
                    sanitized_gate[wire_field] = max(0, wires)
                elif isinstance(wires, list):
                    sanitized_gate[wire_field] = [max(0, w) for w in wires if isinstance(w, int)]
                    
        return sanitized_gate


class NoiseAnalyzer:
    """Analyzes noise patterns for potential information leakage."""
    
    def __init__(self):
        """Initialize noise analyzer."""
        self.baseline_patterns = {}
        
    def analyze_noise_signature(self, results: Dict[str, Any], 
                               backend_name: str) -> List[SecurityIssue]:
        """Analyze noise patterns in quantum results."""
        issues = []
        
        if "counts" not in results:
            return issues
            
        counts = results["counts"]
        if not isinstance(counts, dict):
            return issues
            
        # Calculate noise signature
        total_shots = sum(counts.values())
        if total_shots == 0:
            return issues
            
        # Calculate probability distribution
        probabilities = {state: count/total_shots for state, count in counts.items()}
        
        # Check for unexpected patterns
        if len(probabilities) == 1:
            # Perfect measurement - suspicious for real hardware
            if "simulator" not in backend_name.lower():
                issues.append(SecurityIssue(
                    severity="low",
                    issue_type="perfect_measurement",
                    description="Perfect measurement result on real hardware",
                    recommendation="Verify measurement result authenticity",
                    affected_component="results"
                ))
                
        # Check for hardware fingerprinting patterns
        entropy = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)
        expected_entropy = np.log2(len(probabilities)) if len(probabilities) > 1 else 0
        
        if entropy > 0 and abs(entropy - expected_entropy) > 2:
            issues.append(SecurityIssue(
                severity="medium",
                issue_type="entropy_anomaly",
                description=f"Unusual entropy pattern detected: {entropy:.2f}",
                recommendation="Check for potential hardware fingerprinting",
                affected_component="results",
                metadata={"entropy": entropy, "expected": expected_entropy}
            ))
            
        return issues


class QuantumSecurityValidator:
    """Main quantum security validator."""
    
    def __init__(self, policy: QuantumSecurityPolicy = None):
        """Initialize quantum security validator."""
        self.policy = policy or QuantumSecurityPolicy()
        self.circuit_analyzer = CircuitAnalyzer(self.policy)
        self.circuit_sanitizer = CircuitSanitizer(self.policy)
        self.noise_analyzer = NoiseAnalyzer()
        
    def validate_circuit(self, circuit: Dict[str, Any], 
                        circuit_id: str = None) -> CircuitSecurityReport:
        """Validate quantum circuit security."""
        if circuit_id is None:
            circuit_id = hashlib.md5(json.dumps(circuit, sort_keys=True).encode()).hexdigest()[:8]
            
        issues = []
        
        # Policy validation
        issues.extend(self.policy.validate_resource_usage(circuit))
        
        # Structure analysis
        issues.extend(self.circuit_analyzer.analyze_circuit_structure(circuit))
        
        # Entanglement analysis
        issues.extend(self.circuit_analyzer.analyze_entanglement_patterns(circuit))
        
        # Parameter analysis
        issues.extend(self.circuit_analyzer.analyze_parameter_security(circuit))
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(issues)
        
        # Determine if circuit is secure
        is_secure = risk_score < 7 and not any(issue.severity == "critical" for issue in issues)
        
        # Generate sanitized circuit if needed
        sanitized_circuit = None
        if not is_secure:
            sanitized_circuit, modifications = self.circuit_sanitizer.sanitize_circuit(circuit)
            
        return CircuitSecurityReport(
            circuit_id=circuit_id,
            is_secure=is_secure,
            issues=issues,
            risk_score=risk_score,
            sanitized_circuit=sanitized_circuit
        )
        
    def validate_quantum_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate complete quantum job."""
        results = {
            "is_secure": True,
            "circuit_reports": [],
            "overall_risk_score": 0,
            "recommendations": []
        }
        
        circuits = job_data.get("circuits", [])
        
        for i, circuit in enumerate(circuits):
            report = self.validate_circuit(circuit, f"circuit_{i}")
            results["circuit_reports"].append(report)
            
            if not report.is_secure:
                results["is_secure"] = False
                
            results["overall_risk_score"] = max(results["overall_risk_score"], report.risk_score)
            
        # Generate recommendations
        if not results["is_secure"]:
            results["recommendations"].append("Review and fix security issues before execution")
            
        if results["overall_risk_score"] > 5:
            results["recommendations"].append("High risk job - consider additional review")
            
        return results
        
    def analyze_quantum_results(self, results: Dict[str, Any], 
                               backend_name: str) -> List[SecurityIssue]:
        """Analyze quantum execution results for security issues."""
        return self.noise_analyzer.analyze_noise_signature(results, backend_name)
        
    def _calculate_risk_score(self, issues: List[SecurityIssue]) -> int:
        """Calculate overall risk score from issues."""
        severity_scores = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        
        total_score = 0
        for issue in issues:
            total_score += severity_scores.get(issue.severity, 0)
            
        # Normalize to 0-10 scale
        return min(10, total_score)


# Global quantum security validator
_global_quantum_validator: Optional[QuantumSecurityValidator] = None

def get_quantum_security_validator() -> QuantumSecurityValidator:
    """Get global quantum security validator."""
    global _global_quantum_validator
    if _global_quantum_validator is None:
        _global_quantum_validator = QuantumSecurityValidator()
    return _global_quantum_validator