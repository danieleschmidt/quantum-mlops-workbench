"""Tests for quantum security validation and circuit sanitization."""

import pytest
import numpy as np
from quantum_mlops.security.quantum_security import (
    QuantumSecurityPolicy, CircuitAnalyzer, CircuitSanitizer,
    QuantumSecurityValidator, ParameterSanitizer, SecurityIssue
)


class TestQuantumSecurityPolicy:
    """Test quantum security policy."""
    
    def test_default_policy(self):
        """Test default policy creation."""
        policy = QuantumSecurityPolicy()
        
        assert policy.config["max_qubits"] > 0
        assert policy.config["max_gates"] > 0
        assert "allowed_gate_types" in policy.config
        assert "h" in policy.config["allowed_gate_types"]
        assert "cnot" in policy.config["allowed_gate_types"]
        
    def test_custom_policy(self):
        """Test custom policy configuration."""
        custom_config = {
            "max_qubits": 20,
            "max_gates": 1000,
            "allowed_gate_types": {"h", "x", "cnot"}
        }
        
        policy = QuantumSecurityPolicy(custom_config)
        
        assert policy.config["max_qubits"] == 20
        assert policy.config["max_gates"] == 1000
        assert len(policy.config["allowed_gate_types"]) == 3
        
    def test_gate_allowed_check(self):
        """Test gate allowance checking."""
        policy = QuantumSecurityPolicy()
        
        assert policy.is_gate_allowed("h") is True
        assert policy.is_gate_allowed("CNOT") is True  # Case insensitive
        assert policy.is_gate_allowed("unknown_gate") is False
        
    def test_resource_validation(self):
        """Test resource usage validation."""
        policy = QuantumSecurityPolicy()
        
        # Valid circuit
        valid_circuit = {
            "n_qubits": 5,
            "gates": [{"type": "h", "qubit": 0}]
        }
        
        issues = policy.validate_resource_usage(valid_circuit)
        assert len(issues) == 0
        
        # Circuit with too many qubits
        invalid_circuit = {
            "n_qubits": policy.config["max_qubits"] + 1,
            "gates": []
        }
        
        issues = policy.validate_resource_usage(invalid_circuit)
        assert len(issues) > 0
        assert any("qubits" in issue.description for issue in issues)


class TestCircuitAnalyzer:
    """Test circuit security analysis."""
    
    @pytest.fixture
    def analyzer(self):
        """Create circuit analyzer."""
        return CircuitAnalyzer()
        
    def test_valid_circuit_analysis(self, analyzer):
        """Test analysis of valid circuit."""
        circuit = {
            "n_qubits": 3,
            "gates": [
                {"type": "h", "qubit": 0},
                {"type": "cnot", "wires": [0, 1]},
                {"type": "measure", "qubit": 2}
            ]
        }
        
        issues = analyzer.analyze_circuit_structure(circuit)
        # Should have minimal or no issues for a reasonable circuit
        critical_issues = [i for i in issues if i.severity == "critical"]
        assert len(critical_issues) == 0
        
    def test_repetitive_gate_detection(self, analyzer):
        """Test detection of repetitive gate patterns."""
        # Circuit with excessive repetition
        gates = [{"type": "x", "qubit": 0} for _ in range(100)]
        circuit = {
            "n_qubits": 1,
            "gates": gates
        }
        
        issues = analyzer.analyze_circuit_structure(circuit)
        repetition_issues = [i for i in issues if "repetition" in i.description.lower()]
        assert len(repetition_issues) > 0
        
    def test_forbidden_gate_detection(self, analyzer):
        """Test detection of forbidden gates."""
        circuit = {
            "n_qubits": 2,
            "gates": [
                {"type": "h", "qubit": 0},
                {"type": "forbidden_gate", "qubit": 1}
            ]
        }
        
        issues = analyzer.analyze_circuit_structure(circuit)
        forbidden_issues = [i for i in issues if i.issue_type == "forbidden_gate"]
        assert len(forbidden_issues) > 0
        
    def test_excessive_measurements_detection(self, analyzer):
        """Test detection of excessive measurements."""
        # Create circuit with too many measurements
        gates = [{"type": "measure", "qubit": i % 2} for i in range(10)]
        circuit = {
            "n_qubits": 2,
            "gates": gates
        }
        
        issues = analyzer.analyze_circuit_structure(circuit)
        measurement_issues = [i for i in issues if "measure" in i.description.lower()]
        assert len(measurement_issues) > 0
        
    def test_entanglement_analysis(self, analyzer):
        """Test entanglement pattern analysis."""
        # Circuit with reasonable entanglement
        circuit = {
            "n_qubits": 4,
            "gates": [
                {"type": "cnot", "wires": [0, 1]},
                {"type": "cnot", "wires": [1, 2]},
                {"type": "cnot", "wires": [2, 3]}
            ]
        }
        
        issues = analyzer.analyze_entanglement_patterns(circuit)
        # Should not flag reasonable entanglement
        excessive_issues = [i for i in issues if "excessive" in i.description.lower()]
        assert len(excessive_issues) == 0
        
    def test_hub_qubit_detection(self, analyzer):
        """Test detection of hub qubit patterns."""
        # Create star pattern (potential fingerprinting)
        gates = [{"type": "cnot", "wires": [0, i+1]} for i in range(8)]
        circuit = {
            "n_qubits": 9,
            "gates": gates
        }
        
        issues = analyzer.analyze_entanglement_patterns(circuit)
        hub_issues = [i for i in issues if "hub" in i.issue_type]
        assert len(hub_issues) > 0
        
    def test_parameter_analysis(self, analyzer):
        """Test parameter security analysis."""
        circuit = {
            "n_qubits": 2,
            "gates": [
                {"type": "rx", "qubit": 0, "angle": np.pi/2},
                {"type": "ry", "qubit": 1, "angle": 2*np.pi},
                {"type": "rz", "qubit": 0, "parameters": [0.1, 0.2, 0.3]}
            ]
        }
        
        issues = analyzer.analyze_parameter_security(circuit)
        # Should not flag reasonable parameters
        range_issues = [i for i in issues if "range" in i.issue_type]
        assert len(range_issues) == 0
        
    def test_extreme_parameter_detection(self, analyzer):
        """Test detection of extreme parameters."""
        circuit = {
            "n_qubits": 1,
            "gates": [
                {"type": "rx", "qubit": 0, "angle": 100*np.pi}  # Extreme angle
            ]
        }
        
        issues = analyzer.analyze_parameter_security(circuit)
        range_issues = [i for i in issues if "range" in i.issue_type]
        assert len(range_issues) > 0


class TestParameterSanitizer:
    """Test parameter sanitization."""
    
    @pytest.fixture
    def sanitizer(self):
        """Create parameter sanitizer."""
        return ParameterSanitizer(precision_digits=6)
        
    def test_float_sanitization(self, sanitizer):
        """Test floating point parameter sanitization."""
        precise_param = 1.23456789012345
        sanitized = sanitizer.sanitize_parameter(precise_param)
        
        assert sanitized == 1.234568  # Rounded to 6 digits
        
    def test_complex_sanitization(self, sanitizer):
        """Test complex parameter sanitization."""
        complex_param = complex(1.23456789, 2.98765432)
        sanitized = sanitizer.sanitize_parameter(complex_param)
        
        assert sanitized.real == 1.234568
        assert sanitized.imag == 2.987654
        
    def test_list_sanitization(self, sanitizer):
        """Test parameter list sanitization."""
        param_list = [1.23456789, 2.98765432, 3.14159265]
        sanitized = sanitizer.sanitize_parameters(param_list)
        
        assert len(sanitized) == 3
        assert sanitized[0] == 1.234568
        assert sanitized[2] == 3.141593
        
    def test_nested_structure_sanitization(self, sanitizer):
        """Test sanitization of nested parameter structures."""
        nested_params = {
            "angles": [1.23456789, 2.98765432],
            "rotation": {"x": 3.14159265, "y": 2.71828182}
        }
        
        sanitized = sanitizer.sanitize_parameters(nested_params)
        
        assert sanitized["angles"][0] == 1.234568
        assert sanitized["rotation"]["x"] == 3.141593
        assert sanitized["rotation"]["y"] == 2.718282


class TestCircuitSanitizer:
    """Test circuit sanitization."""
    
    @pytest.fixture
    def sanitizer(self):
        """Create circuit sanitizer."""
        return CircuitSanitizer()
        
    def test_basic_circuit_sanitization(self, sanitizer):
        """Test basic circuit sanitization."""
        circuit = {
            "n_qubits": 3,
            "gates": [
                {"type": "h", "qubit": 0},
                {"type": "rx", "qubit": 1, "angle": 1.23456789},
                {"type": "cnot", "wires": [0, 1]}
            ]
        }
        
        sanitized, modifications = sanitizer.sanitize_circuit(circuit)
        
        assert sanitized["n_qubits"] == 3
        assert len(sanitized["gates"]) == 3
        # Check parameter was sanitized
        rx_gate = sanitized["gates"][1]
        assert abs(rx_gate["angle"] - 1.234568) < 1e-6
        
    def test_forbidden_gate_removal(self, sanitizer):
        """Test removal of forbidden gates."""
        circuit = {
            "n_qubits": 2,
            "gates": [
                {"type": "h", "qubit": 0},
                {"type": "unknown_gate", "qubit": 1},  # Should be removed
                {"type": "cnot", "wires": [0, 1]}
            ]
        }
        
        sanitized, modifications = sanitizer.sanitize_circuit(circuit)
        
        assert len(sanitized["gates"]) == 2  # One gate removed
        assert any("Removed forbidden gate" in mod for mod in modifications)
        
    def test_qubit_limit_enforcement(self, sanitizer):
        """Test qubit count limiting."""
        max_qubits = sanitizer.policy.config["max_qubits"]
        circuit = {
            "n_qubits": max_qubits + 10,
            "gates": []
        }
        
        sanitized, modifications = sanitizer.sanitize_circuit(circuit)
        
        assert sanitized["n_qubits"] == max_qubits
        assert any("Limited qubit count" in mod for mod in modifications)
        
    def test_gate_count_limiting(self, sanitizer):
        """Test gate count limiting."""
        max_gates = sanitizer.policy.config["max_gates"]
        # Create circuit with too many gates
        gates = [{"type": "x", "qubit": 0} for _ in range(max_gates + 100)]
        circuit = {
            "n_qubits": 1,
            "gates": gates
        }
        
        sanitized, modifications = sanitizer.sanitize_circuit(circuit)
        
        assert len(sanitized["gates"]) == max_gates
        assert any("Limited gate count" in mod for mod in modifications)


class TestQuantumSecurityValidator:
    """Test main quantum security validator."""
    
    @pytest.fixture
    def validator(self):
        """Create quantum security validator."""
        return QuantumSecurityValidator()
        
    def test_secure_circuit_validation(self, validator):
        """Test validation of secure circuit."""
        circuit = {
            "n_qubits": 3,
            "gates": [
                {"type": "h", "qubit": 0},
                {"type": "cnot", "wires": [0, 1]},
                {"type": "rx", "qubit": 2, "angle": np.pi/4}
            ]
        }
        
        report = validator.validate_circuit(circuit)
        
        assert report.circuit_id is not None
        assert report.risk_score < 7  # Should be low risk
        # May have minor issues but should be generally secure
        critical_issues = [i for i in report.issues if i.severity == "critical"]
        assert len(critical_issues) == 0
        
    def test_insecure_circuit_validation(self, validator):
        """Test validation of insecure circuit."""
        # Create problematic circuit
        circuit = {
            "n_qubits": 200,  # Too many qubits
            "gates": [{"type": "forbidden_gate", "qubit": 0}] * 1000  # Forbidden gates
        }
        
        report = validator.validate_circuit(circuit)
        
        assert report.is_secure is False
        assert report.risk_score >= 7
        assert len(report.issues) > 0
        assert report.sanitized_circuit is not None
        
    def test_quantum_job_validation(self, validator):
        """Test validation of complete quantum job."""
        job_data = {
            "circuits": [
                {
                    "n_qubits": 2,
                    "gates": [{"type": "h", "qubit": 0}, {"type": "cnot", "wires": [0, 1]}]
                },
                {
                    "n_qubits": 3,
                    "gates": [{"type": "x", "qubit": 0}, {"type": "measure", "qubit": 0}]
                }
            ]
        }
        
        results = validator.validate_quantum_job(job_data)
        
        assert "is_secure" in results
        assert "circuit_reports" in results
        assert "overall_risk_score" in results
        assert len(results["circuit_reports"]) == 2
        
    def test_risk_score_calculation(self, validator):
        """Test risk score calculation."""
        # Low risk issues
        low_risk_issues = [
            SecurityIssue("low", "test", "Low risk issue", "Fix it", "component")
        ]
        
        # High risk issues  
        high_risk_issues = [
            SecurityIssue("critical", "test", "Critical issue", "Fix immediately", "component"),
            SecurityIssue("high", "test", "High risk issue", "Fix soon", "component")
        ]
        
        low_score = validator._calculate_risk_score(low_risk_issues)
        high_score = validator._calculate_risk_score(high_risk_issues)
        
        assert low_score < high_score
        assert low_score <= 10
        assert high_score <= 10


class TestQuantumSecurityIntegration:
    """Integration tests for quantum security system."""
    
    def test_end_to_end_circuit_processing(self):
        """Test end-to-end circuit security processing."""
        validator = QuantumSecurityValidator()
        
        # Create a circuit with various issues
        circuit = {
            "n_qubits": 5,
            "gates": [
                {"type": "h", "qubit": 0},
                {"type": "rx", "qubit": 1, "angle": 1.23456789012},  # High precision
                {"type": "cnot", "wires": [0, 1]},
                {"type": "measure", "qubit": 0},
                {"type": "measure", "qubit": 1},
                {"type": "measure", "qubit": 2},  # Excessive measurements
            ]
        }
        
        # Validate circuit
        report = validator.validate_circuit(circuit)
        
        # Should detect issues but not be critical
        assert len(report.issues) > 0
        assert report.risk_score > 0
        
        # If not secure, should provide sanitized version
        if not report.is_secure:
            assert report.sanitized_circuit is not None
            
            # Sanitized circuit should be more secure
            sanitized_report = validator.validate_circuit(report.sanitized_circuit)
            assert sanitized_report.risk_score <= report.risk_score
            
    def test_policy_customization_effect(self):
        """Test effect of custom security policies."""
        # Strict policy
        strict_policy = QuantumSecurityPolicy({
            "max_qubits": 2,
            "max_gates": 5,
            "allowed_gate_types": {"h", "x"}
        })
        
        # Lenient policy
        lenient_policy = QuantumSecurityPolicy({
            "max_qubits": 100,
            "max_gates": 10000,
            "allowed_gate_types": {"h", "x", "y", "z", "cnot", "rx", "ry", "rz"}
        })
        
        circuit = {
            "n_qubits": 3,
            "gates": [
                {"type": "h", "qubit": 0},
                {"type": "cnot", "wires": [0, 1]},
                {"type": "rx", "qubit": 2, "angle": np.pi}
            ]
        }
        
        strict_validator = QuantumSecurityValidator(strict_policy)
        lenient_validator = QuantumSecurityValidator(lenient_policy)
        
        strict_report = strict_validator.validate_circuit(circuit)
        lenient_report = lenient_validator.validate_circuit(circuit)
        
        # Strict policy should find more issues
        assert len(strict_report.issues) >= len(lenient_report.issues)
        assert strict_report.risk_score >= lenient_report.risk_score