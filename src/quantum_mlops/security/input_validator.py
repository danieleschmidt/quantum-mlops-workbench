"""Input validation and sanitization for quantum circuits and parameters."""

import re
import json
import math
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of input validation."""
    
    is_valid: bool
    sanitized_data: Any = None
    errors: List[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class BaseValidator(ABC):
    """Base class for validators."""
    
    @abstractmethod
    def validate(self, data: Any) -> ValidationResult:
        """Validate input data."""
        pass
        
    def sanitize(self, data: Any) -> Any:
        """Sanitize input data."""
        return data


class NumericValidator(BaseValidator):
    """Validator for numeric values."""
    
    def __init__(self, min_value: float = None, max_value: float = None,
                 allow_nan: bool = False, allow_inf: bool = False):
        """Initialize numeric validator."""
        self.min_value = min_value
        self.max_value = max_value
        self.allow_nan = allow_nan
        self.allow_inf = allow_inf
        
    def validate(self, data: Any) -> ValidationResult:
        """Validate numeric data."""
        errors = []
        warnings = []
        
        # Convert to float if possible
        try:
            if isinstance(data, (int, float, np.number)):
                value = float(data)
            elif isinstance(data, str):
                value = float(data)
            else:
                return ValidationResult(False, errors=["Invalid numeric type"])
        except (ValueError, TypeError):
            return ValidationResult(False, errors=["Cannot convert to numeric value"])
            
        # Check for NaN and Inf
        if math.isnan(value):
            if not self.allow_nan:
                errors.append("NaN values not allowed")
            else:
                warnings.append("NaN value detected")
                
        if math.isinf(value):
            if not self.allow_inf:
                errors.append("Infinite values not allowed")
            else:
                warnings.append("Infinite value detected")
                
        # Check range
        if self.min_value is not None and value < self.min_value:
            errors.append(f"Value {value} below minimum {self.min_value}")
            
        if self.max_value is not None and value > self.max_value:
            errors.append(f"Value {value} above maximum {self.max_value}")
            
        is_valid = len(errors) == 0
        sanitized = self.sanitize(value) if is_valid else None
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_data=sanitized,
            errors=errors,
            warnings=warnings
        )
        
    def sanitize(self, value: float) -> float:
        """Sanitize numeric value."""
        # Clamp to range if specified
        if self.min_value is not None:
            value = max(value, self.min_value)
        if self.max_value is not None:
            value = min(value, self.max_value)
        return value


class StringValidator(BaseValidator):
    """Validator for string values."""
    
    def __init__(self, min_length: int = 0, max_length: int = None,
                 pattern: str = None, allowed_chars: str = None,
                 forbidden_patterns: List[str] = None):
        """Initialize string validator."""
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = re.compile(pattern) if pattern else None
        self.allowed_chars = set(allowed_chars) if allowed_chars else None
        self.forbidden_patterns = [re.compile(p) for p in (forbidden_patterns or [])]
        
    def validate(self, data: Any) -> ValidationResult:
        """Validate string data."""
        errors = []
        warnings = []
        
        # Convert to string
        if not isinstance(data, str):
            try:
                value = str(data)
                warnings.append("Converted to string")
            except Exception:
                return ValidationResult(False, errors=["Cannot convert to string"])
        else:
            value = data
            
        # Check length
        if len(value) < self.min_length:
            errors.append(f"String length {len(value)} below minimum {self.min_length}")
            
        if self.max_length is not None and len(value) > self.max_length:
            errors.append(f"String length {len(value)} above maximum {self.max_length}")
            
        # Check pattern
        if self.pattern and not self.pattern.match(value):
            errors.append("String does not match required pattern")
            
        # Check allowed characters
        if self.allowed_chars:
            invalid_chars = set(value) - self.allowed_chars
            if invalid_chars:
                errors.append(f"Invalid characters: {invalid_chars}")
                
        # Check forbidden patterns
        for pattern in self.forbidden_patterns:
            if pattern.search(value):
                errors.append(f"String contains forbidden pattern: {pattern.pattern}")
                
        is_valid = len(errors) == 0
        sanitized = self.sanitize(value) if is_valid else None
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_data=sanitized,
            errors=errors,
            warnings=warnings
        )
        
    def sanitize(self, value: str) -> str:
        """Sanitize string value."""
        # Remove forbidden characters if allowed_chars specified
        if self.allowed_chars:
            value = ''.join(c for c in value if c in self.allowed_chars)
            
        # Truncate if too long
        if self.max_length is not None:
            value = value[:self.max_length]
            
        return value


class ListValidator(BaseValidator):
    """Validator for list/array values."""
    
    def __init__(self, min_length: int = 0, max_length: int = None,
                 item_validator: BaseValidator = None,
                 unique_items: bool = False):
        """Initialize list validator."""
        self.min_length = min_length
        self.max_length = max_length
        self.item_validator = item_validator
        self.unique_items = unique_items
        
    def validate(self, data: Any) -> ValidationResult:
        """Validate list data."""
        errors = []
        warnings = []
        
        # Convert to list if possible
        if isinstance(data, (list, tuple, np.ndarray)):
            items = list(data)
        else:
            return ValidationResult(False, errors=["Invalid list type"])
            
        # Check length
        if len(items) < self.min_length:
            errors.append(f"List length {len(items)} below minimum {self.min_length}")
            
        if self.max_length is not None and len(items) > self.max_length:
            errors.append(f"List length {len(items)} above maximum {self.max_length}")
            
        # Check uniqueness
        if self.unique_items and len(items) != len(set(items)):
            errors.append("List contains duplicate items")
            
        # Validate items
        sanitized_items = []
        if self.item_validator:
            for i, item in enumerate(items):
                result = self.item_validator.validate(item)
                if not result.is_valid:
                    errors.extend([f"Item {i}: {error}" for error in result.errors])
                else:
                    sanitized_items.append(result.sanitized_data)
                    warnings.extend([f"Item {i}: {warning}" for warning in result.warnings])
        else:
            sanitized_items = items
            
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_data=sanitized_items if is_valid else None,
            errors=errors,
            warnings=warnings
        )


class DictValidator(BaseValidator):
    """Validator for dictionary values."""
    
    def __init__(self, required_keys: List[str] = None,
                 optional_keys: List[str] = None,
                 key_validators: Dict[str, BaseValidator] = None,
                 allow_extra_keys: bool = True):
        """Initialize dictionary validator."""
        self.required_keys = set(required_keys or [])
        self.optional_keys = set(optional_keys or [])
        self.allowed_keys = self.required_keys | self.optional_keys
        self.key_validators = key_validators or {}
        self.allow_extra_keys = allow_extra_keys
        
    def validate(self, data: Any) -> ValidationResult:
        """Validate dictionary data."""
        errors = []
        warnings = []
        
        if not isinstance(data, dict):
            return ValidationResult(False, errors=["Invalid dictionary type"])
            
        # Check required keys
        missing_keys = self.required_keys - set(data.keys())
        if missing_keys:
            errors.append(f"Missing required keys: {missing_keys}")
            
        # Check extra keys
        if not self.allow_extra_keys:
            extra_keys = set(data.keys()) - self.allowed_keys
            if extra_keys:
                errors.append(f"Unexpected keys: {extra_keys}")
                
        # Validate key values
        sanitized_data = {}
        for key, value in data.items():
            if key in self.key_validators:
                result = self.key_validators[key].validate(value)
                if not result.is_valid:
                    errors.extend([f"Key '{key}': {error}" for error in result.errors])
                else:
                    sanitized_data[key] = result.sanitized_data
                    warnings.extend([f"Key '{key}': {warning}" for warning in result.warnings])
            else:
                sanitized_data[key] = value
                
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_data=sanitized_data if is_valid else None,
            errors=errors,
            warnings=warnings
        )


class QuantumParameterValidator(BaseValidator):
    """Validator for quantum circuit parameters."""
    
    def __init__(self):
        """Initialize quantum parameter validator."""
        # Angle parameters should be in range [-2π, 2π]
        self.angle_validator = NumericValidator(
            min_value=-2*math.pi, 
            max_value=2*math.pi
        )
        
    def validate(self, data: Any) -> ValidationResult:
        """Validate quantum parameter."""
        # Handle different parameter types
        if isinstance(data, (int, float, complex)):
            if isinstance(data, complex):
                # Validate complex parameters (e.g., for unitary matrices)
                return self._validate_complex_parameter(data)
            else:
                # Validate angle parameters
                return self.angle_validator.validate(data)
                
        elif isinstance(data, (list, np.ndarray)):
            # Parameter list/array
            return self._validate_parameter_array(data)
            
        else:
            return ValidationResult(False, errors=["Invalid parameter type"])
            
    def _validate_complex_parameter(self, value: complex) -> ValidationResult:
        """Validate complex parameter."""
        errors = []
        warnings = []
        
        # Check magnitude (should be reasonable for quantum gates)
        magnitude = abs(value)
        if magnitude > 10:
            warnings.append(f"Large parameter magnitude: {magnitude}")
            
        if math.isnan(value.real) or math.isnan(value.imag):
            errors.append("Complex parameter contains NaN")
            
        if math.isinf(value.real) or math.isinf(value.imag):
            errors.append("Complex parameter contains Inf")
            
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_data=value if is_valid else None,
            errors=errors,
            warnings=warnings
        )
        
    def _validate_parameter_array(self, data: List[Any]) -> ValidationResult:
        """Validate parameter array."""
        validator = ListValidator(
            min_length=1,
            max_length=1000,  # Reasonable limit
            item_validator=NumericValidator(
                min_value=-4*math.pi,
                max_value=4*math.pi
            )
        )
        return validator.validate(data)


class QuantumGateValidator(BaseValidator):
    """Validator for quantum gates."""
    
    def __init__(self):
        """Initialize quantum gate validator."""
        # Valid gate types
        self.valid_gates = {
            # Single-qubit gates
            'h', 'hadamard', 'x', 'pauli_x', 'y', 'pauli_y', 'z', 'pauli_z',
            's', 'sdg', 't', 'tdg', 'rx', 'ry', 'rz', 'u1', 'u2', 'u3',
            'id', 'identity',
            
            # Two-qubit gates
            'cnot', 'cx', 'cy', 'cz', 'ch', 'crx', 'cry', 'crz', 'cu1', 'cu2', 'cu3',
            'swap', 'iswap', 'cswap', 'ccx', 'toffoli',
            
            # Three-qubit gates
            'ccx', 'cswap', 'fredkin',
            
            # Measurement
            'measure', 'measurement'
        }
        
        self.parameter_validator = QuantumParameterValidator()
        
    def validate(self, data: Any) -> ValidationResult:
        """Validate quantum gate."""
        errors = []
        warnings = []
        
        if not isinstance(data, dict):
            return ValidationResult(False, errors=["Gate must be a dictionary"])
            
        # Check required fields
        if 'type' not in data:
            errors.append("Gate missing 'type' field")
        else:
            gate_type = data['type'].lower()
            if gate_type not in self.valid_gates:
                errors.append(f"Invalid gate type: {gate_type}")
                
        # Check wires/qubits
        wires = data.get('wires', data.get('qubit', data.get('qubits')))
        if wires is None:
            errors.append("Gate missing qubit specification")
        else:
            wire_result = self._validate_wires(wires)
            if not wire_result.is_valid:
                errors.extend(wire_result.errors)
            else:
                data['wires'] = wire_result.sanitized_data
                
        # Check parameters
        if 'angle' in data or 'angles' in data or 'parameters' in data:
            param_data = data.get('angle') or data.get('angles') or data.get('parameters')
            param_result = self.parameter_validator.validate(param_data)
            if not param_result.is_valid:
                errors.extend([f"Parameter error: {error}" for error in param_result.errors])
            else:
                warnings.extend([f"Parameter warning: {warning}" for warning in param_result.warnings])
                
        # Gate-specific validation
        gate_type = data.get('type', '').lower()
        if gate_type in ['cnot', 'cx', 'cy', 'cz', 'swap']:
            # Two-qubit gates need exactly 2 qubits
            wires = data.get('wires', [])
            if len(wires) != 2:
                errors.append(f"{gate_type} gate requires exactly 2 qubits")
                
        elif gate_type in ['ccx', 'toffoli', 'cswap', 'fredkin']:
            # Three-qubit gates need exactly 3 qubits
            wires = data.get('wires', [])
            if len(wires) != 3:
                errors.append(f"{gate_type} gate requires exactly 3 qubits")
                
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_data=data if is_valid else None,
            errors=errors,
            warnings=warnings
        )
        
    def _validate_wires(self, wires: Any) -> ValidationResult:
        """Validate wire/qubit specification."""
        if isinstance(wires, int):
            if wires < 0:
                return ValidationResult(False, errors=["Negative qubit index"])
            if wires > 1000:  # Reasonable limit
                return ValidationResult(False, errors=["Qubit index too large"])
            return ValidationResult(True, sanitized_data=[wires])
            
        elif isinstance(wires, (list, tuple)):
            errors = []
            sanitized_wires = []
            
            for wire in wires:
                if not isinstance(wire, int):
                    errors.append(f"Invalid wire type: {type(wire)}")
                elif wire < 0:
                    errors.append(f"Negative qubit index: {wire}")
                elif wire > 1000:
                    errors.append(f"Qubit index too large: {wire}")
                else:
                    sanitized_wires.append(wire)
                    
            return ValidationResult(
                is_valid=len(errors) == 0,
                sanitized_data=sanitized_wires if len(errors) == 0 else None,
                errors=errors
            )
            
        else:
            return ValidationResult(False, errors=["Invalid wire specification"])


class QuantumCircuitValidator(BaseValidator):
    """Validator for quantum circuits."""
    
    def __init__(self, max_qubits: int = 100, max_gates: int = 10000):
        """Initialize quantum circuit validator."""
        self.max_qubits = max_qubits
        self.max_gates = max_gates
        self.gate_validator = QuantumGateValidator()
        
    def validate(self, data: Any) -> ValidationResult:
        """Validate quantum circuit."""
        errors = []
        warnings = []
        
        if not isinstance(data, dict):
            return ValidationResult(False, errors=["Circuit must be a dictionary"])
            
        # Check basic structure
        if 'gates' not in data:
            errors.append("Circuit missing 'gates' field")
        else:
            gates = data['gates']
            if not isinstance(gates, list):
                errors.append("Gates must be a list")
            elif len(gates) > self.max_gates:
                errors.append(f"Too many gates: {len(gates)} > {self.max_gates}")
                
        # Check qubit count
        n_qubits = data.get('n_qubits', 0)
        if not isinstance(n_qubits, int) or n_qubits <= 0:
            errors.append("Invalid qubit count")
        elif n_qubits > self.max_qubits:
            errors.append(f"Too many qubits: {n_qubits} > {self.max_qubits}")
            
        # Validate individual gates
        if 'gates' in data and isinstance(data['gates'], list):
            sanitized_gates = []
            max_qubit_used = -1
            
            for i, gate in enumerate(data['gates']):
                gate_result = self.gate_validator.validate(gate)
                if not gate_result.is_valid:
                    errors.extend([f"Gate {i}: {error}" for error in gate_result.errors])
                else:
                    sanitized_gates.append(gate_result.sanitized_data)
                    warnings.extend([f"Gate {i}: {warning}" for warning in gate_result.warnings])
                    
                    # Track maximum qubit used
                    wires = gate_result.sanitized_data.get('wires', [])
                    if wires:
                        max_qubit_used = max(max_qubit_used, max(wires))
                        
            # Check consistency between n_qubits and actual usage
            if max_qubit_used >= 0 and max_qubit_used >= n_qubits:
                warnings.append(f"Gate uses qubit {max_qubit_used} but circuit has only {n_qubits} qubits")
                
        # Check for suspicious patterns
        if 'gates' in data:
            suspicious_patterns = self._detect_suspicious_patterns(data['gates'])
            warnings.extend(suspicious_patterns)
            
        is_valid = len(errors) == 0
        sanitized_data = data.copy() if is_valid else None
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_data=sanitized_data,
            errors=errors,
            warnings=warnings
        )
        
    def _detect_suspicious_patterns(self, gates: List[Dict]) -> List[str]:
        """Detect suspicious patterns in circuit."""
        warnings = []
        
        # Check for excessive repetition
        gate_types = [gate.get('type', '').lower() for gate in gates]
        
        # Same gate repeated many times
        for gate_type in set(gate_types):
            count = gate_types.count(gate_type)
            if count > len(gates) * 0.8:  # More than 80% of gates are the same
                warnings.append(f"Excessive repetition of {gate_type} gate ({count} times)")
                
        # Check for very deep circuits
        if len(gates) > 1000:
            warnings.append(f"Very deep circuit with {len(gates)} gates")
            
        # Check for patterns that might indicate information extraction
        measurement_count = sum(1 for g in gate_types if 'measure' in g)
        if measurement_count > len(set(gate_types)) * 2:
            warnings.append("Excessive measurements detected")
            
        return warnings


class QuantumInputValidator:
    """Main validator for quantum inputs."""
    
    def __init__(self, max_qubits: int = 100, max_gates: int = 10000):
        """Initialize quantum input validator."""
        self.circuit_validator = QuantumCircuitValidator(max_qubits, max_gates)
        self.parameter_validator = QuantumParameterValidator()
        
    def validate_circuit(self, circuit: Dict[str, Any]) -> ValidationResult:
        """Validate quantum circuit."""
        return self.circuit_validator.validate(circuit)
        
    def validate_parameters(self, parameters: Any) -> ValidationResult:
        """Validate quantum parameters."""
        return self.parameter_validator.validate(parameters)
        
    def validate_job_input(self, job_data: Dict[str, Any]) -> ValidationResult:
        """Validate complete job input."""
        errors = []
        warnings = []
        sanitized_data = {}
        
        # Validate circuits
        if 'circuits' in job_data:
            circuits = job_data['circuits']
            if not isinstance(circuits, list):
                errors.append("Circuits must be a list")
            else:
                sanitized_circuits = []
                for i, circuit in enumerate(circuits):
                    result = self.validate_circuit(circuit)
                    if not result.is_valid:
                        errors.extend([f"Circuit {i}: {error}" for error in result.errors])
                    else:
                        sanitized_circuits.append(result.sanitized_data)
                        warnings.extend([f"Circuit {i}: {warning}" for warning in result.warnings])
                        
                sanitized_data['circuits'] = sanitized_circuits
                
        # Validate shots
        if 'shots' in job_data:
            shots = job_data['shots']
            shots_validator = NumericValidator(min_value=1, max_value=100000)
            result = shots_validator.validate(shots)
            if not result.is_valid:
                errors.extend([f"Shots: {error}" for error in result.errors])
            else:
                sanitized_data['shots'] = result.sanitized_data
                
        # Validate backend
        if 'backend' in job_data:
            backend = job_data['backend']
            backend_validator = StringValidator(
                min_length=1,
                max_length=100,
                pattern=r'^[a-zA-Z0-9_.-]+$'
            )
            result = backend_validator.validate(backend)
            if not result.is_valid:
                errors.extend([f"Backend: {error}" for error in result.errors])
            else:
                sanitized_data['backend'] = result.sanitized_data
                
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_data=sanitized_data if is_valid else None,
            errors=errors,
            warnings=warnings
        )


class InputValidator:
    """General input validator with quantum-specific extensions."""
    
    def __init__(self):
        """Initialize input validator."""
        self.quantum_validator = QuantumInputValidator()
        
    def validate_string(self, data: Any, **kwargs) -> ValidationResult:
        """Validate string input."""
        validator = StringValidator(**kwargs)
        return validator.validate(data)
        
    def validate_numeric(self, data: Any, **kwargs) -> ValidationResult:
        """Validate numeric input."""
        validator = NumericValidator(**kwargs)
        return validator.validate(data)
        
    def validate_list(self, data: Any, **kwargs) -> ValidationResult:
        """Validate list input."""
        validator = ListValidator(**kwargs)
        return validator.validate(data)
        
    def validate_dict(self, data: Any, **kwargs) -> ValidationResult:
        """Validate dictionary input."""
        validator = DictValidator(**kwargs)
        return validator.validate(data)
        
    def validate_quantum_circuit(self, data: Any) -> ValidationResult:
        """Validate quantum circuit."""
        return self.quantum_validator.validate_circuit(data)
        
    def validate_quantum_job(self, data: Any) -> ValidationResult:
        """Validate quantum job input."""
        return self.quantum_validator.validate_job_input(data)


# Global input validator
_global_input_validator: Optional[InputValidator] = None

def get_input_validator() -> InputValidator:
    """Get global input validator."""
    global _global_input_validator
    if _global_input_validator is None:
        _global_input_validator = InputValidator()
    return _global_input_validator