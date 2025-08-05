"""Input validation and data sanitization for quantum MLOps."""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import re
import logging

import numpy as np

from .core import QuantumDevice
from .exceptions import QuantumMLOpsException
from .i18n import translate as _

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    
    is_valid: bool
    error_messages: List[str]
    warnings: List[str]
    sanitized_data: Optional[Any] = None


class QuantumDataValidator:
    """Validator for quantum machine learning data and parameters."""
    
    def __init__(self, strict_mode: bool = True) -> None:
        """Initialize quantum data validator.
        
        Args:
            strict_mode: Enable strict validation rules
        """
        self.strict_mode = strict_mode
    
    def validate_training_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        min_samples: int = 10,
        max_samples: int = 100000,
        max_features: int = 50
    ) -> ValidationResult:
        """Validate quantum training data.
        
        Args:
            X: Training features
            y: Training labels
            min_samples: Minimum number of training samples
            max_samples: Maximum number of training samples
            max_features: Maximum number of features
            
        Returns:
            ValidationResult with sanitized data
        """
        errors = []
        warnings = []
        
        # Check data types
        if not isinstance(X, np.ndarray):
            try:
                X = np.array(X, dtype=float)
                warnings.append("Converted X to numpy array")
            except (ValueError, TypeError):
                errors.append(_("invalid_training_data", field="X"))
        
        if not isinstance(y, np.ndarray):
            try:
                y = np.array(y)
                warnings.append("Converted y to numpy array")
            except (ValueError, TypeError):
                errors.append(_("invalid_training_data", field="y"))
        
        if errors:
            return ValidationResult(False, errors, warnings)
        
        # Check dimensions
        if X.ndim != 2:
            errors.append(f"X must be 2D array, got {X.ndim}D")
        
        if y.ndim != 1:
            if y.ndim == 2 and y.shape[1] == 1:
                y = y.flatten()
                warnings.append("Flattened y from 2D to 1D")
            else:
                errors.append(f"y must be 1D array, got {y.ndim}D")
        
        # Check sample count consistency
        if X.shape[0] != y.shape[0]:
            errors.append(f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}")
        
        # Check sample count limits
        n_samples = X.shape[0]
        if n_samples < min_samples:
            errors.append(f"Too few samples: {n_samples} < {min_samples}")
        elif n_samples > max_samples:
            if self.strict_mode:
                errors.append(f"Too many samples: {n_samples} > {max_samples}")
            else:
                warnings.append(f"Large dataset: {n_samples} samples may be slow")
        
        # Check feature count
        if X.ndim == 2 and X.shape[1] > max_features:
            if self.strict_mode:
                errors.append(f"Too many features: {X.shape[1]} > {max_features}")
            else:
                warnings.append(f"High-dimensional data: {X.shape[1]} features")
        
        # Check for NaN/Inf values
        if np.isnan(X).any():
            if self.strict_mode:
                errors.append("X contains NaN values")
            else:
                # Replace NaN with mean
                X = X.copy()
                nan_mask = np.isnan(X)
                X[nan_mask] = np.nanmean(X, axis=0)[nan_mask[:, 0]]
                warnings.append("Replaced NaN values in X with column means")
        
        if np.isinf(X).any():
            errors.append("X contains infinite values")
        
        if np.isnan(y).any():
            errors.append("y contains NaN values")
        
        if np.isinf(y).any():
            errors.append("y contains infinite values")
        
        # Check data ranges for quantum encoding
        X_range = np.ptp(X, axis=0)  # Peak-to-peak range
        if np.any(X_range > 10):
            warnings.append("Large feature ranges detected, consider normalization")
        
        # Check label distribution
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            errors.append("Labels must have at least 2 unique values")
        elif len(unique_labels) > 10:
            warnings.append(f"Many classes detected: {len(unique_labels)}")
        
        # Binary classification check
        if len(unique_labels) == 2 and not set(unique_labels).issubset({0, 1}):
            warnings.append("Binary labels not in {0, 1} format")
        
        sanitized_data = (X, y) if not errors else None
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            error_messages=errors,
            warnings=warnings,
            sanitized_data=sanitized_data
        )
    
    def validate_quantum_circuit_params(
        self,
        n_qubits: int,
        device: Union[str, QuantumDevice],
        layers: Optional[int] = None,
        **kwargs: Any
    ) -> ValidationResult:
        """Validate quantum circuit parameters.
        
        Args:
            n_qubits: Number of qubits
            device: Quantum backend device
            layers: Number of circuit layers
            **kwargs: Additional circuit parameters
            
        Returns:
            ValidationResult with sanitized parameters
        """
        errors = []
        warnings = []
        sanitized_params = {}
        
        # Validate n_qubits
        if not isinstance(n_qubits, int):
            try:
                n_qubits = int(n_qubits)
                warnings.append("Converted n_qubits to integer")
            except (ValueError, TypeError):
                errors.append(_("invalid_qubits"))
        
        if n_qubits <= 0:
            errors.append(_("invalid_qubits"))
        elif n_qubits > 30:
            if self.strict_mode:
                errors.append(f"Too many qubits for simulation: {n_qubits} > 30")
            else:
                warnings.append(f"Large qubit count may be slow: {n_qubits}")
        elif n_qubits > 20:
            warnings.append(f"High qubit count: {n_qubits}")
        
        sanitized_params['n_qubits'] = n_qubits
        
        # Validate device
        if isinstance(device, str):
            try:
                device = QuantumDevice(device)
                sanitized_params['device'] = device
            except ValueError:
                valid_devices = [d.value for d in QuantumDevice]
                errors.append(f"Invalid device: {device}. Must be one of: {valid_devices}")
        elif isinstance(device, QuantumDevice):
            sanitized_params['device'] = device
        else:
            errors.append("Device must be string or QuantumDevice enum")
        
        # Validate layers
        if layers is not None:
            if not isinstance(layers, int):
                try:
                    layers = int(layers)
                    warnings.append("Converted layers to integer")
                except (ValueError, TypeError):
                    errors.append("Layers must be integer")
            
            if layers <= 0:
                errors.append("Layers must be positive")
            elif layers > 10:
                warnings.append(f"Deep circuit: {layers} layers may have training issues")
            
            sanitized_params['layers'] = layers
        
        # Validate additional parameters
        if 'entanglement' in kwargs:
            entanglement = kwargs['entanglement']
            valid_entanglements = ['linear', 'circular', 'full']
            if entanglement not in valid_entanglements:
                errors.append(f"Invalid entanglement: {entanglement}. Must be one of: {valid_entanglements}")
            else:
                sanitized_params['entanglement'] = entanglement
        
        if 'shots' in kwargs:
            shots = kwargs['shots']
            if not isinstance(shots, int) or shots <= 0:
                errors.append("Shots must be positive integer")
            elif shots > 100000:
                warnings.append(f"High shot count: {shots}")
            else:
                sanitized_params['shots'] = shots
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            error_messages=errors,
            warnings=warnings,
            sanitized_data=sanitized_params
        )
    
    def validate_hyperparameter_search_space(
        self,
        search_space: Dict[str, Any]
    ) -> ValidationResult:
        """Validate hyperparameter search space definition.
        
        Args:
            search_space: Search space dictionary
            
        Returns:
            ValidationResult with sanitized search space
        """
        errors = []
        warnings = []
        sanitized_space = {}
        
        required_params = ['n_qubits', 'learning_rate']
        recommended_params = ['epochs', 'layers']
        
        # Check required parameters
        for param in required_params:
            if param not in search_space:
                if self.strict_mode:
                    errors.append(f"Missing required parameter: {param}")
                else:
                    warnings.append(f"Missing recommended parameter: {param}")
        
        # Check recommended parameters
        for param in recommended_params:
            if param not in search_space:
                warnings.append(f"Consider adding parameter: {param}")
        
        # Validate each parameter
        for param_name, param_config in search_space.items():
            try:
                sanitized_param = self._validate_search_param(param_name, param_config)
                sanitized_space[param_name] = sanitized_param
            except ValueError as e:
                errors.append(f"Invalid search space for {param_name}: {e}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            error_messages=errors,
            warnings=warnings,
            sanitized_data=sanitized_space
        )
    
    def _validate_search_param(self, param_name: str, param_config: Any) -> Any:
        """Validate individual search space parameter."""
        if isinstance(param_config, list):
            # Categorical parameter
            if len(param_config) == 0:
                raise ValueError("Empty categorical list")
            return param_config
        
        elif isinstance(param_config, tuple) and len(param_config) == 2:
            # Continuous parameter range
            low, high = param_config
            if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                raise ValueError("Range bounds must be numeric")
            if low >= high:
                raise ValueError("Range low must be less than high")
            return param_config
        
        elif isinstance(param_config, dict):
            # Structured parameter definition
            if 'type' not in param_config:
                raise ValueError("Missing 'type' in parameter config")
            
            param_type = param_config['type']
            if param_type == 'int':
                if 'low' not in param_config or 'high' not in param_config:
                    raise ValueError("Integer parameter missing 'low' or 'high'")
                low, high = param_config['low'], param_config['high']
                if not isinstance(low, int) or not isinstance(high, int):
                    raise ValueError("Integer parameter bounds must be integers")
                if low >= high:
                    raise ValueError("Integer parameter low must be less than high")
            
            elif param_type == 'log':
                if 'low' not in param_config or 'high' not in param_config:
                    raise ValueError("Log parameter missing 'low' or 'high'")
                low, high = param_config['low'], param_config['high']
                if low <= 0 or high <= 0:
                    raise ValueError("Log parameter bounds must be positive")
                if low >= high:
                    raise ValueError("Log parameter low must be less than high")
            
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
            
            return param_config
        
        else:
            # Fixed parameter value
            return param_config
    
    def validate_circuit_description(
        self,
        circuit: Dict[str, Any]
    ) -> ValidationResult:
        """Validate quantum circuit description.
        
        Args:
            circuit: Circuit description dictionary
            
        Returns:
            ValidationResult with sanitized circuit
        """
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ['gates', 'n_qubits']
        for field in required_fields:
            if field not in circuit:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return ValidationResult(False, errors, warnings)
        
        # Validate n_qubits
        n_qubits = circuit['n_qubits']
        if not isinstance(n_qubits, int) or n_qubits <= 0:
            errors.append("n_qubits must be positive integer")
        
        # Validate gates
        gates = circuit['gates']
        if not isinstance(gates, list):
            errors.append("Gates must be a list")
        else:
            for i, gate in enumerate(gates):
                gate_errors = self._validate_gate(gate, n_qubits, i)
                errors.extend(gate_errors)
        
        # Check circuit depth
        if len(gates) > 1000:
            warnings.append(f"Very deep circuit: {len(gates)} gates")
        elif len(gates) > 100:
            warnings.append(f"Deep circuit: {len(gates)} gates")
        
        # Check for common issues
        if self._has_measurement_in_middle(gates):
            warnings.append("Measurements in middle of circuit")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            error_messages=errors,
            warnings=warnings,
            sanitized_data=circuit if len(errors) == 0 else None
        )
    
    def _validate_gate(self, gate: Dict[str, Any], n_qubits: int, gate_index: int) -> List[str]:
        """Validate individual gate in circuit."""
        errors = []
        
        if not isinstance(gate, dict):
            errors.append(f"Gate {gate_index} must be dictionary")
            return errors
        
        if 'type' not in gate:
            errors.append(f"Gate {gate_index} missing 'type'")
            return errors
        
        gate_type = gate['type']
        valid_gates = {
            'h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 's', 't',
            'cnot', 'cx', 'cy', 'cz', 'swap', 'ccx'
        }
        
        if gate_type not in valid_gates:
            errors.append(f"Unknown gate type: {gate_type}")
        
        # Validate qubit indices
        if gate_type in ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 's', 't']:
            # Single-qubit gates
            if 'qubit' not in gate:
                errors.append(f"Single-qubit gate {gate_type} missing 'qubit'")
            else:
                qubit = gate['qubit']
                if not isinstance(qubit, int) or qubit < 0 or qubit >= n_qubits:
                    errors.append(f"Invalid qubit index: {qubit}")
        
        elif gate_type in ['cnot', 'cx', 'cy', 'cz', 'swap']:
            # Two-qubit gates
            if 'control' not in gate or 'target' not in gate:
                errors.append(f"Two-qubit gate {gate_type} missing control/target")
            else:
                control, target = gate['control'], gate['target']
                if not isinstance(control, int) or control < 0 or control >= n_qubits:
                    errors.append(f"Invalid control qubit: {control}")
                if not isinstance(target, int) or target < 0 or target >= n_qubits:
                    errors.append(f"Invalid target qubit: {target}")
                if control == target:
                    errors.append("Control and target qubits cannot be the same")
        
        elif gate_type == 'ccx':
            # Three-qubit gate
            required_fields = ['control1', 'control2', 'target']
            for field in required_fields:
                if field not in gate:
                    errors.append(f"Toffoli gate missing '{field}'")
        
        # Validate rotation angles
        if gate_type in ['rx', 'ry', 'rz'] and 'angle' in gate:
            angle = gate['angle']
            if not isinstance(angle, (int, float)):
                errors.append(f"Gate angle must be numeric, got {type(angle)}")
            elif abs(angle) > 100:  # Reasonable sanity check
                errors.append(f"Unusually large rotation angle: {angle}")
        
        return errors
    
    def _has_measurement_in_middle(self, gates: List[Dict[str, Any]]) -> bool:
        """Check if there are measurement operations in the middle of the circuit."""
        measurement_types = {'measure', 'measurement'}
        
        for i, gate in enumerate(gates[:-1]):  # Exclude last gate
            if gate.get('type') in measurement_types:
                return True
        
        return False
    
    def sanitize_string_input(self, input_str: str, max_length: int = 1000) -> str:
        """Sanitize string input for security.
        
        Args:
            input_str: Input string to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
        """
        if not isinstance(input_str, str):
            input_str = str(input_str)
        
        # Truncate if too long
        if len(input_str) > max_length:
            input_str = input_str[:max_length]
        
        # Remove potentially dangerous characters
        # Allow alphanumeric, spaces, hyphens, underscores, dots
        sanitized = re.sub(r'[^a-zA-Z0-9\s\-_.]', '', input_str)
        
        # Remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    def validate_file_upload(
        self,
        filename: str,
        file_content: bytes,
        allowed_extensions: List[str] = ['.json', '.npy', '.csv'],
        max_size_mb: int = 100
    ) -> ValidationResult:
        """Validate uploaded file.
        
        Args:
            filename: Name of uploaded file
            file_content: File content as bytes
            allowed_extensions: List of allowed file extensions
            max_size_mb: Maximum file size in MB
            
        Returns:
            ValidationResult for file validation
        """
        errors = []
        warnings = []
        
        # Check file extension
        file_ext = None
        for ext in allowed_extensions:
            if filename.lower().endswith(ext.lower()):
                file_ext = ext
                break
        
        if not file_ext:
            errors.append(f"Invalid file extension. Allowed: {allowed_extensions}")
        
        # Check file size
        file_size_mb = len(file_content) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            errors.append(f"File too large: {file_size_mb:.1f}MB > {max_size_mb}MB")
        elif file_size_mb > max_size_mb * 0.8:
            warnings.append(f"Large file: {file_size_mb:.1f}MB")
        
        # Basic content validation
        try:
            if file_ext == '.json':
                import json
                json.loads(file_content.decode('utf-8'))
            elif file_ext == '.npy':
                import io
                np.load(io.BytesIO(file_content))
        except Exception as e:
            errors.append(f"Invalid file content: {e}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            error_messages=errors,
            warnings=warnings,
            sanitized_data=file_content if len(errors) == 0 else None
        )