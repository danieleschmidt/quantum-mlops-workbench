# Quantum Backend Integration

This document describes the quantum backend integration system implemented for the Quantum MLOps Workbench, providing real quantum computing backend support through PennyLane, Qiskit, and AWS Braket.

## Overview

The quantum backend system provides a unified interface for executing quantum circuits across multiple quantum computing platforms with automatic fallback mechanisms, error handling, and optimization.

### Supported Backends

1. **PennyLane** - Local simulators and hardware access
2. **Qiskit** - IBM Quantum simulators and hardware 
3. **AWS Braket** - Cloud quantum computing services

## Architecture

### Core Components

```
quantum_mlops/backends/
├── __init__.py              # Public API exports
├── base.py                  # Abstract base classes
├── pennylane_backend.py     # PennyLane implementation
├── qiskit_backend.py        # Qiskit implementation  
├── braket_backend.py        # AWS Braket implementation
├── backend_manager.py       # Backend management and fallback
└── quantum_executor.py      # High-level execution interface
```

### Key Classes

- **`QuantumBackend`** - Abstract base for all backends
- **`BackendManager`** - Manages backend selection and fallback
- **`QuantumExecutor`** - High-level execution interface
- **`QuantumMLPipeline`** - Enhanced with real backend support

## Quick Start

### Basic Usage

```python
from quantum_mlops.core import QuantumMLPipeline, QuantumDevice
from quantum_mlops.backends import QuantumExecutor

# Create ML pipeline with real backend
def my_circuit():
    return 0.5

pipeline = QuantumMLPipeline(
    my_circuit, 
    n_qubits=4,
    device=QuantumDevice.SIMULATOR,
    shots=1024
)

# Train model (will use real quantum backends if available)
X_train = np.random.random((100, 4))
y_train = np.random.randint(0, 2, 100)

model = pipeline.train(X_train, y_train, epochs=50)
```

### Direct Backend Usage

```python
from quantum_mlops.backends import QuantumExecutor

executor = QuantumExecutor()

# Define quantum circuit
circuit = {
    "gates": [
        {"type": "h", "qubit": 0},
        {"type": "cx", "control": 0, "target": 1},
        {"type": "ry", "qubit": 0, "angle": np.pi/4}
    ],
    "n_qubits": 2
}

# Execute on best available backend
result = executor.execute(circuit, shots=1024)
print(f"Expectation value: {result.expectation_value}")
```

## Backend Configuration

### PennyLane Backend

```python
from quantum_mlops.backends import PennyLaneBackend

# Local simulator
backend = PennyLaneBackend("default.qubit", wires=8, shots=1024)

# Lightning simulator (faster)
backend = PennyLaneBackend("lightning.qubit", wires=20, shots=None)  # Analytic
```

### Qiskit Backend

```python
from quantum_mlops.backends import QiskitBackend

# Local Aer simulator
backend = QiskitBackend("qasm_simulator", shots=1024)

# IBM Quantum (requires credentials)
backend = QiskitBackend(
    "ibmq_qasm_simulator", 
    use_ibm_runtime=True,
    ibm_token="your_token_here"
)
```

### AWS Braket Backend

```python
from quantum_mlops.backends import BraketBackend

# Local simulator
backend = BraketBackend("local:braket/braket.local.qubit")

# AWS cloud device (requires AWS credentials)
backend = BraketBackend(
    "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
    aws_region="us-east-1",
    s3_bucket="my-braket-bucket"
)
```

## Advanced Features

### Automatic Backend Selection

```python
from quantum_mlops.backends import BackendManager

manager = BackendManager()

# Create backends for different devices
manager.create_backend(QuantumDevice.SIMULATOR, wires=8)
manager.create_backend(QuantumDevice.IBM_QUANTUM, backend="ibmq_qasm_simulator")
manager.create_backend(QuantumDevice.AWS_BRAKET, device_arn="local:braket/braket.local.qubit")

# Execute with automatic fallback
circuits = [{"gates": [{"type": "h", "qubit": 0}], "n_qubits": 1}]
job = manager.execute_with_fallback(circuits, ["ibm_backend", "pennylane_simulator"])
```

### Cost Estimation

```python
executor = QuantumExecutor()

cost_info = executor.estimate_execution_cost(
    circuits=[circuit],
    backend_name="braket_sv1", 
    shots=10000
)

print(f"Estimated cost: ${cost_info['total_cost']:.4f}")
print(f"Execution time: {cost_info['estimated_time_minutes']:.1f} minutes")
```

### Performance Benchmarking

```python
# Benchmark all available backends
results = executor.benchmark_backends(shots=100)

for backend, performance in results.items():
    if performance["success"]:
        print(f"{backend}: {performance['execution_time']:.3f}s")
```

## Error Handling and Fallback

The system implements robust error handling with automatic fallback:

1. **Connection Errors** - Automatically try alternative backends
2. **Execution Timeouts** - Cancel jobs and retry or fallback
3. **Resource Limits** - Select backends based on circuit requirements
4. **Hardware Maintenance** - Detect unavailable hardware and use simulators

### Fallback Order

Default fallback order (customizable):
1. `pennylane_default.qubit`
2. `qiskit_qasm_simulator` 
3. `braket_local`

## Installation and Setup

### Dependencies

Install quantum computing libraries as needed:

```bash
# Core dependencies (always required)
pip install numpy

# PennyLane support
pip install pennylane pennylane-lightning

# Qiskit support  
pip install qiskit qiskit-machine-learning qiskit-ibm-runtime

# AWS Braket support
pip install amazon-braket-sdk boto3
```

### Optional Configurations

#### IBM Quantum Setup

1. Create account at [IBM Quantum](https://quantum-computing.ibm.com/)
2. Get API token from your account
3. Configure credentials:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

# Save credentials (one-time setup)
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="your_token_here"
)
```

#### AWS Braket Setup

1. Configure AWS credentials:

```bash
aws configure
# Enter your AWS Access Key ID, Secret Access Key, and region
```

2. Create S3 bucket for results:

```bash
aws s3 mb s3://my-braket-results-bucket
```

## Integration with MLOps Pipeline

### Enhanced QuantumMLPipeline

The `QuantumMLPipeline` class now supports real quantum backends:

```python
# Backend information
info = pipeline.get_backend_info()
print(f"Using device: {info['device']}")
print(f"Available backends: {info['available_backends']}")

# Performance benchmarking
benchmark = pipeline.benchmark_execution(test_samples=10)
print(f"Simulation time: {benchmark['simulation_time']:.3f}s")
print(f"Backend time: {benchmark['backend_time']:.3f}s")
```

### Circuit Translation

The system automatically translates high-level circuit descriptions to backend-specific formats:

```python
# Generic circuit description
circuit = {
    "gates": [
        {"type": "ry", "qubit": 0, "angle": np.pi/4},
        {"type": "cnot", "control": 0, "target": 1}
    ],
    "n_qubits": 2,
    "parameters": [0.1, 0.2, 0.3]  # Trainable parameters
}

# Automatically compiled for each backend:
# - PennyLane: QNode creation
# - Qiskit: QuantumCircuit with Parameter objects  
# - Braket: Circuit with parameterized gates
```

## Testing

### Running Tests

```bash
# Unit tests
python -m pytest tests/unit/test_backends.py -v

# Integration tests  
python -m pytest tests/integration/test_backend_integration.py -v

# Hardware tests (requires credentials)
QUANTUM_HARDWARE_TESTS=true python -m pytest tests/integration/test_quantum_hardware.py -v
```

### Demo Script

```bash
python examples/quantum_backend_demo.py
```

## Performance Considerations

### Backend Selection Guidelines

1. **Development/Testing**: Use `default.qubit` (PennyLane) or `qasm_simulator` (Qiskit)
2. **Small Circuits (<10 qubits)**: Local simulators are fastest
3. **Large Circuits**: Consider `lightning.qubit` or cloud simulators
4. **Production**: Real hardware for final validation

### Optimization Tips

1. **Circuit Depth**: Minimize for better hardware performance
2. **Shot Count**: Balance accuracy vs execution time/cost
3. **Batch Execution**: Group similar circuits for efficiency
4. **Caching**: Leverage built-in circuit compilation caching

## API Reference

### Core Classes

#### QuantumBackend

```python
class QuantumBackend(ABC):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_available(self) -> bool: ...
    def submit_job(self, circuits: List[Dict], shots: int) -> QuantumJob: ...
    def get_job_results(self, job_id: str) -> List[CircuitResult]: ...
```

#### QuantumExecutor

```python
class QuantumExecutor:
    def execute(self, circuits, backend_names=None, shots=1024) -> Union[CircuitResult, List[CircuitResult]]: ...
    def execute_async(self, circuits, backend_names=None, shots=1024) -> str: ...
    def benchmark_backends(self, test_circuit=None, shots=100) -> Dict: ...
    def estimate_execution_cost(self, circuits, backend_name, shots=1024) -> Dict: ...
```

## Troubleshooting

### Common Issues

#### ImportError: Module not found

```bash
# Install missing dependencies
pip install pennylane qiskit amazon-braket-sdk
```

#### Backend Connection Failed

```python
# Check backend status
executor = QuantumExecutor()
status = executor.get_backend_status()
print(status)
```

#### IBM Quantum Authentication

```python
# Verify credentials
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
print(service.backends())
```

#### AWS Braket Permissions

```bash
# Check AWS configuration
aws sts get-caller-identity
aws braket get-device --device-arn arn:aws:braket:::device/quantum-simulator/amazon/sv1
```

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now backend operations will show detailed logs
```

## Contributing

To extend the backend system:

1. **New Backend**: Inherit from `QuantumBackend` and implement required methods
2. **New Device**: Add to `QuantumDevice` enum and update `BackendManager.create_backend()`
3. **Optimization**: Extend `BackendManager.optimize_backend_selection()`

See `CONTRIBUTING.md` for detailed guidelines.

## License

This quantum backend integration is part of the Quantum MLOps Workbench and is licensed under the MIT License.