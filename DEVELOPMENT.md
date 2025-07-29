# Development Guide

## Quick Start

### Prerequisites

- Python 3.9+ 
- Git
- Access to quantum computing backends (optional for simulator-only development)

### Setup Development Environment

1. **Clone and Install**:
```bash
git clone https://github.com/yourusername/quantum-mlops-workbench.git
cd quantum-mlops-workbench
make install-dev
```

2. **Verify Installation**:
```bash
# Run tests
make test

# Check quantum backends
quantum-mlops status
```

## Development Workflow

### 1. Code Quality Standards

**Automated Formatting**:
```bash
make format  # Auto-format with black, ruff, isort
```

**Linting and Type Checking**:
```bash
make lint      # Run all linting checks
make type-check # MyPy type validation
```

**Pre-commit Hooks**:
```bash
pre-commit install  # Install hooks (done by make install-dev)
pre-commit run --all-files  # Manual run
```

### 2. Testing Strategy

**Test Categories**:
- `tests/unit/`: Fast unit tests for individual components
- `tests/integration/`: End-to-end workflow tests  
- `tests/quantum/`: Quantum-specific hardware tests

**Running Tests**:
```bash
make test           # All tests with coverage
make test-fast      # Skip slow tests
make test-quantum   # Quantum-specific tests only
```

**Quantum Test Markers**:
```python
@pytest.mark.simulation  # Simulator-only tests
@pytest.mark.hardware    # Real quantum hardware required
@pytest.mark.slow       # Long-running tests
```

### 3. Quantum Development Best Practices

**Circuit Design**:
- Keep circuit depth minimal for NISQ devices
- Use hardware-native gates when possible
- Implement noise-aware circuit design

**Testing Quantum Circuits**:
```python
from quantum_mlops.testing import QuantumTestCase

class TestMyQuantumAlgorithm(QuantumTestCase):
    def test_gradient_stability(self):
        model = self.create_model(n_qubits=4)
        variance = self.measure_gradient_variance(model)
        self.assertLess(variance, 0.1)
```

**Backend Selection**:
```python
# Development: Use simulator for speed
device = QuantumDevice.SIMULATOR

# Testing: Validate on real hardware
device = QuantumDevice.IBM_QUANTUM if CI else QuantumDevice.SIMULATOR

# Production: Optimize for performance/cost
device = optimize_backend_selection(requirements)
```

### 4. Adding New Features

**1. Create Feature Branch**:
```bash
git checkout -b feature/new-quantum-algorithm
```

**2. Implement with Tests**:
- Add implementation in `src/quantum_mlops/`
- Add unit tests in `tests/unit/`
- Add integration tests in `tests/integration/`
- Add quantum-specific tests in `tests/quantum/`

**3. Validate Implementation**:
```bash
make test           # Run all tests
make lint           # Check code quality
make security       # Security validation
```

**4. Update Documentation**:
- Add docstrings (Google style)
- Update API documentation
- Add usage examples

## Quantum Backend Configuration

### Simulator Setup (Default)
```python
# No additional setup required
pipeline = QuantumMLPipeline(
    circuit=my_circuit,
    n_qubits=4,
    device=QuantumDevice.SIMULATOR
)
```

### AWS Braket Setup
```bash
# Install AWS dependencies
pip install quantum-mlops-workbench[aws]

# Configure credentials
aws configure
export AWS_DEFAULT_REGION=us-east-1
```

### IBM Quantum Setup
```bash
# Install IBM dependencies  
pip install quantum-mlops-workbench[ibm]

# Set IBM Quantum token
export IBM_QUANTUM_TOKEN=your_token_here
```

### IonQ Setup
```bash
# Set up through cloud provider (AWS/Azure/GCP)
export IONQ_API_KEY=your_api_key_here
```

## Performance Optimization

### Profiling Quantum Circuits
```python
from quantum_mlops.profiling import QuantumProfiler

profiler = QuantumProfiler()
with profiler.profile("my_circuit"):
    result = pipeline.train(X_train, y_train)

# View performance report
profiler.report()
```

### Circuit Optimization
```python
from quantum_mlops.optimization import CircuitOptimizer

optimizer = CircuitOptimizer(target_hardware='ibmq_toronto')
optimized_circuit = optimizer.compile(
    original_circuit,
    optimization_level=3
)
```

## Debugging

### Quantum State Inspection
```python
# Enable state vector logging
pipeline = QuantumMLPipeline(
    circuit=my_circuit,
    n_qubits=4,
    debug_mode=True
)

# Monitor quantum states during training
monitor = QuantumMonitor("debug_experiment")
with monitor.start_run():
    model = pipeline.train(X_train, y_train)
    # States logged automatically
```

### Common Issues

**1. Circuit Depth Too High**:
- Solution: Use circuit optimization or reduce layers
- Check: `model.circuit_depth < hardware_limit`

**2. Gradient Vanishing**:
- Solution: Adjust learning rate or use gradient-free optimization
- Check: `gradient_variance > 1e-6`

**3. Hardware Compatibility**:
- Solution: Use automatic gate decomposition
- Check: `assert_native_gates(circuit, backend)`

## Release Process

### 1. Version Update
```bash
# Update version in pyproject.toml and __init__.py
git add . && git commit -m "Bump version to X.Y.Z"
```

### 2. Testing
```bash
# Comprehensive testing
make test
make test-quantum --backend=all

# Security check
make security
```

### 3. Documentation
```bash
# Update documentation
make docs

# Check for broken links
make docs-check
```

### 4. Release
```bash
# Build distribution
make build

# Test upload
make publish-test

# Production release
make publish
```

## Contributing Guidelines

### Code Review Checklist
- [ ] All tests pass (`make test`)
- [ ] Code follows style guide (`make lint`)
- [ ] Security checks pass (`make security`)
- [ ] Documentation updated
- [ ] Quantum circuits optimized for target hardware
- [ ] Noise resilience validated
- [ ] Performance impact assessed

### Documentation Standards
- Use Google-style docstrings
- Include usage examples
- Document quantum-specific considerations
- Add type hints for all public APIs

### Quantum-Specific Review Points
- Circuit depth optimization
- Hardware compatibility validation
- Noise model accuracy
- Gradient stability analysis
- Resource usage efficiency