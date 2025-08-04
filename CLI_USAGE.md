# Quantum MLOps CLI - Comprehensive Usage Guide

The Quantum MLOps Workbench provides a powerful command-line interface for managing quantum machine learning workflows. This guide covers all commands and features.

## Installation & Setup

```bash
# Install the package
pip install quantum-mlops-workbench

# Verify installation
quantum-mlops version

# Initialize configuration
quantum-mlops config set default_backend simulator
quantum-mlops config set default_shots 1024
```

## Command Structure

The CLI is organized into logical command groups:

```
quantum-mlops <group> <command> [options]
```

### Available Command Groups

- **`model`** - Model training, evaluation, and prediction
- **`backend`** - Quantum backend management  
- **`monitor`** - Monitoring and visualization
- **`test`** - Testing and benchmarking
- **`config`** - Configuration management
- **`export`** - Export and reporting

## Model Commands

### Training Models

Train a quantum ML model with comprehensive options:

```bash
# Basic training
quantum-mlops model train data.csv

# Advanced training with custom parameters
quantum-mlops model train data.csv \
  --model-name my-quantum-model \
  --n-qubits 6 \
  --backend ibm_quantum \
  --epochs 200 \
  --learning-rate 0.005 \
  --layers 4 \
  --shots 2048 \
  --experiment-name experiment-1 \
  --track-gradients

# Training with specific backend configuration
quantum-mlops model train training_data.csv \
  --backend aws_braket \
  --n-qubits 4 \
  --shots 1000 \
  --save-model true
```

**Options:**
- `--model-name`: Name for the trained model
- `--n-qubits`: Number of qubits (default: 4)
- `--backend`: Quantum backend (simulator, ibm_quantum, aws_braket, ionq)
- `--epochs`: Training epochs (default: 100)
- `--learning-rate`: Optimizer learning rate (default: 0.01)
- `--layers`: Number of circuit layers (default: 3)
- `--shots`: Number of quantum shots (default: 1024)
- `--experiment-name`: Name for experiment tracking
- `--track-gradients`: Enable gradient statistics tracking

### Evaluating Models

Evaluate trained models with noise analysis:

```bash
# Basic evaluation
quantum-mlops model evaluate my-model test.csv

# Evaluation with noise analysis
quantum-mlops model evaluate my-model test.csv \
  --noise-models "depolarizing,amplitude_damping,phase_damping" \
  --backend simulator \
  --output evaluation_results.json

# Evaluate specific model version
quantum-mlops model evaluate my-model_v1.0 test_data.csv \
  --output results.json
```

**Options:**
- `--output`: Output file for results
- `--noise-models`: Comma-separated noise models to simulate
- `--backend`: Backend for evaluation

### Making Predictions

Generate predictions with trained models:

```bash
# Predict from file
quantum-mlops model predict my-model input.csv

# Predict single sample (JSON format)
quantum-mlops model predict my-model "[0.1, 0.2, 0.3, 0.4]"

# Predictions with custom output format
quantum-mlops model predict my-model input.csv \
  --output predictions.csv \
  --format-output csv \
  --backend simulator
```

**Options:**
- `--output`: Output file for predictions
- `--backend`: Prediction backend
- `--format-output`: Output format (json, csv, npy)

### Managing Models

List and manage registered models:

```bash
# List all models
quantum-mlops model list

# List with detailed information
quantum-mlops model list --show-details

# Filter models by name
quantum-mlops model list --name-filter "quantum"
```

## Backend Commands

### Listing Backends

View available quantum backends:

```bash
# List all backends
quantum-mlops backend list

# Show only available backends
quantum-mlops backend list --available-only

# Detailed backend information
quantum-mlops backend list --show-details
```

### Testing Backends

Test backend connectivity and performance:

```bash
# Test simulator
quantum-mlops backend test simulator

# Test with custom parameters
quantum-mlops backend test ibm_quantum \
  --n-qubits 2 \
  --shots 500 \
  --timeout 600 \
  --verbose

# Test specific backend
quantum-mlops backend test aws_braket \
  --n-qubits 4 \
  --shots 1000
```

**Options:**
- `--n-qubits`: Number of qubits for test
- `--shots`: Number of shots
- `--timeout`: Timeout in seconds
- `--verbose`: Verbose output

### Configuring Backends

Interactive backend configuration:

```bash
# Configure IBM Quantum
quantum-mlops backend configure ibm_quantum

# Configure AWS Braket
quantum-mlops backend configure aws_braket

# Configure IonQ
quantum-mlops backend configure ionq
```

The CLI will prompt for required credentials and settings interactively.

### Backend Status

Check real-time backend status:

```bash
# Show all backend status
quantum-mlops backend status

# Filter specific backend
quantum-mlops backend status --backend ibm_quantum

# Refresh status from providers
quantum-mlops backend status --refresh
```

## Monitoring Commands

### Dashboard

Launch interactive monitoring dashboard:

```bash
# Basic dashboard
quantum-mlops monitor dashboard

# Monitor specific experiment
quantum-mlops monitor dashboard \
  --experiment my-experiment \
  --port 8080 \
  --host 0.0.0.0 \
  --metrics "loss,accuracy,fidelity" \
  --refresh-rate 3
```

**Options:**
- `--experiment`: Experiment to monitor
- `--port`: Dashboard port (default: 8050)
- `--host`: Dashboard host (default: localhost)
- `--metrics`: Metrics to display
- `--refresh-rate`: Refresh rate in seconds

### Metrics Display

Display experiment metrics and statistics:

```bash
# Show all experiments
quantum-mlops monitor metrics

# Show specific experiment
quantum-mlops monitor metrics --experiment my-experiment

# Show specific run
quantum-mlops monitor metrics \
  --experiment my-experiment \
  --run-id run_123 \
  --last-n 20

# Export metrics
quantum-mlops monitor metrics \
  --experiment my-experiment \
  --export-format json
```

### Live Monitoring

Real-time metrics monitoring:

```bash
# Live monitoring
quantum-mlops monitor live my-experiment

# Custom metrics and refresh rate
quantum-mlops monitor live my-experiment \
  --metrics "loss,accuracy,gradient_variance" \
  --refresh-rate 1
```

## Testing Commands

### Running Tests

Execute quantum ML test suites:

```bash
# Run all tests
quantum-mlops test run

# Run specific test type
quantum-mlops test run --test-type unit
quantum-mlops test run --test-type integration
quantum-mlops test run --test-type hardware

# Verbose testing
quantum-mlops test run \
  --test-type all \
  --backend simulator \
  --verbose \
  --parallel true \
  --timeout 600
```

**Options:**
- `--test-type`: Test type (unit, integration, hardware, all)
- `--backend`: Backend for testing
- `--verbose`: Verbose output
- `--parallel`: Run tests in parallel
- `--timeout`: Test timeout in seconds

### Benchmarking

Performance benchmarking of quantum backends:

```bash
# Basic benchmark
quantum-mlops test benchmark

# Custom benchmark parameters
quantum-mlops test benchmark \
  --backend simulator \
  --n-qubits 6 \
  --samples 500 \
  --output benchmark_results.json

# Backend comparison
quantum-mlops test benchmark --backend ibm_quantum --samples 100
quantum-mlops test benchmark --backend aws_braket --samples 100
```

**Options:**
- `--backend`: Backend to benchmark
- `--n-qubits`: Number of qubits
- `--samples`: Number of samples
- `--output`: Output file for results

## Configuration Commands

### Viewing Configuration

Display current configuration:

```bash
# Show all configuration
quantum-mlops config show
```

### Setting Configuration

Set configuration values:

```bash
# Set default backend
quantum-mlops config set default_backend ibm_quantum

# Set default shots
quantum-mlops config set default_shots 2048

# Set experiment directory
quantum-mlops config set experiment_dir "/path/to/experiments"

# Set boolean values
quantum-mlops config set auto_save_models true
```

### Resetting Configuration

Reset configuration to defaults:

```bash
# Interactive reset (with confirmation)
quantum-mlops config reset

# Force reset (skip confirmation)
quantum-mlops config reset --yes
```

## Export Commands

### Report Export

Export comprehensive experiment reports:

```bash
# HTML report
quantum-mlops export report \
  --experiment my-experiment \
  --format html \
  --include-plots true

# PDF report
quantum-mlops export report \
  --experiment my-experiment \
  --format pdf \
  --output detailed_report.pdf

# JSON report
quantum-mlops export report \
  --experiment my-experiment \
  --format json
```

**Options:**
- `--experiment`: Experiment to export
- `--format`: Report format (html, pdf, json)
- `--output`: Output file path
- `--include-plots`: Include plots in report

### Model Export

Export trained models in various formats:

```bash
# ONNX export
quantum-mlops export model my-model --format onnx

# Pickle export
quantum-mlops export model my-model \
  --format pickle \
  --output my_model.pkl

# JSON export
quantum-mlops export model my-model \
  --format json \
  --output model_data.json
```

**Options:**
- `--format`: Export format (onnx, torchscript, pickle, json)
- `--output`: Output file path

## Main Commands

### Project Initialization

Initialize new quantum ML projects:

```bash
# Basic project
quantum-mlops init my-project

# Advanced project template
quantum-mlops init my-project \
  --template advanced \
  --force
```

**Options:**
- `--template`: Project template (basic, advanced)
- `--force`: Overwrite existing project

### Interactive Mode

Launch interactive exploration mode:

```bash
quantum-mlops interactive
```

Interactive commands:
- `help` - Show available commands
- `status` - Show system status
- `backends` - List available backends
- `models` - List registered models
- `experiments` - List experiments
- `config` - Show configuration
- `exit` - Exit interactive mode

### Version Information

Show version and dependency information:

```bash
quantum-mlops version
```

## Configuration Files

The CLI uses configuration files stored in `~/.quantum-mlops/`:

```
~/.quantum-mlops/
├── config.json          # Main configuration
├── models/              # Model registry
│   ├── registry.json    # Model metadata
│   └── *.pkl           # Model files
└── experiments/         # Experiment tracking
    └── exp_*/          # Individual experiments
```

### Example Configuration

```json
{
  "default_backend": "simulator",
  "default_shots": 1024,
  "experiment_dir": "/home/user/.quantum-mlops/experiments",
  "model_registry": "/home/user/.quantum-mlops/models",
  "log_level": "INFO",
  "ibm_quantum_config": {
    "token": "***",
    "hub": "ibm-q",
    "group": "open",
    "project": "main"
  }
}
```

## Error Handling & Debugging

### Debug Mode

Enable debug mode for detailed error information:

```bash
quantum-mlops --debug model train data.csv
```

### Common Issues

1. **Backend Connection Errors**: Check configuration and credentials
2. **Data Format Issues**: Ensure CSV/NPY files have correct format
3. **Memory Issues**: Reduce qubit count or use smaller datasets
4. **Timeout Issues**: Increase timeout values for slow operations

## Best Practices

### 1. Project Organization

```bash
# Initialize project structure
quantum-mlops init my-quantum-project --template advanced
cd my-quantum-project

# Organize data
mkdir -p data/{raw,processed,test}
mkdir -p models/{trained,exported}
mkdir -p results/{experiments,reports}
```

### 2. Configuration Management

```bash
# Set up environment-specific configs
quantum-mlops config set default_backend simulator  # Development
quantum-mlops config set default_backend ibm_quantum  # Production
```

### 3. Experiment Tracking

```bash
# Use meaningful experiment names
quantum-mlops model train data.csv --experiment-name "iris-classification-v1"

# Track important experiments
quantum-mlops monitor dashboard --experiment "production-run-2024"
```

### 4. Model Management

```bash
# Use semantic versioning for models
quantum-mlops model train data.csv --model-name "classifier-v1.0"

# Export models for deployment
quantum-mlops export model classifier-v1.0 --format onnx
```

### 5. Testing & Validation

```bash
# Test before production deployment
quantum-mlops test run --test-type all
quantum-mlops backend test ibm_quantum --n-qubits 2

# Benchmark performance
quantum-mlops test benchmark --backend simulator --samples 1000
```

## Advanced Workflows

### Complete ML Pipeline

```bash
# 1. Initialize project
quantum-mlops init quantum-classifier --template advanced
cd quantum-classifier

# 2. Configure backend
quantum-mlops backend configure ibm_quantum

# 3. Train model with experiment tracking
quantum-mlops model train data/train.csv \
  --model-name classifier-v1 \
  --experiment-name production-training \
  --backend ibm_quantum \
  --n-qubits 4 \
  --epochs 150

# 4. Monitor training
quantum-mlops monitor dashboard --experiment production-training

# 5. Evaluate model
quantum-mlops model evaluate classifier-v1 data/test.csv \
  --noise-models "depolarizing,amplitude_damping" \
  --output evaluation.json

# 6. Generate predictions
quantum-mlops model predict classifier-v1 data/new_data.csv \
  --output predictions.csv

# 7. Export model and report
quantum-mlops export model classifier-v1 --format onnx
quantum-mlops export report --experiment production-training --format pdf
```

### Multi-Backend Comparison

```bash
# Test multiple backends
for backend in simulator ibm_quantum aws_braket; do
  quantum-mlops backend test $backend --n-qubits 4
  quantum-mlops test benchmark --backend $backend --samples 100 \
    --output "benchmark_${backend}.json"
done

# Compare results
quantum-mlops monitor metrics --export-format json
```

## Support & Resources

- **Documentation**: Built-in help with `quantum-mlops <command> --help`
- **Examples**: Use `quantum-mlops interactive` for guided exploration  
- **Configuration**: `quantum-mlops config show` for current settings
- **Version Info**: `quantum-mlops version` for dependency status

The Quantum MLOps CLI provides a comprehensive, production-ready interface for quantum machine learning workflows. All commands include detailed help documentation accessible via `--help`.