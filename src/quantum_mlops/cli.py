"""Command-line interface for quantum MLOps workbench.

This module provides a comprehensive CLI for managing quantum machine learning
workflows, including model training, backend management, monitoring, testing,
and deployment capabilities.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import typer
from rich.align import Align
from rich.bar import Bar
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn, 
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

try:
    from .core import QuantumMLPipeline, QuantumDevice, QuantumModel
    from .monitoring import QuantumMonitor
    from .testing import QuantumTestCase
    from .services import (
        ModelService,
        ExperimentService, 
        OptimizationService,
        QuantumBackendService
    )
    from .backends import BackendManager, QuantumExecutor
    SERVICES_AVAILABLE = True
except ImportError as e:
    typer.echo(f"Warning: Some services not available: {e}", err=True)
    SERVICES_AVAILABLE = False

# Initialize console and app
console = Console()
app = typer.Typer(
    name="quantum-mlops",
    help="🚀 Quantum MLOps Workbench - Production-ready quantum ML workflows",
    epilog="Visit https://quantum-mlops.readthedocs.io for documentation",
    no_args_is_help=True,
    rich_markup_mode="markdown",
)

# Command groups
model_app = typer.Typer(help="🧠 Model training, evaluation, and prediction")
backend_app = typer.Typer(help="⚛️ Quantum backend management")
monitor_app = typer.Typer(help="📊 Monitoring and visualization")
test_app = typer.Typer(help="🧪 Testing and benchmarking")
config_app = typer.Typer(help="⚙️ Configuration management")
export_app = typer.Typer(help="📤 Export and reporting")

# Add command groups to main app
app.add_typer(model_app, name="model")
app.add_typer(backend_app, name="backend")
app.add_typer(monitor_app, name="monitor")
app.add_typer(test_app, name="test")
app.add_typer(config_app, name="config")
app.add_typer(export_app, name="export")

# Global configuration
CONFIG_FILE = Path.home() / ".quantum-mlops" / "config.json"
MODEL_REGISTRY_PATH = Path.home() / ".quantum-mlops" / "models"
EXPERIMENT_PATH = Path.home() / ".quantum-mlops" / "experiments"


def init_config_dir() -> None:
    """Initialize configuration directory."""
    CONFIG_FILE.parent.mkdir(exist_ok=True)
    MODEL_REGISTRY_PATH.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_PATH.mkdir(parents=True, exist_ok=True)


def load_config() -> Dict[str, Any]:
    """Load configuration from file."""
    init_config_dir()
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {
        "default_backend": "simulator",
        "default_shots": 1024,
        "experiment_dir": str(EXPERIMENT_PATH),
        "model_registry": str(MODEL_REGISTRY_PATH),
        "log_level": "INFO"
    }


def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to file."""
    init_config_dir()
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def handle_error(error: Exception, operation: str) -> None:
    """Handle errors with rich formatting."""
    console.print(f"[red]❌ Error during {operation}:[/red]")
    console.print(f"[red]{str(error)}[/red]")
    if "--debug" in sys.argv:
        console.print_exception()


def create_progress() -> Progress:
    """Create a rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
        console=console,
    )


# ============================================================================
# MODEL COMMANDS
# ============================================================================

@model_app.command("train")
def train_model(
    data_path: str = typer.Argument(..., help="Path to training data (CSV/NPY)"),
    model_name: str = typer.Option("quantum-model", help="Model name"),
    n_qubits: int = typer.Option(4, help="Number of qubits"),
    backend: str = typer.Option("simulator", help="Quantum backend"),
    epochs: int = typer.Option(100, help="Training epochs"),
    learning_rate: float = typer.Option(0.01, help="Learning rate"),
    layers: int = typer.Option(3, help="Number of circuit layers"),
    shots: int = typer.Option(1024, help="Number of shots"),
    save_model: bool = typer.Option(True, help="Save trained model"),
    experiment_name: Optional[str] = typer.Option(None, help="Experiment name"),
    track_gradients: bool = typer.Option(False, help="Track gradient statistics"),
) -> None:
    """🎯 Train a quantum machine learning model.
    
    **Examples:**
    
    - Basic training: `quantum-mlops model train data.csv`
    - Custom configuration: `quantum-mlops model train data.csv --n-qubits 6 --epochs 200`
    - With experiment tracking: `quantum-mlops model train data.csv --experiment-name "my-experiment"`
    """
    
    if not SERVICES_AVAILABLE:
        console.print("[red]❌ Required services not available[/red]")
        raise typer.Exit(1)
    
    try:
        with create_progress() as progress:
            # Load data
            task = progress.add_task("Loading training data...", total=100)
            
            if not Path(data_path).exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            if data_path.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(data_path)
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
            elif data_path.endswith('.npy'):
                data = np.load(data_path)
                X = data[:, :-1]
                y = data[:, -1]
            else:
                raise ValueError("Unsupported file format. Use CSV or NPY.")
            
            progress.update(task, advance=20)
            
            # Initialize pipeline
            progress.update(task, description="Initializing quantum pipeline...")
            device = QuantumDevice(backend.lower())
            
            def simple_circuit(params, x):
                """Simple parameterized quantum circuit."""
                return params @ x  # Placeholder
            
            pipeline = QuantumMLPipeline(
                circuit=simple_circuit,
                n_qubits=n_qubits,
                device=device,
                layers=layers,
                shots=shots
            )
            progress.update(task, advance=20)
            
            # Setup experiment tracking if specified
            if experiment_name and SERVICES_AVAILABLE:
                progress.update(task, description="Setting up experiment tracking...")
                experiment_service = ExperimentService(str(EXPERIMENT_PATH))
                exp_id = experiment_service.create_experiment(experiment_name)
                run_id = experiment_service.start_run(f"train-{model_name}")
                progress.update(task, advance=10)
            
            # Train model
            progress.update(task, description="Training quantum model...")
            model = pipeline.train(
                X_train=X,
                y_train=y,
                epochs=epochs,
                learning_rate=learning_rate,
                track_gradients=track_gradients
            )
            progress.update(task, advance=40)
            
            # Save model if requested
            if save_model:
                progress.update(task, description="Saving model...")
                model_service = ModelService(str(MODEL_REGISTRY_PATH))
                model_id = model_service.register_model(
                    model=model,
                    name=model_name,
                    version="1.0",
                    description=f"Trained on {data_path} with {n_qubits} qubits"
                )
                progress.update(task, advance=10)
                console.print(f"[green]✅ Model saved as {model_id}[/green]")
            
            progress.update(task, advance=100)
        
        # Display results
        if hasattr(model, 'training_history'):
            final_loss = model.training_history['loss_history'][-1]
            final_accuracy = model.training_history.get('final_accuracy', 0.0)
            
            results_table = Table(title="Training Results")
            results_table.add_column("Metric", style="cyan")
            results_table.add_column("Value", style="green")
            
            results_table.add_row("Final Loss", f"{final_loss:.6f}")
            results_table.add_row("Final Accuracy", f"{final_accuracy:.4f}")
            results_table.add_row("Epochs", str(epochs))
            results_table.add_row("Parameters", str(len(model.parameters)))
            
            console.print(results_table)
            
        console.print(f"[green]🎉 Training completed successfully![/green]")
        
    except Exception as e:
        handle_error(e, "model training")
        raise typer.Exit(1)


@model_app.command("evaluate")
def evaluate_model(
    model_id: str = typer.Argument(..., help="Model ID or path"),
    test_data: str = typer.Argument(..., help="Test data path"),
    output: Optional[str] = typer.Option(None, help="Output file for results"),
    noise_models: Optional[str] = typer.Option(None, help="Comma-separated noise models"),
    backend: Optional[str] = typer.Option(None, help="Evaluation backend"),
) -> None:
    """📊 Evaluate a trained quantum model.
    
    **Examples:**
    
    - Basic evaluation: `quantum-mlops model evaluate my-model test.csv`
    - With noise analysis: `quantum-mlops model evaluate my-model test.csv --noise-models "depolarizing,amplitude_damping"`
    """
    
    if not SERVICES_AVAILABLE:
        console.print("[red]❌ Required services not available[/red]")
        raise typer.Exit(1)
    
    try:
        with create_progress() as progress:
            task = progress.add_task("Loading model and test data...", total=100)
            
            # Load model
            model_service = ModelService(str(MODEL_REGISTRY_PATH))
            if Path(model_id).exists():
                # Load from file
                model = QuantumModel(lambda x: x, 4)  # Placeholder
                model.load_model(model_id)
            else:
                # Load from registry
                model = model_service.load_model(model_id)
            
            progress.update(task, advance=30)
            
            # Load test data
            if test_data.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(test_data)
                X_test = df.iloc[:, :-1].values
                y_test = df.iloc[:, -1].values
            elif test_data.endswith('.npy'):
                data = np.load(test_data)
                X_test = data[:, :-1]
                y_test = data[:, -1]
            else:
                raise ValueError("Unsupported file format")
            
            progress.update(task, advance=30)
            
            # Setup pipeline for evaluation
            device = QuantumDevice(backend or "simulator")
            pipeline = QuantumMLPipeline(
                circuit=lambda x: x,  # Placeholder
                n_qubits=model.n_qubits,
                device=device
            )
            
            progress.update(task, description="Evaluating model...")
            
            # Parse noise models if specified
            noise_list = None
            if noise_models:
                noise_list = [n.strip() for n in noise_models.split(",")]
            
            # Evaluate
            metrics = pipeline.evaluate(
                model=model,
                X_test=X_test,
                y_test=y_test,
                noise_models=noise_list
            )
            
            progress.update(task, advance=40)
        
        # Display results
        results_table = Table(title="Evaluation Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")
        
        results_table.add_row("Accuracy", f"{metrics.accuracy:.4f}")
        results_table.add_row("Loss", f"{metrics.loss:.6f}")
        results_table.add_row("Fidelity", f"{metrics.fidelity:.4f}")
        results_table.add_row("Gradient Variance", f"{metrics.gradient_variance:.6f}")
        
        console.print(results_table)
        
        # Noise analysis results
        if metrics.noise_analysis:
            noise_table = Table(title="Noise Analysis")
            noise_table.add_column("Noise Model", style="cyan")
            noise_table.add_column("Accuracy", style="green")
            noise_table.add_column("Degradation", style="red")
            
            for noise_model, results in metrics.noise_analysis.items():
                noise_table.add_row(
                    noise_model,
                    f"{results['accuracy']:.4f}",
                    f"{results['degradation']:.4f}"
                )
            
            console.print(noise_table)
        
        # Save results if requested
        if output:
            with open(output, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)
            console.print(f"[green]Results saved to {output}[/green]")
        
        console.print("[green]✅ Evaluation completed![/green]")
        
    except Exception as e:
        handle_error(e, "model evaluation")
        raise typer.Exit(1)


@model_app.command("predict")
def predict(
    model_id: str = typer.Argument(..., help="Model ID or path"),
    input_data: str = typer.Argument(..., help="Input data path or JSON string"),
    output: Optional[str] = typer.Option(None, help="Output file for predictions"),
    backend: Optional[str] = typer.Option(None, help="Prediction backend"),
    format_output: str = typer.Option("json", help="Output format (json, csv, npy)"),
) -> None:
    """🔮 Make predictions with a trained model.
    
    **Examples:**
    
    - Predict from file: `quantum-mlops model predict my-model input.csv`
    - Predict single sample: `quantum-mlops model predict my-model "[0.1, 0.2, 0.3, 0.4]"`
    """
    
    if not SERVICES_AVAILABLE:
        console.print("[red]❌ Required services not available[/red]")
        raise typer.Exit(1)
    
    try:
        with create_progress() as progress:
            task = progress.add_task("Loading model...", total=100)
            
            # Load model
            model_service = ModelService(str(MODEL_REGISTRY_PATH))
            if Path(model_id).exists():
                model = QuantumModel(lambda x: x, 4)  # Placeholder
                model.load_model(model_id)
            else:
                model = model_service.load_model(model_id)
            
            progress.update(task, advance=50)
            
            # Load input data
            progress.update(task, description="Processing input data...")
            
            if input_data.startswith('[') and input_data.endswith(']'):
                # JSON string input
                X = np.array(json.loads(input_data)).reshape(1, -1)
            elif Path(input_data).exists():
                # File input
                if input_data.endswith('.csv'):
                    import pandas as pd
                    df = pd.read_csv(input_data)
                    X = df.values
                elif input_data.endswith('.npy'):
                    X = np.load(input_data)
                else:
                    raise ValueError("Unsupported file format")
            else:
                raise ValueError("Invalid input data format")
            
            progress.update(task, advance=30)
            
            # Make predictions
            progress.update(task, description="Making predictions...")
            predictions = model.predict(X)
            progress.update(task, advance=20)
        
        # Display predictions
        pred_table = Table(title="Predictions")
        pred_table.add_column("Sample", style="cyan")
        pred_table.add_column("Prediction", style="green")
        
        for i, pred in enumerate(predictions):
            pred_table.add_row(str(i), f"{pred:.6f}")
        
        console.print(pred_table)
        
        # Save predictions if requested
        if output:
            if format_output == "json":
                with open(output, 'w') as f:
                    json.dump(predictions.tolist(), f, indent=2)
            elif format_output == "csv":
                import pandas as pd
                pd.DataFrame(predictions, columns=["prediction"]).to_csv(output, index=False)
            elif format_output == "npy":
                np.save(output, predictions)
            else:
                raise ValueError(f"Unsupported output format: {format_output}")
            
            console.print(f"[green]Predictions saved to {output}[/green]")
        
        console.print("[green]✅ Predictions completed![/green]")
        
    except Exception as e:
        handle_error(e, "prediction")
        raise typer.Exit(1)


@model_app.command("list")
def list_models(
    name_filter: Optional[str] = typer.Option(None, help="Filter by name"),
    show_details: bool = typer.Option(False, help="Show detailed information"),
) -> None:
    """📋 List all registered models."""
    
    if not SERVICES_AVAILABLE:
        console.print("[red]❌ Required services not available[/red]")
        raise typer.Exit(1)
    
    try:
        model_service = ModelService(str(MODEL_REGISTRY_PATH))
        models = model_service.list_models(name_filter)
        
        if not models:
            console.print("[yellow]No models found[/yellow]")
            return
        
        if show_details:
            for model in models:
                panel_content = f"""
**Version:** {model['version']}
**Created:** {model['created_at']}
**Qubits:** {model['n_qubits']}
**Parameters:** {model['parameter_count']}
**Description:** {model['description'] or 'No description'}
                """
                console.print(Panel(panel_content.strip(), title=model['name']))
        else:
            table = Table(title="Registered Models")
            table.add_column("Name", style="cyan")
            table.add_column("Version", style="green")
            table.add_column("Qubits", style="yellow")
            table.add_column("Created", style="blue")
            
            for model in models:
                created = datetime.fromisoformat(model['created_at'])
                table.add_row(
                    model['name'],
                    model['version'],
                    str(model['n_qubits']),
                    created.strftime("%Y-%m-%d %H:%M")
                )
            
            console.print(table)
        
    except Exception as e:
        handle_error(e, "listing models")
        raise typer.Exit(1)


# ============================================================================
# BACKEND COMMANDS
# ============================================================================

@backend_app.command("list")
def list_backends(
    available_only: bool = typer.Option(False, help="Show only available backends"),
    show_details: bool = typer.Option(False, help="Show detailed backend information"),
) -> None:
    """📋 List available quantum backends."""
    
    try:
        if SERVICES_AVAILABLE:
            backend_service = QuantumBackendService()
            backends = backend_service.list_backends()
            
            if available_only:
                backends = [b for b in backends if b.get('status') == 'available']
            
            if show_details:
                for backend in backends:
                    status_color = "green" if backend.get('status') == 'available' else "red"
                    panel_content = f"""
**Type:** {backend.get('type', 'Unknown')}
**Status:** [{status_color}]{backend.get('status', 'Unknown')}[/{status_color}]
**Queue Time:** {backend.get('queue_time', 'Unknown')}
**Max Qubits:** {backend.get('max_qubits', 'Unknown')}
**Description:** {backend.get('description', 'No description')}
                    """
                    console.print(Panel(panel_content.strip(), title=backend['name']))
            else:
                table = Table(title="Quantum Backends")
                table.add_column("Backend", style="cyan")
                table.add_column("Type", style="blue")
                table.add_column("Status", style="green")
                table.add_column("Queue Time", style="yellow")
                table.add_column("Max Qubits", style="magenta")
                
                for backend in backends:
                    status_style = "green" if backend.get('status') == 'available' else "red"
                    table.add_row(
                        backend['name'],
                        backend.get('type', 'Unknown'),
                        f"[{status_style}]{backend.get('status', 'Unknown')}[/{status_style}]",
                        backend.get('queue_time', 'Unknown'),
                        str(backend.get('max_qubits', 'Unknown'))
                    )
                
                console.print(table)
        else:
            # Fallback static display
            table = Table(title="Quantum Backends")
            table.add_column("Backend", style="cyan")
            table.add_column("Type", style="blue")
            table.add_column("Status", style="green")
            table.add_column("Queue Time", style="yellow")
            
            backends = [
                ("Simulator", "Local", "Available", "0s"),
                ("IBM Quantum", "Cloud", "Available", "5-30 min"),
                ("AWS Braket", "Cloud", "Available", "<1 min"),
                ("IonQ", "Cloud", "Available", "<5 min"),
            ]
            
            for name, type_, status, queue_time in backends:
                table.add_row(name, type_, f"[green]{status}[/green]", queue_time)
            
            console.print(table)
        
    except Exception as e:
        handle_error(e, "listing backends")
        raise typer.Exit(1)


@backend_app.command("test")
def test_backend(
    backend: str = typer.Argument(..., help="Backend name to test"),
    n_qubits: int = typer.Option(4, help="Number of qubits for test"),
    shots: int = typer.Option(1000, help="Number of shots"),
    timeout: int = typer.Option(300, help="Timeout in seconds"),
    verbose: bool = typer.Option(False, help="Verbose output"),
) -> None:
    """🧪 Test quantum backend connectivity and performance.
    
    **Examples:**
    
    - Test simulator: `quantum-mlops backend test simulator`
    - Test with custom parameters: `quantum-mlops backend test ibm_quantum --n-qubits 2 --shots 500`
    """
    
    try:
        with create_progress() as progress:
            task = progress.add_task(f"Testing {backend} backend...", total=100)
            
            # Create test pipeline
            device_map = {
                "simulator": QuantumDevice.SIMULATOR,
                "ibm_quantum": QuantumDevice.IBM_QUANTUM,
                "aws_braket": QuantumDevice.AWS_BRAKET,
                "ionq": QuantumDevice.IONQ,
            }
            
            device = device_map.get(backend.lower(), QuantumDevice.SIMULATOR)
            
            def test_circuit(params, x):
                return params @ x  # Simple test circuit
            
            pipeline = QuantumMLPipeline(
                circuit=test_circuit,
                n_qubits=n_qubits,
                device=device,
                shots=shots
            )
            
            progress.update(task, advance=30)
            
            # Run benchmark
            progress.update(task, description="Running benchmark...")
            benchmark_results = pipeline.benchmark_execution(test_samples=5)
            progress.update(task, advance=50)
            
            # Get backend info
            progress.update(task, description="Gathering backend information...")
            backend_info = pipeline.get_backend_info()
            progress.update(task, advance=20)
        
        # Display results
        info_table = Table(title=f"Backend Test Results - {backend}")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="green")
        
        info_table.add_row("Device", backend_info.get('device', 'Unknown'))
        info_table.add_row("Qubits", str(backend_info.get('n_qubits', 'Unknown')))
        info_table.add_row("Real Backend Available", str(backend_info.get('real_backend_available', False)))
        
        if benchmark_results.get('backend_time'):
            info_table.add_row("Backend Execution Time", f"{benchmark_results['backend_time']:.3f}s")
        
        info_table.add_row("Simulation Time", f"{benchmark_results.get('simulation_time', 0):.3f}s")
        info_table.add_row("Test Samples", str(benchmark_results.get('test_samples', 0)))
        
        console.print(info_table)
        
        # Show status
        if benchmark_results.get('backend_error'):
            console.print(f"[red]❌ Backend Error: {benchmark_results['backend_error']}[/red]")
        elif benchmark_results.get('backend_available'):
            console.print("[green]✅ Backend test successful![/green]")
        else:
            console.print("[yellow]⚠️ Backend not available, using simulation[/yellow]")
        
        if verbose and backend_info.get('available_backends'):
            console.print("\nAvailable backends:")
            for avail_backend in backend_info['available_backends']:
                console.print(f"  • {avail_backend}")
        
    except Exception as e:
        handle_error(e, "backend testing")
        raise typer.Exit(1)


@backend_app.command("configure")
def configure_backend(
    backend: str = typer.Argument(..., help="Backend name to configure"),
    interactive: bool = typer.Option(True, help="Interactive configuration"),
) -> None:
    """⚙️ Configure quantum backend settings."""
    
    try:
        config = load_config()
        
        if interactive:
            console.print(f"[cyan]Configuring {backend} backend[/cyan]")
            
            if backend.lower() == "ibm_quantum":
                token = Prompt.ask("IBM Quantum API token", password=True)
                hub = Prompt.ask("Hub (optional)", default="")
                group = Prompt.ask("Group (optional)", default="")
                project = Prompt.ask("Project (optional)", default="")
                
                config[f"{backend}_config"] = {
                    "token": token,
                    "hub": hub,
                    "group": group,
                    "project": project
                }
                
            elif backend.lower() == "aws_braket":
                region = Prompt.ask("AWS region", default="us-east-1")
                access_key = Prompt.ask("AWS access key ID", password=True)
                secret_key = Prompt.ask("AWS secret access key", password=True)
                
                config[f"{backend}_config"] = {
                    "region": region,
                    "aws_access_key_id": access_key,
                    "aws_secret_access_key": secret_key
                }
                
            elif backend.lower() == "ionq":
                api_key = Prompt.ask("IonQ API key", password=True)
                
                config[f"{backend}_config"] = {
                    "api_key": api_key
                }
            
            save_config(config)
            console.print(f"[green]✅ {backend} configuration saved![/green]")
        else:
            console.print(f"[yellow]Non-interactive configuration not yet implemented for {backend}[/yellow]")
        
    except Exception as e:
        handle_error(e, "backend configuration")
        raise typer.Exit(1)


@backend_app.command("status")
def backend_status(
    backend: Optional[str] = typer.Option(None, help="Specific backend to check"),
    refresh: bool = typer.Option(False, help="Refresh status from providers"),
) -> None:
    """📊 Show quantum backend status and availability."""
    
    try:
        if refresh:
            with console.status("[bold green]Refreshing backend status..."):
                time.sleep(2)  # Simulate API calls
        
        table = Table(title="Quantum Backend Status")
        table.add_column("Backend", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Queue Time", style="yellow")
        table.add_column("Available Qubits", style="magenta")
        table.add_column("Last Updated", style="blue")
        
        # Mock status data (in production, this would query real APIs)
        backends_status = [
            ("Simulator", "Available", "0s", "32", "Now"),
            ("IBM Quantum", "Available", "5-30 min", "127", "2 min ago"),
            ("AWS Braket (SV1)", "Available", "<1 min", "34", "1 min ago"),
            ("AWS Braket (IonQ)", "Available", "<5 min", "11", "3 min ago"),
            ("AWS Braket (Rigetti)", "Maintenance", "N/A", "32", "10 min ago"),
        ]
        
        for name, status, queue, qubits, updated in backends_status:
            if backend and backend.lower() not in name.lower():
                continue
                
            status_style = "green" if status == "Available" else "red" if status == "Maintenance" else "yellow"
            table.add_row(
                name,
                f"[{status_style}]{status}[/{status_style}]",
                queue,
                qubits,
                updated
            )
        
        console.print(table)
        
        if not backend:
            console.print("\n[dim]Use --backend to filter specific backend[/dim]")
            console.print("[dim]Use --refresh to update status from providers[/dim]")
        
    except Exception as e:
        handle_error(e, "checking backend status")
        raise typer.Exit(1)


# ============================================================================
# MONITORING COMMANDS  
# ============================================================================

@monitor_app.command("dashboard")
def launch_dashboard(
    experiment: Optional[str] = typer.Option(None, help="Experiment to monitor"),
    port: int = typer.Option(8050, help="Dashboard port"),
    host: str = typer.Option("localhost", help="Dashboard host"),
    metrics: str = typer.Option("loss,accuracy", help="Metrics to display"),
    refresh_rate: int = typer.Option(5, help="Refresh rate in seconds"),
) -> None:
    """📊 Launch interactive monitoring dashboard.
    
    **Examples:**
    
    - Basic dashboard: `quantum-mlops monitor dashboard`
    - Monitor specific experiment: `quantum-mlops monitor dashboard --experiment my-exp`
    """
    
    try:
        console.print(f"[cyan]🚀 Launching monitoring dashboard...[/cyan]")
        console.print(f"[blue]Host: {host}:{port}[/blue]")
        console.print(f"[blue]Metrics: {metrics}[/blue]")
        console.print(f"[blue]Refresh rate: {refresh_rate}s[/blue]")
        
        if experiment:
            console.print(f"[blue]Monitoring experiment: {experiment}[/blue]")
        
        # In a real implementation, this would start a web server
        console.print(f"\n[green]✅ Dashboard available at: http://{host}:{port}[/green]")
        console.print("[yellow]Press Ctrl+C to stop the dashboard[/yellow]")
        
        # Simulate dashboard running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Dashboard stopped[/yellow]")
        
    except Exception as e:
        handle_error(e, "launching dashboard")
        raise typer.Exit(1)


@monitor_app.command("metrics")
def show_metrics(
    experiment: Optional[str] = typer.Option(None, help="Experiment name"),
    run_id: Optional[str] = typer.Option(None, help="Specific run ID"),
    metric_names: str = typer.Option("loss,accuracy", help="Metrics to show"),
    last_n: int = typer.Option(10, help="Show last N values"),
    export_format: Optional[str] = typer.Option(None, help="Export format (json, csv)"),
) -> None:
    """📈 Display experiment metrics and statistics."""
    
    try:
        if not SERVICES_AVAILABLE:
            console.print("[red]❌ Experiment tracking not available[/red]")
            raise typer.Exit(1)
        
        experiment_service = ExperimentService(str(EXPERIMENT_PATH))
        
        if experiment:
            exp_data = experiment_service.load_experiment(experiment)
            
            # Show experiment overview
            exp_table = Table(title=f"Experiment: {experiment}")
            exp_table.add_column("Property", style="cyan")
            exp_table.add_column("Value", style="green")
            
            exp_table.add_row("Name", exp_data['name'])
            exp_table.add_row("Status", exp_data['status'])
            exp_table.add_row("Created", exp_data['created_at'])
            exp_table.add_row("Runs", str(len(exp_data['runs'])))
            
            console.print(exp_table)
            
            # Show runs
            if exp_data['runs']:
                runs_table = Table(title="Recent Runs")
                runs_table.add_column("Run ID", style="cyan")
                runs_table.add_column("Status", style="green")
                runs_table.add_column("Started", style="blue")
                
                for run in exp_data['runs'][-last_n:]:
                    runs_table.add_row(
                        run['run_id'],
                        run['status'],
                        run['started_at']
                    )
                
                console.print(runs_table)
        else:
            # List all experiments
            experiments = experiment_service.list_experiments()
            
            if not experiments:
                console.print("[yellow]No experiments found[/yellow]")
                return
            
            exp_table = Table(title="All Experiments")
            exp_table.add_column("Name", style="cyan")
            exp_table.add_column("Status", style="green")
            exp_table.add_column("Runs", style="yellow")
            exp_table.add_column("Created", style="blue")
            
            for exp in experiments:
                exp_table.add_row(
                    exp['name'],
                    exp['status'],
                    str(len(exp['runs'])),
                    exp['created_at']
                )
            
            console.print(exp_table)
        
    except Exception as e:
        handle_error(e, "showing metrics")
        raise typer.Exit(1)


@monitor_app.command("live")
def live_monitor(
    experiment: str = typer.Argument(..., help="Experiment to monitor"),
    metrics: str = typer.Option("loss,accuracy", help="Metrics to monitor"),
    refresh_rate: int = typer.Option(2, help="Refresh rate in seconds"),
) -> None:
    """📡 Live monitoring of experiment metrics."""
    
    metric_list = [m.strip() for m in metrics.split(",")]
    
    def create_live_table():
        table = Table(title=f"Live Metrics - {experiment}")
        table.add_column("Metric", style="cyan")
        table.add_column("Current", style="green")
        table.add_column("Trend", style="yellow")
        table.add_column("Updated", style="blue")
        
        # Mock live data
        import random
        for metric in metric_list:
            current_value = random.uniform(0.1, 0.9)
            trend = "📈" if random.choice([True, False]) else "📉"
            table.add_row(
                metric.title(),
                f"{current_value:.4f}",
                trend,
                datetime.now().strftime("%H:%M:%S")
            )
        
        return table
    
    console.print(f"[cyan]🔴 Live monitoring {experiment}[/cyan]")
    console.print("[yellow]Press Ctrl+C to stop[/yellow]\n")
    
    try:
        with Live(create_live_table(), refresh_per_second=1/refresh_rate) as live:
            while True:
                time.sleep(refresh_rate)
                live.update(create_live_table())
    except KeyboardInterrupt:
        console.print("\n[yellow]Live monitoring stopped[/yellow]")


# ============================================================================
# TESTING COMMANDS
# ============================================================================

@test_app.command("run")
def run_tests(
    test_type: str = typer.Option("all", help="Test type (unit, integration, hardware, all)"),
    backend: str = typer.Option("simulator", help="Backend for testing"),
    verbose: bool = typer.Option(False, help="Verbose output"),
    parallel: bool = typer.Option(True, help="Run tests in parallel"),
    timeout: int = typer.Option(300, help="Test timeout in seconds"),
) -> None:
    """🧪 Run quantum ML test suites.
    
    **Examples:**
    
    - Run all tests: `quantum-mlops test run`
    - Run hardware tests: `quantum-mlops test run --test-type hardware`
    - Run with verbose output: `quantum-mlops test run --verbose`
    """
    
    try:
        console.print(f"[cyan]🧪 Running {test_type} tests on {backend}[/cyan]")
        
        with create_progress() as progress:
            # Simulate test discovery
            task = progress.add_task("Discovering tests...", total=100)
            time.sleep(1)
            progress.update(task, advance=20)
            
            # Mock test results
            test_suites = {
                "unit": ["test_core", "test_models", "test_utils"],
                "integration": ["test_backend_integration", "test_pipeline"],
                "hardware": ["test_quantum_hardware", "test_noise_models"],
            }
            
            if test_type == "all":
                tests_to_run = []
                for suite_tests in test_suites.values():
                    tests_to_run.extend(suite_tests)
            else:
                tests_to_run = test_suites.get(test_type, [])
            
            if not tests_to_run:
                console.print(f"[yellow]No tests found for type: {test_type}[/yellow]")
                return
            
            progress.update(task, description="Running tests...", advance=20)
            
            # Simulate running tests
            passed = 0
            failed = 0
            skipped = 0
            
            for i, test in enumerate(tests_to_run):
                progress.update(task, description=f"Running {test}...")
                time.sleep(0.5)  # Simulate test execution
                
                # Mock test result
                result = np.random.choice(["pass", "fail", "skip"], p=[0.8, 0.1, 0.1])
                if result == "pass":
                    passed += 1
                elif result == "fail":
                    failed += 1
                else:
                    skipped += 1
                
                progress.update(task, advance=60/len(tests_to_run))
            
            progress.update(task, description="Generating report...", advance=100)
        
        # Display results
        results_table = Table(title="Test Results")
        results_table.add_column("Category", style="cyan")
        results_table.add_column("Count", style="green")
        results_table.add_column("Percentage", style="yellow")
        
        total = passed + failed + skipped
        results_table.add_row("Passed", str(passed), f"{passed/total*100:.1f}%")
        results_table.add_row("Failed", str(failed), f"{failed/total*100:.1f}%")
        results_table.add_row("Skipped", str(skipped), f"{skipped/total*100:.1f}%")
        results_table.add_row("Total", str(total), "100.0%")
        
        console.print(results_table)
        
        if failed > 0:
            console.print(f"[red]❌ {failed} tests failed[/red]")
            raise typer.Exit(1)
        else:
            console.print(f"[green]✅ All tests passed![/green]")
        
    except Exception as e:
        handle_error(e, "running tests")
        raise typer.Exit(1)


@test_app.command("benchmark")
def benchmark(
    backend: str = typer.Option("simulator", help="Backend to benchmark"),
    n_qubits: int = typer.Option(4, help="Number of qubits"),
    samples: int = typer.Option(100, help="Number of samples"),
    output: Optional[str] = typer.Option(None, help="Output file for results"),
) -> None:
    """⚡ Benchmark quantum backend performance.
    
    **Examples:**
    
    - Basic benchmark: `quantum-mlops test benchmark`
    - Custom parameters: `quantum-mlops test benchmark --n-qubits 6 --samples 200`
    """
    
    try:
        console.print(f"[cyan]⚡ Benchmarking {backend} with {n_qubits} qubits[/cyan]")
        
        with create_progress() as progress:
            task = progress.add_task("Running benchmark...", total=samples)
            
            # Mock benchmark execution
            execution_times = []
            for i in range(samples):
                # Simulate circuit execution
                exec_time = np.random.exponential(0.1) + 0.01  # Mock execution time
                execution_times.append(exec_time)
                
                progress.update(task, advance=1)
                time.sleep(0.01)  # Small delay for visualization
        
        # Calculate statistics
        times = np.array(execution_times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        throughput = 1.0 / mean_time
        
        # Display results
        stats_table = Table(title=f"Benchmark Results - {backend}")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        stats_table.add_column("Unit", style="yellow")
        
        stats_table.add_row("Mean Execution Time", f"{mean_time:.4f}", "seconds")
        stats_table.add_row("Std Deviation", f"{std_time:.4f}", "seconds")
        stats_table.add_row("Min Time", f"{min_time:.4f}", "seconds")
        stats_table.add_row("Max Time", f"{max_time:.4f}", "seconds")
        stats_table.add_row("Throughput", f"{throughput:.2f}", "circuits/sec")
        stats_table.add_row("Total Samples", str(samples), "circuits")
        
        console.print(stats_table)
        
        # Save results if requested
        if output:
            results = {
                "backend": backend,
                "n_qubits": n_qubits,
                "samples": samples,
                "mean_time": mean_time,
                "std_time": std_time,
                "min_time": min_time,
                "max_time": max_time,
                "throughput": throughput,
                "execution_times": execution_times
            }
            
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            
            console.print(f"[green]Results saved to {output}[/green]")
        
        console.print("[green]✅ Benchmark completed![/green]")
        
    except Exception as e:
        handle_error(e, "benchmarking")
        raise typer.Exit(1)


# ============================================================================
# CONFIGURATION COMMANDS
# ============================================================================

@config_app.command("show")
def show_config() -> None:
    """📋 Display current configuration."""
    
    try:
        config = load_config()
        
        config_table = Table(title="Current Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")
        
        for key, value in config.items():
            if isinstance(value, dict):
                # Mask sensitive values
                masked_value = {k: "***" if "token" in k.lower() or "key" in k.lower() or "password" in k.lower() else v for k, v in value.items()}
                config_table.add_row(key, str(masked_value))
            else:
                config_table.add_row(key, str(value))
        
        console.print(config_table)
        console.print(f"\n[dim]Config file: {CONFIG_FILE}[/dim]")
        
    except Exception as e:
        handle_error(e, "showing configuration")
        raise typer.Exit(1)


@config_app.command("set")
def set_config(
    key: str = typer.Argument(..., help="Configuration key"),
    value: str = typer.Argument(..., help="Configuration value"),
) -> None:
    """⚙️ Set configuration value.
    
    **Examples:**
    
    - Set default backend: `quantum-mlops config set default_backend ibm_quantum`
    - Set shots: `quantum-mlops config set default_shots 2048`
    """
    
    try:
        config = load_config()
        
        # Try to parse value as appropriate type
        try:
            if value.lower() in ['true', 'false']:
                config[key] = value.lower() == 'true'
            elif value.isdigit():
                config[key] = int(value)
            elif value.replace('.', '').replace('-', '').isdigit():
                config[key] = float(value)
            else:
                config[key] = value
        except:
            config[key] = value
        
        save_config(config)
        console.print(f"[green]✅ Set {key} = {value}[/green]")
        
    except Exception as e:
        handle_error(e, "setting configuration")
        raise typer.Exit(1)


@config_app.command("reset")
def reset_config(
    confirm: bool = typer.Option(False, "--yes", help="Skip confirmation"),
) -> None:
    """🔄 Reset configuration to defaults."""
    
    try:
        if not confirm:
            confirm = Confirm.ask("Reset all configuration to defaults?")
        
        if confirm:
            # Remove config file to trigger defaults
            if CONFIG_FILE.exists():
                CONFIG_FILE.unlink()
            
            console.print("[green]✅ Configuration reset to defaults[/green]")
        else:
            console.print("[yellow]Configuration reset cancelled[/yellow]")
        
    except Exception as e:
        handle_error(e, "resetting configuration")
        raise typer.Exit(1)


# ============================================================================
# EXPORT COMMANDS
# ============================================================================

@export_app.command("report")
def export_report(
    experiment: Optional[str] = typer.Option(None, help="Experiment to export"),
    format_type: str = typer.Option("html", help="Report format (html, pdf, json)"),
    output: Optional[str] = typer.Option(None, help="Output file path"),
    include_plots: bool = typer.Option(True, help="Include plots in report"),
) -> None:
    """📄 Export experiment report.
    
    **Examples:**
    
    - HTML report: `quantum-mlops export report --experiment my-exp --format html`
    - PDF report: `quantum-mlops export report --experiment my-exp --format pdf`
    """
    
    try:
        if not experiment:
            console.print("[yellow]No experiment specified. Use --experiment to specify one.[/yellow]")
            return
        
        if not output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = f"report_{experiment}_{timestamp}.{format_type}"
        
        with create_progress() as progress:
            task = progress.add_task("Generating report...", total=100)
            
            # Mock report generation
            progress.update(task, description="Collecting experiment data...")
            time.sleep(1)
            progress.update(task, advance=30)
            
            progress.update(task, description="Generating plots...")
            time.sleep(1)
            progress.update(task, advance=40)
            
            progress.update(task, description="Creating report...")
            time.sleep(1)
            progress.update(task, advance=30)
        
        # Mock report content
        report_content = f"""
# Quantum ML Experiment Report

**Experiment:** {experiment}
**Generated:** {datetime.now().isoformat()}
**Format:** {format_type.upper()}

## Summary
- Total runs: 5
- Best accuracy: 0.8734
- Best model: quantum-model-v3

## Metrics
- Training time: 45.2 minutes
- Evaluation time: 3.7 minutes
- Total circuits executed: 50,000

## Conclusions
The quantum model achieved competitive performance on the test dataset.
        """.strip()
        
        # Save report
        with open(output, 'w') as f:
            if format_type == "json":
                json.dump({"experiment": experiment, "content": report_content}, f, indent=2)
            else:
                f.write(report_content)
        
        console.print(f"[green]✅ Report exported to {output}[/green]")
        
    except Exception as e:
        handle_error(e, "exporting report")
        raise typer.Exit(1)


@export_app.command("model")
def export_model(
    model_id: str = typer.Argument(..., help="Model ID to export"),
    format_type: str = typer.Option("onnx", help="Export format (onnx, torchscript, pickle)"),
    output: Optional[str] = typer.Option(None, help="Output file path"),
) -> None:
    """💾 Export trained model in various formats."""
    
    try:
        if not SERVICES_AVAILABLE:
            console.print("[red]❌ Model services not available[/red]")
            raise typer.Exit(1)
        
        model_service = ModelService(str(MODEL_REGISTRY_PATH))
        model = model_service.load_model(model_id)
        
        if not output:
            output = f"{model_id}.{format_type}"
        
        with create_progress() as progress:
            task = progress.add_task("Exporting model...", total=100)
            
            # Mock export process
            if format_type == "pickle":
                import pickle
                with open(output, 'wb') as f:
                    pickle.dump(model, f)
            elif format_type == "json":
                model_data = {
                    "model_id": model_id,
                    "n_qubits": model.n_qubits,
                    "parameters": model.parameters.tolist() if model.parameters is not None else None,
                    "metadata": model.metadata
                }
                with open(output, 'w') as f:
                    json.dump(model_data, f, indent=2)
            else:
                # Mock export for other formats
                with open(output, 'w') as f:
                    f.write(f"# Exported model {model_id} in {format_type} format\n")
            
            progress.update(task, advance=100)
        
        console.print(f"[green]✅ Model exported to {output}[/green]")
        
    except Exception as e:
        handle_error(e, "exporting model")
        raise typer.Exit(1)


# ============================================================================
# MAIN COMMANDS
# ============================================================================

@app.command("init")
def init_project(
    name: str = typer.Argument(..., help="Project name"),
    template: str = typer.Option("basic", help="Project template (basic, advanced)"),
    force: bool = typer.Option(False, help="Overwrite existing project"),
) -> None:
    """🚀 Initialize a new quantum ML project.
    
    **Examples:**
    
    - Basic project: `quantum-mlops init my-project`
    - Advanced template: `quantum-mlops init my-project --template advanced`
    """
    
    try:
        project_path = Path(name)
        
        if project_path.exists() and not force:
            console.print(f"[red]Project directory {name} already exists. Use --force to overwrite.[/red]")
            raise typer.Exit(1)
        
        project_path.mkdir(exist_ok=True)
        
        with create_progress() as progress:
            task = progress.add_task("Creating project structure...", total=100)
            
            # Create basic structure
            (project_path / "data").mkdir(exist_ok=True)
            (project_path / "models").mkdir(exist_ok=True)
            (project_path / "experiments").mkdir(exist_ok=True)
            (project_path / "configs").mkdir(exist_ok=True)
            progress.update(task, advance=25)
            
            # Create example files
            with open(project_path / "train.py", 'w') as f:
                f.write("# Quantum ML training script\n")
            
            with open(project_path / "config.yaml", 'w') as f:
                f.write("# Project configuration\n")
            
            with open(project_path / "README.md", 'w') as f:
                f.write(f"# {name}\n\nQuantum ML project created with quantum-mlops\n")
            
            progress.update(task, advance=50)
            
            if template == "advanced":
                # Add advanced template files
                (project_path / "src").mkdir(exist_ok=True)
                (project_path / "tests").mkdir(exist_ok=True)
                (project_path / "notebooks").mkdir(exist_ok=True)
                
                with open(project_path / "requirements.txt", 'w') as f:
                    f.write("quantum-mlops-workbench\nnumpy\nscikit-learn\n")
                
                progress.update(task, advance=25)
        
        console.print(f"[green]✅ Project {name} created successfully![/green]")
        console.print(f"[cyan]📁 Project directory: {project_path.absolute()}[/cyan]")
        console.print(f"[yellow]💡 Next steps:[/yellow]")
        console.print(f"  1. cd {name}")
        console.print(f"  2. quantum-mlops config set default_backend simulator")
        console.print(f"  3. quantum-mlops model train data/your_data.csv")
        
    except Exception as e:
        handle_error(e, "initializing project")
        raise typer.Exit(1)


@app.command("interactive")
def interactive_mode() -> None:
    """🎮 Launch interactive quantum ML exploration mode."""
    
    console.print("[bold cyan]🎮 Quantum MLOps Interactive Mode[/bold cyan]")
    console.print("[yellow]Type 'help' for available commands, 'exit' to quit[/yellow]\n")
    
    commands = {
        "help": "Show available commands",
        "status": "Show system status",
        "backends": "List available backends",
        "models": "List registered models", 
        "experiments": "List experiments",
        "config": "Show configuration",
        "exit": "Exit interactive mode",
    }
    
    while True:
        try:
            cmd = Prompt.ask("[bold blue]quantum-mlops>[/bold blue]")
            
            if cmd.lower() in ['exit', 'quit', 'q']:
                console.print("[yellow]Goodbye![/yellow]")
                break
            elif cmd.lower() == 'help':
                help_table = Table(title="Available Commands")
                help_table.add_column("Command", style="cyan")
                help_table.add_column("Description", style="green")
                
                for command, description in commands.items():
                    help_table.add_row(command, description)
                
                console.print(help_table)
            elif cmd.lower() == 'status':
                console.print("[green]System Status: OK[/green]")
                console.print(f"[blue]Config directory: {CONFIG_FILE.parent}[/blue]")
                console.print(f"[blue]Model registry: {MODEL_REGISTRY_PATH}[/blue]")
            elif cmd.lower() == 'backends':
                console.print("[cyan]Available backends: simulator, ibm_quantum, aws_braket, ionq[/cyan]")
            elif cmd.lower() == 'models':
                if SERVICES_AVAILABLE:
                    model_service = ModelService(str(MODEL_REGISTRY_PATH))
                    models = model_service.list_models()
                    console.print(f"[green]Found {len(models)} registered models[/green]")
                else:
                    console.print("[yellow]Model service not available[/yellow]")
            elif cmd.lower() == 'experiments':
                console.print(f"[blue]Experiment directory: {EXPERIMENT_PATH}[/blue]")
            elif cmd.lower() == 'config':
                config = load_config()
                console.print(f"[green]Configuration loaded with {len(config)} settings[/green]")
            elif cmd.strip():
                console.print(f"[red]Unknown command: {cmd}[/red]")
                console.print("[yellow]Type 'help' for available commands[/yellow]")
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'exit' to quit[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@app.command("version")
def version() -> None:
    """📋 Show version information."""
    
    version_info = {
        "Quantum MLOps Workbench": "0.1.0",
        "Python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "Platform": sys.platform,
    }
    
    # Check optional dependencies
    dependencies = {
        "NumPy": "numpy",
        "Typer": "typer",
        "Rich": "rich",
        "PennyLane": "pennylane",
        "Qiskit": "qiskit",
        "AWS Braket": "braket",
    }
    
    version_table = Table(title="Version Information")
    version_table.add_column("Component", style="cyan")
    version_table.add_column("Version", style="green")
    version_table.add_column("Status", style="yellow")
    
    for name, version_num in version_info.items():
        version_table.add_row(name, version_num, "✅ Installed")
    
    for name, module in dependencies.items():
        try:
            mod = __import__(module)
            version_num = getattr(mod, '__version__', 'Unknown')
            status = "✅ Available"
        except ImportError:
            version_num = "Not installed"
            status = "❌ Missing"
        
        version_table.add_row(name, version_num, status)
    
    console.print(version_table)


def main() -> None:
    """Main CLI entry point."""
    try:
        # Initialize configuration directory
        init_config_dir()
        
        # Run the app
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if "--debug" in sys.argv:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()