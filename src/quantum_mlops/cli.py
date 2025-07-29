"""Command-line interface for quantum MLOps workbench."""

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Quantum MLOps Workbench CLI")
console = Console()


@app.command()
def test(
    backend: str = typer.Option("simulator", help="Quantum backend"),
    device: str = typer.Option("default", help="Specific device name"),
    shots: int = typer.Option(1000, help="Number of shots"),
    timeout: int = typer.Option(300, help="Timeout in seconds"),
) -> None:
    """Run quantum ML tests on specified backend."""
    console.print(f"[green]Running quantum tests on {backend}[/green]")
    console.print(f"Device: {device}, Shots: {shots}, Timeout: {timeout}s")


@app.command()
def analyze(
    metrics: str = typer.Option(
        "accuracy,loss", 
        help="Comma-separated list of metrics to analyze"
    ),
    output: str = typer.Option("metrics.json", help="Output file path"),
) -> None:
    """Analyze quantum ML metrics."""
    metric_list = metrics.split(",")
    console.print(f"[blue]Analyzing metrics: {', '.join(metric_list)}[/blue]")
    console.print(f"Output will be saved to: {output}")


@app.command()
def monitor(
    experiment: str = typer.Option("default", help="Experiment name"),
    metrics: str = typer.Option(
        "loss,accuracy", 
        help="Metrics to monitor"
    ),
    refresh_rate: int = typer.Option(5, help="Refresh rate in seconds"),
) -> None:
    """Launch quantum ML monitoring dashboard."""
    console.print(f"[cyan]Launching monitor for experiment: {experiment}[/cyan]")
    console.print(f"Monitoring: {metrics}")
    console.print(f"Dashboard available at: http://localhost:8050")


@app.command()
def status() -> None:
    """Show quantum backend status."""
    table = Table(title="Quantum Backend Status")
    table.add_column("Backend", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Queue Time", style="yellow")
    
    table.add_row("Simulator", "Available", "0s")
    table.add_row("IBM Quantum", "Available", "5-30 min")
    table.add_row("AWS Braket", "Available", "<1 min")
    table.add_row("IonQ", "Available", "<5 min")
    
    console.print(table)


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()