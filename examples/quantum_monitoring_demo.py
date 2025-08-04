#!/usr/bin/env python3
"""
Quantum ML Monitoring Demo

This script demonstrates the comprehensive monitoring capabilities of the enhanced
quantum MLOps monitoring system, including real-time metrics tracking, visualization,
and alerting for quantum machine learning experiments.
"""

import numpy as np
import time
from pathlib import Path

# Import quantum monitoring components
from quantum_mlops.monitoring import (
    QuantumMonitor,
    QuantumMetricsCalculator,
    QuantumVisualization,
    create_quantum_monitor
)
from quantum_mlops.core import QuantumMLPipeline, QuantumDevice


def simulate_quantum_training_step(step: int, n_qubits: int = 4):
    """Simulate a quantum training step with realistic metrics."""
    
    # Simulate parameter updates
    n_params = 2 * n_qubits * 3  # Rough estimate for variational circuit
    parameters = np.random.uniform(-np.pi, np.pi, n_params)
    
    # Add some parameter drift over time
    drift = 0.1 * np.sin(step * 0.1) * np.random.normal(0, 0.05, n_params)
    parameters += drift
    
    # Simulate gradients with occasional spikes
    gradients = np.random.normal(0, 0.1, n_params)
    if step % 50 == 0:  # Occasional gradient spike
        gradients *= 5.0
    
    # Simulate loss with convergence behavior
    base_loss = 1.0 * np.exp(-step * 0.01) + 0.1
    noise = np.random.normal(0, 0.05)
    loss = max(0.01, base_loss + noise)
    
    # Simulate quantum state
    state_dim = 2 ** n_qubits
    state_vector = np.random.complex128(state_dim)
    state_vector = state_vector / np.linalg.norm(state_vector)  # Normalize
    
    # Add some structure to the state evolution
    phase_evolution = np.exp(1j * step * 0.1)
    state_vector *= phase_evolution
    
    return {
        'parameters': parameters,
        'gradients': gradients,
        'loss': loss,
        'state_vector': state_vector
    }


def simulate_circuit_execution(step: int):
    """Simulate quantum circuit execution with realistic timing and costs."""
    
    # Simulate execution metrics
    base_time = 0.5  # Base execution time in seconds
    queue_factor = np.random.exponential(0.2)  # Queue time variability
    execution_time = base_time + np.random.normal(0, 0.1)
    queue_time = queue_factor * 10  # Queue time in seconds
    
    # Simulate cost (higher for real hardware)
    cost_per_shot = 0.001  # Simulated cost per shot
    shots = 1024
    cost = cost_per_shot * shots + np.random.normal(0, 0.0001)
    
    # Circuit description
    circuit_description = {
        'gates': [
            {'type': 'ry', 'qubit': 0, 'angle': np.random.uniform(0, 2*np.pi)},
            {'type': 'ry', 'qubit': 1, 'angle': np.random.uniform(0, 2*np.pi)},
            {'type': 'cnot', 'control': 0, 'target': 1},
            {'type': 'ry', 'qubit': 2, 'angle': np.random.uniform(0, 2*np.pi)},
            {'type': 'cnot', 'control': 1, 'target': 2},
        ],
        'n_qubits': 4,
        'measurements': [{'type': 'expectation', 'wires': 0, 'observable': 'Z'}]
    }
    
    backend_info = {
        'name': 'simulator' if step % 3 != 0 else 'ibm_quantum',
        'status': 'online',
        'queue_length': max(0, int(np.random.normal(5, 2)))
    }
    
    return {
        'circuit_description': circuit_description,
        'execution_time': execution_time,
        'queue_time': queue_time,
        'cost': cost,
        'backend_info': backend_info
    }


def run_comprehensive_monitoring_demo():
    """Run comprehensive quantum monitoring demonstration."""
    
    print("🚀 Starting Quantum ML Monitoring Demo")
    print("=" * 60)
    
    # Create quantum monitor with comprehensive configuration
    monitor = create_quantum_monitor(
        experiment_name="quantum_vqe_demo",
        enable_mlflow=True,  # Will gracefully handle if MLflow not available
        enable_wandb=False,  # Disable W&B for demo to avoid API key requirements
        storage_path="./demo_monitoring_data",
        alert_config={
            'fidelity_drop': 0.1,
            'gradient_explosion': 8.0,
            'queue_time_limit': 120.0,
            'cost_spike': 3.0
        }
    )
    
    try:
        # Start monitoring run
        with monitor.start_run("vqe_optimization_demo"):
            print("📊 Monitoring run started")
            
            # Start real-time monitoring
            monitor.start_realtime_monitoring(update_interval=2.0)
            print("⚡ Real-time monitoring activated")
            
            # Simulate quantum training with comprehensive logging
            n_steps = 100
            n_qubits = 4
            
            print(f"\n🔬 Simulating {n_steps} training steps with {n_qubits} qubits")
            print("-" * 50)
            
            for step in range(n_steps):
                # Simulate training step
                training_data = simulate_quantum_training_step(step, n_qubits)
                
                # Log training metrics
                monitor.log_training_step(
                    loss=training_data['loss'],
                    gradients=training_data['gradients'],
                    parameters=training_data['parameters'],
                    step=step,
                    additional_metrics={
                        'learning_rate': 0.01 * (0.95 ** (step // 10)),
                        'batch_size': 32,
                        'epoch': step // 10
                    }
                )
                
                # Log quantum state information
                monitor.log_quantum_state(
                    state_vector=training_data['state_vector'],
                    step=step
                )
                
                # Simulate and log circuit execution (every few steps)
                if step % 5 == 0:
                    exec_data = simulate_circuit_execution(step)
                    monitor.log_circuit_execution(
                        circuit_description=exec_data['circuit_description'],
                        execution_time=exec_data['execution_time'],
                        queue_time=exec_data['queue_time'],
                        cost=exec_data['cost'],
                        backend_info=exec_data['backend_info']
                    )
                
                # Log some additional quantum-specific metrics
                if step % 10 == 0:
                    # Calculate quantum volume
                    qv = monitor.metrics_calculator.calculate_quantum_volume(
                        n_qubits=n_qubits,
                        circuit_depth=5,
                        gate_fidelity=0.99 - step * 0.0001  # Slight degradation over time
                    )
                    
                    monitor.log_metrics({
                        'quantum_volume': qv,
                        'hardware_fidelity': 0.99 - step * 0.0001,
                        'coherence_time': 100 - step * 0.1,  # Microseconds
                        'error_rate': 0.001 + step * 0.00001
                    }, step=step)
                
                # Progress indication
                if step % 20 == 0:
                    print(f"   Step {step:3d}/{n_steps}: Loss = {training_data['loss']:.4f}")
                
                # Small delay to make monitoring more realistic
                time.sleep(0.05)
            
            print("\n✅ Training simulation completed")
            
            # Generate comprehensive summary
            print("\n📈 Generating comprehensive summary...")
            summary = monitor.get_comprehensive_summary()
            
            print("\n📊 Experiment Summary:")
            print(f"   Duration: {summary['experiment_info']['duration']:.2f} seconds")
            print(f"   Training Steps: {summary['training_summary']['total_steps']}")
            print(f"   Final Loss: {summary['training_summary']['final_loss']:.6f}")
            print(f"   Best Loss: {summary['training_summary']['best_loss']:.6f}")
            print(f"   States Logged: {summary['quantum_summary']['states_logged']}")
            
            if summary['quantum_summary']['avg_fidelity']:
                print(f"   Avg Fidelity: {summary['quantum_summary']['avg_fidelity']:.4f}")
                print(f"   Avg Entanglement: {summary['quantum_summary']['avg_entanglement']:.4f}")
            
            print(f"   Total Cost: ${summary['execution_summary']['total_cost']:.4f}")
            print(f"   Backends Used: {summary['execution_summary']['backends_used']}")
            
            # Alert summary
            alert_summary = summary['alert_summary']
            if alert_summary['total_alerts'] > 0:
                print(f"\n⚠️  Alerts Generated: {alert_summary['total_alerts']}")
                for alert_type, count in alert_summary['alert_types'].items():
                    print(f"   {alert_type}: {count}")
            else:
                print("\n✅ No alerts triggered during experiment")
            
            # Create visualizations and dashboard
            print("\n🎨 Creating visualizations...")
            try:
                dashboard_path = monitor.create_static_dashboard()
                if dashboard_path:
                    print(f"   Dashboard saved: {dashboard_path}")
            except Exception as e:
                print(f"   Dashboard creation failed: {e}")
            
            # Generate comprehensive report
            print("\n📄 Generating monitoring report...")
            try:
                report_path = monitor.generate_report(include_visualizations=True)
                if report_path:
                    print(f"   Report saved: {report_path}")
            except Exception as e:
                print(f"   Report generation failed: {e}")
            
            # Export metrics in different formats
            print("\n💾 Exporting metrics...")
            try:
                # Export as JSON
                monitor.export_metrics("demo_metrics.json", format="json")
                print("   Metrics exported as JSON")
                
                # Export as CSV
                monitor.export_metrics("demo_metrics.csv", format="csv")
                print("   Metrics exported as CSV")
                
                # Export comprehensive data
                monitor.export_metrics("demo_comprehensive.json", format="comprehensive")
                print("   Comprehensive data exported")
                
            except Exception as e:
                print(f"   Export failed: {e}")
    
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ensure monitoring is stopped
        monitor.stop_realtime_monitoring()
        print("\n🛑 Real-time monitoring stopped")
    
    print("\n🎉 Quantum ML Monitoring Demo completed!")
    print("=" * 60)


def demonstrate_visualization_features():
    """Demonstrate quantum-specific visualization features."""
    
    print("\n🎨 Quantum Visualization Demo")
    print("-" * 40)
    
    try:
        # Create single qubit states for Bloch sphere demo
        print("📊 Creating Bloch sphere visualizations...")
        
        # |0⟩ state
        state_0 = np.array([1.0, 0.0], dtype=complex)
        QuantumVisualization.plot_bloch_sphere(state_0, "bloch_state_0.png")
        
        # |+⟩ state (superposition)
        state_plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        QuantumVisualization.plot_bloch_sphere(state_plus, "bloch_state_plus.png")
        
        # |i⟩ state
        state_i = np.array([1.0, 1j], dtype=complex) / np.sqrt(2)
        QuantumVisualization.plot_bloch_sphere(state_i, "bloch_state_i.png")
        
        print("   Bloch sphere plots saved")
        
        # Create state evolution plot
        print("📈 Creating state evolution visualization...")
        
        # Simulate evolving 2-qubit state
        n_steps = 50
        state_evolution = []
        
        for step in range(n_steps):
            # Evolving superposition state
            angle = step * 0.1
            state = np.array([
                np.cos(angle) + 0.1j * np.sin(angle),
                0.2 * np.sin(angle),
                0.3 * np.cos(2 * angle),
                0.4 * np.sin(2 * angle) + 0.1j * np.cos(angle)
            ], dtype=complex)
            state = state / np.linalg.norm(state)
            state_evolution.append(state)
        
        QuantumVisualization.plot_state_evolution(state_evolution, "state_evolution.png")
        print("   State evolution plot saved")
        
    except Exception as e:
        print(f"   Visualization demo failed: {e}")


def demonstrate_metrics_calculator():
    """Demonstrate quantum metrics calculation features."""
    
    print("\n🔬 Quantum Metrics Calculator Demo")
    print("-" * 40)
    
    calc = QuantumMetricsCalculator()
    
    # Demonstrate fidelity calculation
    print("📊 Fidelity Calculation:")
    state1 = np.array([1.0, 0.0], dtype=complex)  # |0⟩
    state2 = np.array([0.0, 1.0], dtype=complex)  # |1⟩
    state3 = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)  # |+⟩
    
    fidelity_01 = calc.calculate_fidelity(state1, state2)
    fidelity_0plus = calc.calculate_fidelity(state1, state3)
    
    print(f"   |0⟩ vs |1⟩: {fidelity_01:.4f}")
    print(f"   |0⟩ vs |+⟩: {fidelity_0plus:.4f}")
    
    # Demonstrate entanglement calculation
    print("\n🔗 Entanglement Entropy Calculation:")
    
    # Bell state |00⟩ + |11⟩
    bell_state = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2)
    entanglement_bell = calc.calculate_entanglement_entropy(bell_state)
    
    # Product state |00⟩
    product_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    entanglement_product = calc.calculate_entanglement_entropy(product_state)
    
    print(f"   Bell state: {entanglement_bell:.4f}")
    print(f"   Product state: {entanglement_product:.4f}")
    
    # Demonstrate circuit depth calculation
    print("\n🔧 Circuit Depth Calculation:")
    
    # Simple circuit
    simple_circuit = [
        {'type': 'ry', 'qubit': 0, 'angle': 0.5},
        {'type': 'cnot', 'control': 0, 'target': 1},
        {'type': 'ry', 'qubit': 1, 'angle': 0.3}
    ]
    
    depth = calc.calculate_circuit_depth(simple_circuit)
    print(f"   Simple circuit depth: {depth}")
    
    # Complex circuit with parallel gates
    complex_circuit = [
        {'type': 'ry', 'qubit': 0, 'angle': 0.5},
        {'type': 'ry', 'qubit': 1, 'angle': 0.3},  # Parallel with above
        {'type': 'cnot', 'control': 0, 'target': 1},
        {'type': 'ry', 'qubit': 2, 'angle': 0.7},  # Parallel with CNOT
        {'type': 'cnot', 'control': 1, 'target': 2}
    ]
    
    complex_depth = calc.calculate_circuit_depth(complex_circuit)
    print(f"   Complex circuit depth: {complex_depth}")


if __name__ == "__main__":
    print("🔬 Quantum MLOps Monitoring System Demo")
    print("=" * 60)
    
    # Run comprehensive monitoring demo
    run_comprehensive_monitoring_demo()
    
    # Demonstrate additional features
    demonstrate_visualization_features()
    demonstrate_metrics_calculator()
    
    print(f"\n📁 Demo files saved in: {Path.cwd()}")
    print("🎯 Check the generated monitoring data, reports, and visualizations!")