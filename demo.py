#!/usr/bin/env python3
"""
Quantum MLOps Workbench — Demo
==============================
Logs 5 experiments with different quantum circuit configurations,
demonstrates circuit versioning, reproducibility checking, and
renders a comparison dashboard.
"""

import random
import sys
import tempfile
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from quantum_mlops import (
    QMLExperiment,
    ExperimentRegistry,
    CircuitVersioning,
    ReproducibilityChecker,
    QMLDashboard,
)
from quantum_mlops.experiment import CircuitSpec, ClassicalSpec


# ── 1. Define experiment configs ─────────────────────────────────────────────

def make_gates(n_qubits: int, depth: int, seed: int) -> list:
    """Simulate a variational ansatz gate sequence."""
    rng = random.Random(seed)
    single = ["H", "X", "Y", "Z", "RX", "RY", "RZ", "S", "T"]
    two_q  = ["CNOT", "CZ", "SWAP"]
    gates = []
    for layer in range(depth):
        for q in range(n_qubits):
            g = rng.choice(single)
            gates.append(f"{g}({q},θ_{layer}_{q})" if g.startswith("R") else f"{g}({q})")
        # entanglement
        for q in range(n_qubits - 1):
            gates.append(f"{rng.choice(two_q)}({q},{q+1})")
    return gates


CONFIGS = [
    dict(name="shallow-4q",   n_qubits=4, depth=2, layers=[64, 32, 10], lr=0.01, epochs=50, seed=42,  tags=["baseline"]),
    dict(name="deep-4q",      n_qubits=4, depth=6, layers=[128, 64, 10], lr=0.005, epochs=100, seed=7,   tags=["deep"]),
    dict(name="wide-8q",      n_qubits=8, depth=3, layers=[256, 128, 10], lr=0.01, epochs=75, seed=99,  tags=["wide"]),
    dict(name="dropout-6q",   n_qubits=6, depth=4, layers=[64, 64, 10], lr=0.001, epochs=200, seed=123, tags=["regularised"]),
    dict(name="repr-shallow-4q", n_qubits=4, depth=2, layers=[64, 32, 10], lr=0.01, epochs=50, seed=42,  tags=["baseline", "repro"]),
]


def simulate_metrics(cfg: dict, seed: int) -> dict:
    """Deterministic fake training result so the demo is reproducible."""
    rng = random.Random(seed + cfg["n_qubits"] * 100 + cfg["depth"])
    base_acc = 0.6 + 0.1 * (cfg["depth"] / 6) + 0.05 * (cfg["n_qubits"] / 8)
    noise     = rng.gauss(0, 0.005)
    acc       = min(0.99, max(0.50, base_acc + noise))
    loss      = -round((acc * 0.8 + rng.gauss(0, 0.01)), 4)
    f1        = round(acc - rng.gauss(0, 0.01), 4)
    return {
        "val_accuracy": round(acc, 4),
        "val_loss":     abs(loss),
        "val_f1":       max(0, f1),
        "train_loss":   abs(loss) - 0.02,
    }


# ── 2. Run experiments ────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  Quantum MLOps Workbench — Demo Run")
print("=" * 60 + "\n")

with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
    db_path = f.name

registry = ExperimentRegistry(db_path)

experiments = []
for cfg in CONFIGS:
    gates = make_gates(cfg["n_qubits"], cfg["depth"], cfg["seed"])
    circuit = CircuitSpec(
        n_qubits=cfg["n_qubits"],
        depth=cfg["depth"],
        gate_sequence=gates,
        entanglement="linear",
    )
    classical = ClassicalSpec(layer_sizes=cfg["layers"])
    exp = QMLExperiment(
        name=cfg["name"],
        circuit=circuit,
        classical=classical,
        hyperparams={"lr": cfg["lr"], "epochs": cfg["epochs"], "optimizer": "adam"},
        random_seed=cfg["seed"],
        tags=cfg["tags"],
    )
    exp.start()
    metrics = simulate_metrics(cfg, cfg["seed"])
    exp.complete(metrics)
    registry.save(exp)
    experiments.append(exp)
    print(f"  ✓ Logged: [{exp.experiment_id}] {exp.name}  acc={metrics['val_accuracy']:.4f}")

print(f"\n  Total experiments in registry: {registry.count()}\n")


# ── 3. Circuit Versioning ─────────────────────────────────────────────────────

print("─" * 60)
print("  Circuit Versioning")
print("─" * 60)

cv = CircuitVersioning()
for i, exp in enumerate(experiments[:3]):
    tag = f"v{i+1}.0-{exp.name}"
    cv.tag(tag, exp.circuit.n_qubits, exp.circuit.gate_sequence,
           description=f"Circuit for {exp.name}", author="Daniel")
    print(f"  Tagged: {tag}  ({len(exp.circuit.gate_sequence)} gates)")

# diff first two
t0, t1 = cv.tags()[0], cv.tags()[1]
diff = cv.diff(t0, t1)
print(f"\n  Diff: {diff.summary()}")

# diff first vs last
t2 = cv.tags()[2]
diff2 = cv.diff(t0, t2)
print(f"  Diff: {diff2.summary()}")

# history
hist = cv.history(t1)
print(f"\n  History for '{t1}': {[v.tag for v in hist]}")


# ── 4. Reproducibility ────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("  Reproducibility Checks")
print("─" * 60)

checker = ReproducibilityChecker(registry, tolerance=0.02)

# same seed / same config → should be reproducible
exp_a = experiments[0]   # shallow-4q  seed=42
exp_b = experiments[-1]  # repr-shallow-4q  seed=42

report1 = checker.check_experiments(exp_a, exp_b)
print(f"\n  Check A (same config):\n{report1.summary()}")

# different config → should NOT be reproducible
report2 = checker.check_experiments(experiments[0], experiments[1])
print(f"\n  Check B (different configs):\n{report2.summary()}")


# ── 5. Dashboard ──────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("  Experiment Dashboard")
print("─" * 60 + "\n")

dash = QMLDashboard(primary_metric="val_accuracy")

# Full comparison table
print(dash.comparison_table(experiments))

# Leaderboard
print()
print(dash.leaderboard(experiments, top_k=3))

# Summary card for best experiment
best = registry.top_k(k=1, metric="val_accuracy")[0]
print()
print(dash.summary_card(best))


# ── 6. Find best 3 ───────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("  Top 3 Experiments")
print("─" * 60)

top3 = registry.top_k(k=3, metric="val_accuracy")
for i, exp in enumerate(top3, 1):
    acc = exp.metrics.get("val_accuracy", 0)
    print(f"  #{i}  [{exp.experiment_id}] {exp.name:<22}  "
          f"acc={acc:.4f}  ({exp.circuit.n_qubits}q depth={exp.circuit.depth})")


# ── Cleanup ───────────────────────────────────────────────────────────────────

os.unlink(db_path)

print("\n" + "=" * 60)
print("  Demo complete. ✓")
print("=" * 60 + "\n")
