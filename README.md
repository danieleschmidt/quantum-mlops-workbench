# Quantum MLOps Workbench

**Experiment tracking, circuit versioning, and reproducibility for hybrid quantum-classical machine learning.**

No external dependencies. Pure Python stdlib + SQLite.

---

## What's Inside

| Module | Purpose |
|---|---|
| `QMLExperiment` | Track a quantum ML run — circuit config, classical layers, hyperparams, metrics |
| `ExperimentRegistry` | SQLite-backed store — save, search, filter, rank experiments |
| `CircuitVersioning` | Tag and diff quantum circuits by gate sequence; walk lineage |
| `ReproducibilityChecker` | Verify two experiments are reproducible — seed, config, metric tolerance |
| `QMLDashboard` | Render text comparison tables and leaderboards |

---

## Quick Start

```python
from quantum_mlops import (
    QMLExperiment, ExperimentRegistry,
    CircuitVersioning, ReproducibilityChecker, QMLDashboard,
)
from quantum_mlops.experiment import CircuitSpec, ClassicalSpec

# Define a hybrid quantum-classical experiment
exp = QMLExperiment(
    name="vqe-ansatz-v1",
    circuit=CircuitSpec(
        n_qubits=4,
        depth=3,
        gate_sequence=["H(0)", "CNOT(0,1)", "RY(1,θ)", "CNOT(1,2)", "RZ(2,θ)"],
        entanglement="linear",
    ),
    classical=ClassicalSpec(layer_sizes=[64, 32, 10]),
    hyperparams={"lr": 0.01, "epochs": 100, "optimizer": "adam"},
    random_seed=42,
    tags=["vqe", "ansatz"],
)

# Run, log metrics
exp.start()
exp.complete({"val_accuracy": 0.87, "val_loss": 0.41})

# Persist
registry = ExperimentRegistry("experiments.db")
registry.save(exp)

# Find the top 3 by accuracy
best = registry.top_k(k=3, metric="val_accuracy")
```

---

## Experiment Tracking

`QMLExperiment` captures the full spec of a quantum ML run:

- **Circuit**: qubit count, depth, gate sequence, entanglement topology
- **Classical layers**: sizes, activation functions
- **Hyperparams**: learning rate, optimizer, epochs, etc.
- **Metrics**: logged during and after training
- **Status lifecycle**: `pending → running → completed / failed`

```python
exp.start()
exp.log_metric("train_loss", 0.52)
exp.complete({"val_accuracy": 0.91, "val_loss": 0.38})
```

---

## Experiment Registry

SQLite-backed registry with search, filter, and ranking:

```python
reg = ExperimentRegistry("experiments.db")

# Search
reg.search(n_qubits=4, status="completed")
reg.search(tag="vqe", min_metric={"val_accuracy": 0.85})

# Rank
reg.top_k(k=5, metric="val_accuracy")

# Export
reg.export_jsonl("experiments.jsonl")
```

---

## Circuit Versioning

Track and diff circuit evolution across experiments:

```python
cv = CircuitVersioning("circuits.json")  # persisted

cv.tag("v1.0", n_qubits=4, gates=["H(0)", "CNOT(0,1)"], description="baseline")
cv.tag("v1.1", n_qubits=4, gates=["H(0)", "CNOT(0,1)", "RZ(1,θ)"], parent="v1.0")

diff = cv.diff("v1.0", "v1.1")
print(diff.summary())
# v1.0 → v1.1: +1 gates  [depth +1, sim=0.67]

# Walk lineage
history = cv.history("v1.1")  # [v1.1, v1.0]
```

---

## Reproducibility Checking

Verify that two experiments are truly reproducible:

```python
checker = ReproducibilityChecker(registry, tolerance=0.01)
report = checker.check(exp_id_a, exp_id_b)
print(report.summary())

# ✅ REPRODUCIBLE  [abc123] vs [def456]
#   Seed match       : ✓
#   Config match     : ✓
#   Metrics (tol=0.010): ✓
```

Three sub-checks:
1. **Seed match** — same `random_seed`
2. **Config match** — identical circuit spec, classical spec, hyperparams
3. **Metrics within tolerance** — all shared metrics agree within `tolerance`

---

## Dashboard

Text-based experiment comparison (no dependencies):

```python
dash = QMLDashboard(primary_metric="val_accuracy")
print(dash.comparison_table(experiments))
print(dash.leaderboard(experiments, top_k=3))
print(dash.summary_card(best_exp))
```

---

## Run the Demo

```bash
python demo.py
```

Logs 5 experiments with different circuit configs, diffs circuits, runs reproducibility checks, and prints a full comparison table + leaderboard.

---

## Run Tests

```bash
pytest tests/ -v
# 38 passed
```

---

## Design Principles

- **Stdlib only** — no numpy, qiskit, or pennylane required; integrates with any quantum framework
- **Framework-agnostic** — circuit specs are strings; bring your own gate set
- **Reproducibility first** — random seed is a first-class field, not an afterthought
- **SQL for queries** — SQLite gives you real filtering without a server

---

## Project Structure

```
src/quantum_mlops/
├── experiment.py          # QMLExperiment, CircuitSpec, ClassicalSpec
├── registry.py            # ExperimentRegistry (SQLite)
├── circuit_versioning.py  # CircuitVersioning, CircuitDiff
├── reproducibility.py     # ReproducibilityChecker, ReproducibilityReport
└── dashboard.py           # QMLDashboard

tests/
└── test_workbench.py      # 38 tests

demo.py                    # end-to-end demo
```

---

## License

MIT
