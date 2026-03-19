"""QMLExperiment — tracks a single quantum ML experiment."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class CircuitSpec:
    """Specification of the quantum circuit component."""
    n_qubits: int
    depth: int                        # number of layers
    gate_sequence: List[str]          # e.g. ["H(0)", "CNOT(0,1)", "RY(1,θ)"]
    entanglement: str = "linear"      # linear | full | circular
    
    def summary(self) -> str:
        return f"{self.n_qubits}q depth={self.depth} [{self.entanglement}]"


@dataclass
class ClassicalSpec:
    """Specification of the classical layers in a hybrid model."""
    layer_sizes: List[int]            # e.g. [64, 32, 10]
    activation: str = "relu"
    output_activation: str = "softmax"

    def summary(self) -> str:
        return "->".join(str(s) for s in self.layer_sizes)


@dataclass
class QMLExperiment:
    """
    Tracks a quantum ML experiment end-to-end.

    Parameters
    ----------
    name : experiment label
    circuit : quantum circuit specification
    classical : classical network specification
    hyperparams : training hyperparameters (lr, epochs, batch_size, optimizer, …)
    random_seed : for reproducibility
    tags : arbitrary labels
    """

    name: str
    circuit: CircuitSpec
    classical: ClassicalSpec
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    random_seed: int = 42
    tags: List[str] = field(default_factory=list)

    # set automatically
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: float = field(default_factory=time.time)
    status: str = "pending"           # pending | running | completed | failed

    # logged during / after training
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Mark experiment as running."""
        self.status = "running"
        self.metadata["started_at"] = time.time()

    def log_metric(self, name: str, value: float) -> None:
        """Log a single metric."""
        self.metrics[name] = value

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log multiple metrics at once."""
        self.metrics.update(metrics)

    def complete(self, metrics: Optional[Dict[str, float]] = None) -> None:
        """Mark experiment complete, optionally logging final metrics."""
        if metrics:
            self.log_metrics(metrics)
        self.status = "completed"
        self.metadata["completed_at"] = time.time()

    def fail(self, reason: str = "") -> None:
        self.status = "failed"
        self.metadata["failure_reason"] = reason

    # ── serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QMLExperiment":
        circuit = CircuitSpec(**data.pop("circuit"))
        classical = ClassicalSpec(**data.pop("classical"))
        return cls(circuit=circuit, classical=classical, **data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    # ── display ──────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return (
            f"[{self.experiment_id}] {self.name}  ({self.status})\n"
            f"  Circuit : {self.circuit.summary()}\n"
            f"  Classical: {self.classical.summary()}\n"
            f"  Seed    : {self.random_seed}\n"
            f"  Metrics : {metric_str or 'none'}\n"
            f"  Tags    : {', '.join(self.tags) or 'none'}"
        )
