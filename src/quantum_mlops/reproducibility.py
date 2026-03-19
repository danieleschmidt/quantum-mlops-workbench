"""ReproducibilityChecker — verify that two experiments are reproducible."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .experiment import QMLExperiment
from .registry import ExperimentRegistry


@dataclass
class ReproducibilityReport:
    """Result of a reproducibility check between two experiments."""
    id_a: str
    id_b: str
    is_reproducible: bool

    # sub-checks
    seed_match: bool
    config_match: bool
    metrics_within_tolerance: bool

    # detail
    config_diffs: Dict[str, Tuple[Any, Any]]   # key → (value_a, value_b)
    metric_diffs: Dict[str, float]              # metric → abs diff
    tolerance: float

    notes: List[str]

    def summary(self) -> str:
        verdict = "✅ REPRODUCIBLE" if self.is_reproducible else "❌ NOT REPRODUCIBLE"
        lines = [
            f"{verdict}  [{self.id_a}] vs [{self.id_b}]",
            f"  Seed match       : {'✓' if self.seed_match else '✗'}",
            f"  Config match     : {'✓' if self.config_match else '✗'}",
            f"  Metrics (tol={self.tolerance:.3f}): {'✓' if self.metrics_within_tolerance else '✗'}",
        ]
        if self.config_diffs:
            lines.append("  Config diffs:")
            for k, (a, b) in self.config_diffs.items():
                lines.append(f"    {k}: {a!r} → {b!r}")
        if self.metric_diffs:
            lines.append("  Metric diffs:")
            for m, d in self.metric_diffs.items():
                lines.append(f"    {m}: Δ={d:.6f}")
        if self.notes:
            lines.extend(f"  Note: {n}" for n in self.notes)
        return "\n".join(lines)


class ReproducibilityChecker:
    """
    Checks whether two experiments are reproducible.

    Reproducibility requires:
      1. Same random seed
      2. Same configuration (circuit spec + classical spec + hyperparams)
      3. All shared metrics agree within `tolerance`

    Usage
    -----
    checker = ReproducibilityChecker(registry, tolerance=0.01)
    report = checker.check("abc123", "def456")
    print(report.summary())
    """

    def __init__(
        self,
        registry: ExperimentRegistry,
        tolerance: float = 0.01,
    ) -> None:
        self.registry = registry
        self.tolerance = tolerance

    # ── public API ───────────────────────────────────────────────────────────

    def check(self, id_a: str, id_b: str) -> ReproducibilityReport:
        """Compare two experiments by ID."""
        exp_a = self._load(id_a)
        exp_b = self._load(id_b)
        return self._compare(exp_a, exp_b)

    def check_experiments(
        self, exp_a: QMLExperiment, exp_b: QMLExperiment
    ) -> ReproducibilityReport:
        """Compare two in-memory experiments directly."""
        return self._compare(exp_a, exp_b)

    # ── internals ────────────────────────────────────────────────────────────

    def _load(self, exp_id: str) -> QMLExperiment:
        exp = self.registry.get(exp_id)
        if exp is None:
            raise KeyError(f"Experiment '{exp_id}' not found in registry.")
        return exp

    def _compare(
        self, exp_a: QMLExperiment, exp_b: QMLExperiment
    ) -> ReproducibilityReport:
        notes: List[str] = []

        # 1. Seed
        seed_match = exp_a.random_seed == exp_b.random_seed
        if not seed_match:
            notes.append(
                f"Seeds differ: {exp_a.random_seed} vs {exp_b.random_seed}"
            )

        # 2. Config
        config_diffs: Dict[str, Tuple[Any, Any]] = {}
        self._diff_circuit(exp_a, exp_b, config_diffs)
        self._diff_classical(exp_a, exp_b, config_diffs)
        self._diff_hyperparams(exp_a, exp_b, config_diffs)
        config_match = len(config_diffs) == 0

        # 3. Metrics
        metric_diffs: Dict[str, float] = {}
        shared_metrics = set(exp_a.metrics) & set(exp_b.metrics)
        if not shared_metrics:
            notes.append("No shared metrics to compare.")
        for m in shared_metrics:
            delta = abs(exp_a.metrics[m] - exp_b.metrics[m])
            metric_diffs[m] = delta

        metrics_within_tolerance = all(
            d <= self.tolerance for d in metric_diffs.values()
        )
        if not metrics_within_tolerance:
            over = {m: d for m, d in metric_diffs.items() if d > self.tolerance}
            notes.append(
                f"Metrics outside tolerance ({self.tolerance}): "
                + ", ".join(f"{m}=Δ{d:.4f}" for m, d in over.items())
            )

        is_reproducible = seed_match and config_match and metrics_within_tolerance

        return ReproducibilityReport(
            id_a=exp_a.experiment_id,
            id_b=exp_b.experiment_id,
            is_reproducible=is_reproducible,
            seed_match=seed_match,
            config_match=config_match,
            metrics_within_tolerance=metrics_within_tolerance,
            config_diffs=config_diffs,
            metric_diffs=metric_diffs,
            tolerance=self.tolerance,
            notes=notes,
        )

    @staticmethod
    def _diff_circuit(
        a: QMLExperiment, b: QMLExperiment, out: Dict[str, Tuple[Any, Any]]
    ) -> None:
        ca, cb = a.circuit, b.circuit
        for attr in ("n_qubits", "depth", "gate_sequence", "entanglement"):
            va, vb = getattr(ca, attr), getattr(cb, attr)
            if va != vb:
                out[f"circuit.{attr}"] = (va, vb)

    @staticmethod
    def _diff_classical(
        a: QMLExperiment, b: QMLExperiment, out: Dict[str, Tuple[Any, Any]]
    ) -> None:
        ca, cb = a.classical, b.classical
        for attr in ("layer_sizes", "activation", "output_activation"):
            va, vb = getattr(ca, attr), getattr(cb, attr)
            if va != vb:
                out[f"classical.{attr}"] = (va, vb)

    @staticmethod
    def _diff_hyperparams(
        a: QMLExperiment, b: QMLExperiment, out: Dict[str, Tuple[Any, Any]]
    ) -> None:
        all_keys = set(a.hyperparams) | set(b.hyperparams)
        for k in all_keys:
            va = a.hyperparams.get(k)
            vb = b.hyperparams.get(k)
            if va != vb:
                out[f"hyperparams.{k}"] = (va, vb)
