"""QMLDashboard — text-based experiment comparison and reporting."""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

from .experiment import QMLExperiment


class QMLDashboard:
    """
    Generates text-based experiment comparison tables.

    Usage
    -----
    dash = QMLDashboard(primary_metric="val_accuracy")
    print(dash.comparison_table(experiments))
    print(dash.summary_card(exp))
    """

    def __init__(
        self,
        primary_metric: str = "val_accuracy",
        higher_is_better: bool = True,
    ) -> None:
        self.primary_metric = primary_metric
        self.higher_is_better = higher_is_better

    # ── public ───────────────────────────────────────────────────────────────

    def comparison_table(
        self,
        experiments: List[QMLExperiment],
        extra_metrics: Optional[List[str]] = None,
        max_rows: int = 20,
    ) -> str:
        """
        Render a comparison table for a list of experiments.

        Columns: rank | id | name | qubits | depth | layers | seed | <metrics>
        Sorted by primary_metric (best first).
        """
        if not experiments:
            return "(no experiments to display)"

        # Determine which metrics to show
        all_metrics: List[str] = []
        for e in experiments:
            for m in e.metrics:
                if m not in all_metrics:
                    all_metrics.append(m)

        show_metrics = [self.primary_metric] + [
            m for m in all_metrics if m != self.primary_metric
        ]
        if extra_metrics:
            for m in extra_metrics:
                if m not in show_metrics:
                    show_metrics.append(m)

        # Sort
        ranked = sorted(
            experiments,
            key=lambda e: e.metrics.get(self.primary_metric, float("-inf")),
            reverse=self.higher_is_better,
        )[:max_rows]

        # Find best value per metric for highlighting
        best_vals: Dict[str, float] = {}
        for m in show_metrics:
            vals = [e.metrics[m] for e in ranked if m in e.metrics]
            if vals:
                best_vals[m] = max(vals) if self.higher_is_better else min(vals)

        # Build rows
        header = ["Rank", "ID", "Name", "Qubits", "Depth", "Layers", "Seed"] + show_metrics
        rows: List[List[str]] = []

        for rank, exp in enumerate(ranked, 1):
            metric_cells = []
            for m in show_metrics:
                val = exp.metrics.get(m)
                if val is None:
                    metric_cells.append("-")
                else:
                    cell = f"{val:.4f}"
                    if best_vals.get(m) == val:
                        cell = f"★{cell}"  # mark best
                    metric_cells.append(cell)

            rows.append([
                str(rank),
                exp.experiment_id,
                _truncate(exp.name, 24),
                str(exp.circuit.n_qubits),
                str(exp.circuit.depth),
                str(len(exp.classical.layer_sizes)),
                str(exp.random_seed),
                *metric_cells,
            ])

        return _render_table(header, rows, title="Experiment Comparison")

    def summary_card(self, exp: QMLExperiment) -> str:
        """Single-experiment summary card."""
        lines = [
            "┌─ Experiment Summary " + "─" * 40,
            f"│  ID      : {exp.experiment_id}",
            f"│  Name    : {exp.name}",
            f"│  Status  : {exp.status}",
            f"│  Seed    : {exp.random_seed}",
            f"│  Tags    : {', '.join(exp.tags) or 'none'}",
            "│",
            f"│  Circuit : {exp.circuit.n_qubits} qubits, depth {exp.circuit.depth}, {exp.circuit.entanglement}",
            f"│  Gates   : {len(exp.circuit.gate_sequence)} total",
            "│  Sequence: " + _truncate(str(exp.circuit.gate_sequence), 50),
            "│",
            f"│  Classical: {exp.classical.summary()} [{exp.classical.activation}]",
            "│",
            "│  Hyperparameters:",
        ]
        for k, v in exp.hyperparams.items():
            lines.append(f"│    {k} = {v}")
        lines.append("│")
        lines.append("│  Metrics:")
        for k, v in exp.metrics.items():
            marker = " ◄" if k == self.primary_metric else ""
            lines.append(f"│    {k} = {v:.6f}{marker}")
        lines.append("└" + "─" * 61)
        return "\n".join(lines)

    def leaderboard(
        self,
        experiments: List[QMLExperiment],
        top_k: int = 5,
    ) -> str:
        """Compact leaderboard showing top-k by primary metric."""
        ranked = sorted(
            [e for e in experiments if self.primary_metric in e.metrics],
            key=lambda e: e.metrics[self.primary_metric],
            reverse=self.higher_is_better,
        )[:top_k]

        if not ranked:
            return f"(no experiments with metric '{self.primary_metric}')"

        lines = [f"🏆 Leaderboard — {self.primary_metric} (top {top_k})"]
        lines.append("─" * 55)
        for i, exp in enumerate(ranked, 1):
            val = exp.metrics[self.primary_metric]
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"#{i} ")
            lines.append(
                f"{medal} [{exp.experiment_id}] {_truncate(exp.name, 22):<22}  "
                f"{self.primary_metric}={val:.4f}  "
                f"({exp.circuit.n_qubits}q d={exp.circuit.depth})"
            )
        return "\n".join(lines)


# ── helpers ───────────────────────────────────────────────────────────────────

def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _render_table(
    header: List[str],
    rows: List[List[str]],
    title: str = "",
) -> str:
    all_rows = [header] + rows
    col_widths = [
        max(len(str(r[i])) for r in all_rows)
        for i in range(len(header))
    ]

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_line = "| " + " | ".join(
        str(h).ljust(col_widths[i]) for i, h in enumerate(header)
    ) + " |"

    lines: List[str] = []
    if title:
        table_width = len(sep)
        lines.append("+" + f" {title} ".center(table_width - 2, "─") + "+")
    lines.append(sep)
    lines.append(header_line)
    lines.append(sep.replace("-", "=").replace("+", "|"))

    for row in rows:
        line = "| " + " | ".join(
            str(row[i]).ljust(col_widths[i]) if i < len(row) else " " * col_widths[i]
            for i in range(len(header))
        ) + " |"
        lines.append(line)

    lines.append(sep)
    return "\n".join(lines)
