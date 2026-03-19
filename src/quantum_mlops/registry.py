"""ExperimentRegistry — SQLite-backed registry with search and filter."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .experiment import QMLExperiment


_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    status        TEXT NOT NULL,
    n_qubits      INTEGER,
    circuit_depth INTEGER,
    n_classical   INTEGER,
    random_seed   INTEGER,
    tags          TEXT,      -- JSON array
    metrics       TEXT,      -- JSON object
    data          TEXT NOT NULL,  -- full JSON blob
    created_at    REAL
);

CREATE INDEX IF NOT EXISTS idx_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_name   ON experiments(name);
CREATE INDEX IF NOT EXISTS idx_qubits ON experiments(n_qubits);
"""


class ExperimentRegistry:
    """
    SQLite-backed store for QMLExperiment objects.

    Usage
    -----
    reg = ExperimentRegistry("experiments.db")
    reg.save(exp)
    results = reg.search(n_qubits=4, status="completed")
    best = reg.top_k(k=3, metric="val_accuracy")
    """

    def __init__(self, db_path: str | Path = "experiments.db") -> None:
        self.db_path = Path(db_path)
        self._init_db()

    # ── internals ────────────────────────────────────────────────────────────

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _row_to_exp(self, row: sqlite3.Row) -> QMLExperiment:
        return QMLExperiment.from_dict(json.loads(row["data"]))

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def save(self, exp: QMLExperiment) -> None:
        """Insert or replace an experiment."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments
                    (experiment_id, name, status, n_qubits, circuit_depth,
                     n_classical, random_seed, tags, metrics, data, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exp.experiment_id,
                    exp.name,
                    exp.status,
                    exp.circuit.n_qubits,
                    exp.circuit.depth,
                    len(exp.classical.layer_sizes),
                    exp.random_seed,
                    json.dumps(exp.tags),
                    json.dumps(exp.metrics),
                    exp.to_json(),
                    exp.created_at,
                ),
            )

    def get(self, experiment_id: str) -> Optional[QMLExperiment]:
        """Fetch a single experiment by ID. Returns None if not found."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT data FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
        return self._row_to_exp(row) if row else None

    def delete(self, experiment_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            )
        return cur.rowcount > 0

    # ── search / filter ──────────────────────────────────────────────────────

    def search(
        self,
        name: Optional[str] = None,
        status: Optional[str] = None,
        n_qubits: Optional[int] = None,
        circuit_depth: Optional[int] = None,
        tag: Optional[str] = None,
        min_metric: Optional[Dict[str, float]] = None,
        predicate: Optional[Callable[[QMLExperiment], bool]] = None,
    ) -> List[QMLExperiment]:
        """
        Filter experiments by column values or an arbitrary predicate.

        Parameters
        ----------
        min_metric : e.g. {"val_accuracy": 0.9} — only exps where metric >= threshold
        predicate  : arbitrary function applied after SQL filtering
        """
        clauses: List[str] = []
        params: List[Any] = []

        if name:
            clauses.append("name LIKE ?")
            params.append(f"%{name}%")
        if status:
            clauses.append("status = ?")
            params.append(status)
        if n_qubits is not None:
            clauses.append("n_qubits = ?")
            params.append(n_qubits)
        if circuit_depth is not None:
            clauses.append("circuit_depth = ?")
            params.append(circuit_depth)
        if tag:
            clauses.append("tags LIKE ?")
            params.append(f'%"{tag}"%')

        sql = "SELECT data FROM experiments"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC"

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        exps = [self._row_to_exp(r) for r in rows]

        if min_metric:
            for metric_name, threshold in min_metric.items():
                exps = [e for e in exps if e.metrics.get(metric_name, -1e9) >= threshold]

        if predicate:
            exps = [e for e in exps if predicate(e)]

        return exps

    def all(self) -> List[QMLExperiment]:
        return self.search()

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]

    # ── ranking ──────────────────────────────────────────────────────────────

    def top_k(
        self,
        k: int = 3,
        metric: str = "val_accuracy",
        higher_is_better: bool = True,
        status: str = "completed",
    ) -> List[QMLExperiment]:
        """Return the k best experiments by a given metric."""
        exps = self.search(status=status)
        exps = [e for e in exps if metric in e.metrics]
        exps.sort(key=lambda e: e.metrics[metric], reverse=higher_is_better)
        return exps[:k]

    # ── bulk export ──────────────────────────────────────────────────────────

    def export_jsonl(self, path: str | Path) -> int:
        """Dump all experiments as JSONL. Returns count written."""
        exps = self.all()
        with open(path, "w") as f:
            for exp in exps:
                f.write(exp.to_json().replace("\n", " ") + "\n")
        return len(exps)
