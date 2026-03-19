"""Tests for the Quantum MLOps Workbench core modules."""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantum_mlops import (
    QMLExperiment,
    ExperimentRegistry,
    CircuitVersioning,
    ReproducibilityChecker,
    QMLDashboard,
)
from quantum_mlops.experiment import CircuitSpec, ClassicalSpec


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_circuit(n_qubits=4, depth=3):
    gates = [f"H({q})" for q in range(n_qubits)] + [f"CNOT({q},{q+1})" for q in range(n_qubits - 1)]
    return CircuitSpec(n_qubits=n_qubits, depth=depth, gate_sequence=gates)


def make_classical(sizes=(64, 32, 10)):
    return ClassicalSpec(layer_sizes=list(sizes))


def make_experiment(name="test-exp", n_qubits=4, depth=3, seed=42, metrics=None):
    exp = QMLExperiment(
        name=name,
        circuit=make_circuit(n_qubits, depth),
        classical=make_classical(),
        hyperparams={"lr": 0.01, "epochs": 50},
        random_seed=seed,
    )
    exp.complete(metrics or {"val_accuracy": 0.85, "val_loss": 0.32})
    return exp


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    reg = ExperimentRegistry(path)
    yield reg
    os.unlink(path)


# ── QMLExperiment ─────────────────────────────────────────────────────────────

class TestQMLExperiment:
    def test_creation(self):
        exp = make_experiment()
        assert exp.name == "test-exp"
        assert exp.circuit.n_qubits == 4
        assert exp.status == "completed"

    def test_lifecycle(self):
        exp = make_experiment()
        exp2 = QMLExperiment(name="x", circuit=make_circuit(), classical=make_classical())
        assert exp2.status == "pending"
        exp2.start()
        assert exp2.status == "running"
        exp2.complete({"acc": 0.9})
        assert exp2.status == "completed"
        assert exp2.metrics["acc"] == 0.9

    def test_fail(self):
        exp = QMLExperiment(name="x", circuit=make_circuit(), classical=make_classical())
        exp.fail("OOM")
        assert exp.status == "failed"
        assert "OOM" in exp.metadata["failure_reason"]

    def test_serialization_round_trip(self):
        exp = make_experiment()
        d = exp.to_dict()
        restored = QMLExperiment.from_dict(d)
        assert restored.experiment_id == exp.experiment_id
        assert restored.circuit.n_qubits == exp.circuit.n_qubits
        assert restored.metrics == exp.metrics

    def test_unique_ids(self):
        ids = {make_experiment().experiment_id for _ in range(50)}
        assert len(ids) == 50  # all unique

    def test_str_repr(self):
        exp = make_experiment()
        s = str(exp)
        assert exp.experiment_id in s
        assert "completed" in s


# ── ExperimentRegistry ────────────────────────────────────────────────────────

class TestExperimentRegistry:
    def test_save_and_get(self, db):
        exp = make_experiment()
        db.save(exp)
        fetched = db.get(exp.experiment_id)
        assert fetched is not None
        assert fetched.name == exp.name

    def test_get_missing(self, db):
        assert db.get("nope") is None

    def test_count(self, db):
        assert db.count() == 0
        db.save(make_experiment())
        assert db.count() == 1
        db.save(make_experiment())
        assert db.count() == 2

    def test_search_by_status(self, db):
        e1 = make_experiment(metrics={"val_accuracy": 0.9})
        e2 = QMLExperiment(name="pending", circuit=make_circuit(), classical=make_classical())
        db.save(e1)
        db.save(e2)
        completed = db.search(status="completed")
        assert len(completed) == 1
        assert completed[0].experiment_id == e1.experiment_id

    def test_search_by_name(self, db):
        db.save(make_experiment(name="vqe-run-1"))
        db.save(make_experiment(name="qaoa-run-1"))
        results = db.search(name="vqe")
        assert len(results) == 1
        assert results[0].name == "vqe-run-1"

    def test_search_by_qubits(self, db):
        db.save(make_experiment(n_qubits=4))
        db.save(make_experiment(n_qubits=8))
        r4 = db.search(n_qubits=4)
        r8 = db.search(n_qubits=8)
        assert len(r4) == 1
        assert len(r8) == 1

    def test_search_min_metric(self, db):
        db.save(make_experiment(name="low", metrics={"val_accuracy": 0.6}))
        db.save(make_experiment(name="high", metrics={"val_accuracy": 0.95}))
        results = db.search(min_metric={"val_accuracy": 0.9})
        assert len(results) == 1
        assert results[0].name == "high"

    def test_top_k(self, db):
        for i, acc in enumerate([0.7, 0.85, 0.9, 0.6, 0.95]):
            db.save(make_experiment(name=f"exp-{i}", metrics={"val_accuracy": acc}))
        top3 = db.top_k(k=3, metric="val_accuracy")
        assert len(top3) == 3
        assert top3[0].metrics["val_accuracy"] == 0.95

    def test_delete(self, db):
        exp = make_experiment()
        db.save(exp)
        assert db.count() == 1
        db.delete(exp.experiment_id)
        assert db.count() == 0

    def test_upsert(self, db):
        exp = make_experiment()
        db.save(exp)
        exp.metrics["val_accuracy"] = 0.99
        db.save(exp)  # should replace
        assert db.count() == 1
        assert db.get(exp.experiment_id).metrics["val_accuracy"] == 0.99

    def test_export_jsonl(self, db, tmp_path):
        for i in range(3):
            db.save(make_experiment(name=f"exp-{i}"))
        out = tmp_path / "out.jsonl"
        count = db.export_jsonl(out)
        assert count == 3
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 3


# ── CircuitVersioning ─────────────────────────────────────────────────────────

class TestCircuitVersioning:
    def test_tag_and_get(self):
        cv = CircuitVersioning()
        v = cv.tag("v1.0", n_qubits=4, gates=["H(0)", "CNOT(0,1)"])
        assert cv.get("v1.0").n_qubits == 4
        assert len(cv) == 1

    def test_duplicate_tag_raises(self):
        cv = CircuitVersioning()
        cv.tag("v1.0", 4, ["H(0)"])
        with pytest.raises(ValueError):
            cv.tag("v1.0", 4, ["X(0)"])

    def test_diff_identical(self):
        cv = CircuitVersioning()
        gates = ["H(0)", "CNOT(0,1)", "RY(1,θ)"]
        cv.tag("v1", 4, gates)
        cv.tag("v2", 4, gates)
        diff = cv.diff("v1", "v2")
        assert diff.is_identical()
        assert diff.similarity == 1.0

    def test_diff_added(self):
        cv = CircuitVersioning()
        cv.tag("v1", 4, ["H(0)", "CNOT(0,1)"])
        cv.tag("v2", 4, ["H(0)", "CNOT(0,1)", "RZ(1,θ)"])
        diff = cv.diff("v1", "v2")
        assert not diff.is_identical()
        assert "RZ(1,θ)" in diff.added
        assert diff.depth_delta == 1

    def test_diff_removed(self):
        cv = CircuitVersioning()
        cv.tag("v1", 4, ["H(0)", "CNOT(0,1)", "X(2)"])
        cv.tag("v2", 4, ["H(0)", "CNOT(0,1)"])
        diff = cv.diff("v1", "v2")
        assert "X(2)" in diff.removed
        assert diff.depth_delta == -1

    def test_diff_reordered(self):
        cv = CircuitVersioning()
        cv.tag("v1", 4, ["H(0)", "X(1)"])
        cv.tag("v2", 4, ["X(1)", "H(0)"])
        diff = cv.diff("v1", "v2")
        assert diff.reordered
        assert diff.similarity == 1.0

    def test_parent_lineage(self):
        cv = CircuitVersioning()
        cv.tag("v1", 4, ["H(0)"])
        cv.tag("v2", 4, ["H(0)", "X(1)"], parent="v1")
        cv.tag("v3", 4, ["H(0)", "X(1)", "CNOT(0,1)"], parent="v2")
        hist = cv.history("v3")
        assert [v.tag for v in hist] == ["v3", "v2", "v1"]

    def test_persistence(self, tmp_path):
        path = tmp_path / "circuits.json"
        cv = CircuitVersioning(path)
        cv.tag("v1", 4, ["H(0)", "CNOT(0,1)"])

        cv2 = CircuitVersioning(path)
        assert len(cv2) == 1
        assert cv2.get("v1").n_qubits == 4

    def test_tags_list(self):
        cv = CircuitVersioning()
        cv.tag("a", 4, ["H(0)"])
        cv.tag("b", 4, ["X(0)"])
        assert set(cv.tags()) == {"a", "b"}


# ── ReproducibilityChecker ────────────────────────────────────────────────────

class TestReproducibilityChecker:
    def test_identical_reproducible(self, db):
        exp = make_experiment(seed=42, metrics={"val_accuracy": 0.85})
        exp2 = make_experiment(seed=42, metrics={"val_accuracy": 0.852})
        db.save(exp)
        db.save(exp2)
        checker = ReproducibilityChecker(db, tolerance=0.01)
        report = checker.check(exp.experiment_id, exp2.experiment_id)
        assert report.is_reproducible

    def test_different_seed_not_reproducible(self, db):
        exp1 = make_experiment(seed=42, metrics={"val_accuracy": 0.85})
        exp2 = make_experiment(seed=99, metrics={"val_accuracy": 0.85})
        db.save(exp1)
        db.save(exp2)
        checker = ReproducibilityChecker(db)
        report = checker.check(exp1.experiment_id, exp2.experiment_id)
        assert not report.is_reproducible
        assert not report.seed_match

    def test_different_metrics_not_reproducible(self, db):
        exp1 = make_experiment(seed=42, metrics={"val_accuracy": 0.8})
        exp2 = make_experiment(seed=42, metrics={"val_accuracy": 0.95})
        db.save(exp1)
        db.save(exp2)
        checker = ReproducibilityChecker(db, tolerance=0.01)
        report = checker.check(exp1.experiment_id, exp2.experiment_id)
        assert not report.is_reproducible
        assert not report.metrics_within_tolerance

    def test_config_diff_detected(self, db):
        exp1 = make_experiment(n_qubits=4, seed=42, metrics={"val_accuracy": 0.85})
        exp2 = make_experiment(n_qubits=8, seed=42, metrics={"val_accuracy": 0.85})
        db.save(exp1)
        db.save(exp2)
        checker = ReproducibilityChecker(db)
        report = checker.check(exp1.experiment_id, exp2.experiment_id)
        assert not report.config_match
        assert "circuit.n_qubits" in report.config_diffs

    def test_summary_includes_verdict(self, db):
        exp = make_experiment()
        db.save(exp)
        checker = ReproducibilityChecker(db)
        report = checker.check(exp.experiment_id, exp.experiment_id)
        summary = report.summary()
        assert "REPRODUCIBLE" in summary

    def test_missing_experiment_raises(self, db):
        checker = ReproducibilityChecker(db)
        with pytest.raises(KeyError):
            checker.check("nonexistent", "also-nonexistent")


# ── QMLDashboard ──────────────────────────────────────────────────────────────

class TestQMLDashboard:
    def test_comparison_table(self):
        exps = [make_experiment(name=f"e{i}", metrics={"val_accuracy": 0.7 + i * 0.05}) for i in range(3)]
        dash = QMLDashboard()
        table = dash.comparison_table(exps)
        assert "val_accuracy" in table
        assert "★" in table  # best marker

    def test_leaderboard(self):
        exps = [make_experiment(name=f"e{i}", metrics={"val_accuracy": 0.7 + i * 0.05}) for i in range(5)]
        dash = QMLDashboard()
        lb = dash.leaderboard(exps, top_k=3)
        assert "🥇" in lb
        assert "🥈" in lb
        assert "🥉" in lb

    def test_summary_card(self):
        exp = make_experiment()
        dash = QMLDashboard()
        card = dash.summary_card(exp)
        assert exp.experiment_id in card
        assert "val_accuracy" in card

    def test_empty_experiments(self):
        dash = QMLDashboard()
        result = dash.comparison_table([])
        assert "no experiments" in result.lower()

    def test_no_primary_metric(self):
        exps = [make_experiment(metrics={"other_metric": 0.5})]
        dash = QMLDashboard(primary_metric="val_accuracy")
        lb = dash.leaderboard(exps)
        assert "no experiments" in lb.lower()

    def test_sorting_order(self):
        e1 = make_experiment(name="low",  metrics={"val_accuracy": 0.6})
        e2 = make_experiment(name="high", metrics={"val_accuracy": 0.95})
        dash = QMLDashboard()
        table = dash.comparison_table([e1, e2])
        # Best rank (#1) should appear before worst (#2)
        pos_high = table.find("0.9500")
        pos_low  = table.find("0.6000")
        assert pos_high < pos_low
