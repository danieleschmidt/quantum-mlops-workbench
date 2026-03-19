"""CircuitVersioning — version control and diffing for quantum circuits."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CircuitVersion:
    """A tagged snapshot of a quantum circuit's gate sequence."""
    tag: str                        # e.g. "v1.0", "ansatz-candidate", "baseline"
    n_qubits: int
    gate_sequence: List[str]        # ordered list of gate strings
    description: str = ""
    author: str = ""
    timestamp: float = field(default_factory=time.time)
    parent_tag: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def depth(self) -> int:
        return len(self.gate_sequence)

    def gate_types(self) -> Dict[str, int]:
        """Count occurrences of each gate type."""
        counts: Dict[str, int] = {}
        for g in self.gate_sequence:
            name = g.split("(")[0]  # "CNOT(0,1)" → "CNOT"
            counts[name] = counts.get(name, 0) + 1
        return counts


@dataclass
class CircuitDiff:
    """Result of diffing two circuit versions."""
    tag_a: str
    tag_b: str
    added: List[str]          # gates in B but not in A (by index)
    removed: List[str]        # gates in A but not in B
    reordered: bool           # same gates but different order
    depth_delta: int          # B.depth - A.depth
    similarity: float         # 0.0 (completely different) → 1.0 (identical)

    def is_identical(self) -> bool:
        return self.similarity == 1.0

    def summary(self) -> str:
        if self.is_identical():
            return f"{self.tag_a} == {self.tag_b} (identical)"
        parts = []
        if self.added:
            parts.append(f"+{len(self.added)} gates")
        if self.removed:
            parts.append(f"-{len(self.removed)} gates")
        if self.reordered:
            parts.append("reordered")
        delta_str = (
            f"depth {'+' if self.depth_delta >= 0 else ''}{self.depth_delta}"
        )
        return (
            f"{self.tag_a} → {self.tag_b}: "
            + ", ".join(parts)
            + f"  [{delta_str}, sim={self.similarity:.2f}]"
        )


class CircuitVersioning:
    """
    Lightweight version control for quantum circuits.

    Supports tagging, diffing, branching (parent references), and
    persistent storage as a JSON registry.

    Usage
    -----
    cv = CircuitVersioning()
    cv.tag("v1.0", n_qubits=4, gates=["H(0)", "CNOT(0,1)"])
    cv.tag("v1.1", n_qubits=4, gates=["H(0)", "CNOT(0,1)", "RZ(1,θ)"], parent="v1.0")
    diff = cv.diff("v1.0", "v1.1")
    print(diff.summary())
    history = cv.history("v1.1")
    """

    def __init__(self, storage_path: Optional[str | Path] = None) -> None:
        self._versions: Dict[str, CircuitVersion] = {}
        self.storage_path = Path(storage_path) if storage_path else None
        if self.storage_path and self.storage_path.exists():
            self._load()

    # ── tagging ──────────────────────────────────────────────────────────────

    def tag(
        self,
        tag: str,
        n_qubits: int,
        gates: List[str],
        description: str = "",
        author: str = "",
        parent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CircuitVersion:
        """Tag a circuit snapshot."""
        if tag in self._versions:
            raise ValueError(f"Tag '{tag}' already exists. Use a new tag or delete first.")
        if parent and parent not in self._versions:
            raise ValueError(f"Parent tag '{parent}' not found.")

        version = CircuitVersion(
            tag=tag,
            n_qubits=n_qubits,
            gate_sequence=list(gates),
            description=description,
            author=author,
            parent_tag=parent,
            metadata=metadata or {},
        )
        self._versions[tag] = version
        if self.storage_path:
            self._save()
        return version

    def get(self, tag: str) -> CircuitVersion:
        if tag not in self._versions:
            raise KeyError(f"Circuit version '{tag}' not found.")
        return self._versions[tag]

    def tags(self) -> List[str]:
        """Return all tags sorted by timestamp."""
        return sorted(self._versions, key=lambda t: self._versions[t].timestamp)

    def delete(self, tag: str) -> None:
        self._versions.pop(tag, None)
        if self.storage_path:
            self._save()

    # ── diffing ──────────────────────────────────────────────────────────────

    def diff(self, tag_a: str, tag_b: str) -> CircuitDiff:
        """
        Compare two circuit versions by gate sequence.

        Uses multiset intersection for similarity; also detects reordering.
        """
        va = self.get(tag_a)
        vb = self.get(tag_b)

        gates_a = va.gate_sequence
        gates_b = vb.gate_sequence

        # Compute added / removed via multiset diff
        from collections import Counter
        cnt_a = Counter(gates_a)
        cnt_b = Counter(gates_b)

        added   = list((cnt_b - cnt_a).elements())
        removed = list((cnt_a - cnt_b).elements())

        # Detect reordering: same multisets but different order
        same_set = (cnt_a == cnt_b)
        reordered = same_set and (gates_a != gates_b)

        # Jaccard-style similarity over multisets
        intersection = sum((cnt_a & cnt_b).values())
        union = sum((cnt_a | cnt_b).values())
        similarity = intersection / union if union > 0 else 1.0

        return CircuitDiff(
            tag_a=tag_a,
            tag_b=tag_b,
            added=added,
            removed=removed,
            reordered=reordered,
            depth_delta=vb.depth() - va.depth(),
            similarity=similarity,
        )

    # ── lineage ──────────────────────────────────────────────────────────────

    def history(self, tag: str) -> List[CircuitVersion]:
        """Walk parent chain from tag back to root."""
        chain = []
        current: Optional[str] = tag
        visited = set()
        while current:
            if current in visited:
                break  # cycle guard
            visited.add(current)
            v = self._versions.get(current)
            if v is None:
                break
            chain.append(v)
            current = v.parent_tag
        return chain

    # ── persistence ──────────────────────────────────────────────────────────

    def _save(self) -> None:
        assert self.storage_path
        data = {tag: asdict(v) for tag, v in self._versions.items()}
        self.storage_path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        assert self.storage_path
        data = json.loads(self.storage_path.read_text())
        self._versions = {tag: CircuitVersion(**v) for tag, v in data.items()}

    def __len__(self) -> int:
        return len(self._versions)

    def __repr__(self) -> str:
        return f"CircuitVersioning({len(self)} versions: {self.tags()})"
