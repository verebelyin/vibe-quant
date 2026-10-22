"""AST boundary detector for the extracted gate chain (lane 4 T6).

Text scans cannot distinguish a stale private from a docstring that merely
mentions it (design revision r2 row 5). This detector walks the AST and flags
ONLY three node shapes, so comments, docstrings and string literals are
invisible by construction:

1. an ``ast.Attribute`` whose ``attr`` is in ``REMOVED_ATTRS``
2. an ``ast.ImportFrom`` alias whose name is in ``REMOVED_NAMES``
3. a ``FunctionDef`` / ``AsyncFunctionDef`` / ``ClassDef`` or a module-level
   ``Assign`` target named in ``REMOVED_NAMES``, outside ``gate_chain.py``

T6 only removes ``_metrics_to_holdout_result``, ``_symbol_mtm_stats`` and
``_FAILED_WINDOW`` (all covered by rule 3 / rule 2 here); T10 extends the sets
with the pipeline shims.
"""

from __future__ import annotations

import ast
from pathlib import Path

REMOVED_NAMES: set[str] = {
    "_metrics_to_holdout_result",
    "_symbol_mtm_stats",
    "_FAILED_WINDOW",
}
REMOVED_ATTRS: set[str] = set()

_GATE_CHAIN = "vibe_quant/discovery/gate_chain.py"
_PIPELINE = "vibe_quant/discovery/pipeline.py"
_SCAN_ROOTS = ("vibe_quant", "scripts", "tests")


class _Detector(ast.NodeVisitor):
    """Collects only the three stale-name node shapes; strings are never visited."""

    def __init__(
        self,
        relpath: str,
        removed_names: set[str],
        removed_attrs: set[str],
        hits: list[str],
    ) -> None:
        self.relpath = relpath
        self.removed_names = removed_names
        self.removed_attrs = removed_attrs
        self.hits = hits
        self._function_depth = 0
        self._in_tests = relpath.startswith("tests/")

    def _hit(self, node: ast.AST, what: str) -> None:
        self.hits.append(f"{self.relpath}:{getattr(node, 'lineno', 0)}: {what}")

    def visit_Module(self, node: ast.Module) -> None:
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id in self.removed_names:
                        self._hit(stmt, f"module-level assign {target.id}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name in self.removed_names:
                self._hit(node, f"import {alias.name}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            node.attr in self.removed_attrs
            and self.relpath not in (_GATE_CHAIN, _PIPELINE)
            and not self._is_test_self_write(node)
        ):
            self._hit(node, f"attribute .{node.attr}")
        self.generic_visit(node)

    def _is_test_self_write(self, node: ast.Attribute) -> bool:
        """``self._fitness_cache`` inside a test function: only ``run`` fakes write it."""
        return (
            self._in_tests
            and node.attr == "_fitness_cache"
            and self._function_depth > 0
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        )

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if node.name in self.removed_names and self.relpath != _GATE_CHAIN:
            self._hit(node, f"definition {node.name}")
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name in self.removed_names and self.relpath != _GATE_CHAIN:
            self._hit(node, f"definition {node.name}")
        self.generic_visit(node)


def find_violations(
    source: str,
    *,
    removed_names: set[str],
    removed_attrs: set[str],
    relpath: str = "<probe>",
) -> list[str]:
    """Stale-name hits in one source text (in-memory sources for the self-check)."""
    hits: list[str] = []
    _Detector(relpath, removed_names, removed_attrs, hits).visit(ast.parse(source))
    return hits


def scan(root: Path) -> list[str]:
    """Every stale-name hit under the scanned trees, relative to ``root``."""
    hits: list[str] = []
    for name in _SCAN_ROOTS:
        for path in sorted((root / name).rglob("*.py")):
            relpath = path.relative_to(root).as_posix()
            hits.extend(
                find_violations(
                    path.read_text(encoding="utf-8"),
                    removed_names=REMOVED_NAMES,
                    removed_attrs=REMOVED_ATTRS,
                    relpath=relpath,
                )
            )
    return hits


_PROBE_SOURCE = "pipe._validate_top_strategies(x, 1)"


def test_detector_self_check() -> None:
    probe = {"_validate_top_strategies"}

    def hits(source: str) -> list[str]:
        return find_violations(
            source, removed_names=probe, removed_attrs=probe, relpath="tests/probe.py"
        )

    assert hits(_PROBE_SOURCE) == ["tests/probe.py:1: attribute ._validate_top_strategies"]

    for silent in ('"""calls _validate_top_strategies"""', "# _validate_top_strategies"):
        assert hits(silent) == []

    for stale in (
        "from pipeline import _validate_top_strategies",
        "_validate_top_strategies = 1",
        "def _validate_top_strategies():\n    return None",
    ):
        assert len(hits(stale)) == 1


def test_no_gate_private_access_outside_module() -> None:
    root = Path(__file__).resolve().parents[2]
    assert scan(root) == []
