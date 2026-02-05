"""Tests for known-result fixture framework."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.fixtures.known_results import (
    KnownResult,
    load_known_results,
    save_known_results,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestKnownResult:
    """Tests for KnownResult class."""

    def test_assert_matches_float_exact(self) -> None:
        """Should match exact float values."""
        case = KnownResult(name="test", input={}, expected=1.5)
        case.assert_matches(1.5)

    def test_assert_matches_float_within_tolerance(self) -> None:
        """Should match floats within tolerance."""
        case = KnownResult(name="test", input={}, expected=1.5, tolerance=0.01)
        case.assert_matches(1.505)

    def test_assert_matches_float_outside_tolerance(self) -> None:
        """Should fail for floats outside tolerance."""
        case = KnownResult(name="test", input={}, expected=1.5, tolerance=0.001)
        with pytest.raises(AssertionError, match="Expected 1.5"):
            case.assert_matches(1.51)

    def test_assert_matches_dict(self) -> None:
        """Should match dict values."""
        case = KnownResult(
            name="test",
            input={},
            expected={"a": 1.0, "b": 2.0},
            tolerance=1e-10,
        )
        case.assert_matches({"a": 1.0, "b": 2.0})

    def test_assert_matches_dict_tolerance(self) -> None:
        """Should match dict float values within tolerance."""
        case = KnownResult(
            name="test",
            input={},
            expected={"value": 1.5},
            tolerance=0.01,
        )
        case.assert_matches({"value": 1.505})

    def test_assert_matches_dict_key_mismatch(self) -> None:
        """Should fail for mismatched keys."""
        case = KnownResult(name="test", input={}, expected={"a": 1})
        with pytest.raises(AssertionError, match="Keys mismatch"):
            case.assert_matches({"b": 1})

    def test_assert_matches_non_float(self) -> None:
        """Should match non-float values exactly."""
        case = KnownResult(name="test", input={}, expected="hello")
        case.assert_matches("hello")

    def test_assert_matches_non_float_mismatch(self) -> None:
        """Should fail for non-matching non-float values."""
        case = KnownResult(name="test", input={}, expected="hello")
        with pytest.raises(AssertionError, match="Expected hello"):
            case.assert_matches("world")


class TestLoadSaveKnownResults:
    """Tests for load/save functions."""

    def test_load_known_results(self) -> None:
        """Should load existing known results."""
        # Uses the ohlc_aggregation.json we created
        cases = load_known_results("ohlc_aggregation")
        assert len(cases) >= 1
        assert cases[0].name == "simple_5m_aggregation"

    def test_load_known_results_not_found(self) -> None:
        """Should raise for missing file."""
        with pytest.raises(FileNotFoundError):
            load_known_results("nonexistent_category")

    def test_save_and_load_roundtrip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should roundtrip save and load."""
        # Temporarily override the known results directory
        test_dir = tmp_path / "known_results"
        test_dir.mkdir()

        import tests.fixtures.known_results as kr_module

        monkeypatch.setattr(kr_module, "KNOWN_RESULTS_DIR", test_dir)

        cases = [
            KnownResult(
                name="case1",
                input={"x": 1},
                expected=2.0,
                tolerance=0.001,
                description="Test case",
                tags=["test"],
            ),
        ]
        save_known_results("test_category", cases, description="Test results")

        loaded = load_known_results("test_category")
        assert len(loaded) == 1
        assert loaded[0].name == "case1"
        assert loaded[0].expected == 2.0
        assert loaded[0].tags == ["test"]
