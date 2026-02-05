"""Known-result fixture framework for regression testing.

This module provides fixtures and utilities for testing against known results,
enabling deterministic regression tests for calculations that should produce
consistent outputs across code changes.

Usage:
    @pytest.fixture
    def sharpe_ratio_cases() -> list[KnownResult]:
        return load_known_results("sharpe_ratio")

    def test_sharpe_ratio_known_results(sharpe_ratio_cases):
        for case in sharpe_ratio_cases:
            result = calculate_sharpe_ratio(case.input)
            case.assert_matches(result)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# Default path for known result files
FIXTURES_DIR = Path(__file__).parent
KNOWN_RESULTS_DIR = FIXTURES_DIR / "known_results"


@dataclass
class KnownResult:
    """A single known-result test case.

    Attributes:
        name: Descriptive name for the test case.
        input: Input data for the calculation.
        expected: Expected output value.
        tolerance: Absolute tolerance for floating point comparison.
        description: Optional description of what this case tests.
    """

    name: str
    input: dict[str, object]
    expected: object
    tolerance: float = 1e-10
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def assert_matches(
        self,
        actual: object,
        tolerance: float | None = None,
    ) -> None:
        """Assert that actual result matches expected within tolerance.

        Args:
            actual: The actual computed result.
            tolerance: Override default tolerance (optional).

        Raises:
            AssertionError: If result doesn't match within tolerance.
        """
        tol = tolerance if tolerance is not None else self.tolerance

        if isinstance(self.expected, float) and isinstance(actual, float):
            if abs(actual - self.expected) > tol:
                raise AssertionError(
                    f"[{self.name}] Expected {self.expected}, got {actual} "
                    f"(diff={abs(actual - self.expected)}, tolerance={tol})"
                )
        elif isinstance(self.expected, dict) and isinstance(actual, dict):
            self._assert_dict_matches(actual, self.expected, tol)
        else:
            if actual != self.expected:
                raise AssertionError(
                    f"[{self.name}] Expected {self.expected}, got {actual}"
                )

    def _assert_dict_matches(
        self,
        actual: dict[str, object],
        expected: dict[str, object],
        tolerance: float,
    ) -> None:
        """Recursively compare dicts with tolerance for floats."""
        if set(actual.keys()) != set(expected.keys()):
            raise AssertionError(
                f"[{self.name}] Keys mismatch: expected {set(expected.keys())}, "
                f"got {set(actual.keys())}"
            )

        for key in expected:
            exp_val = expected[key]
            act_val = actual[key]

            if isinstance(exp_val, float) and isinstance(act_val, float):
                if abs(act_val - exp_val) > tolerance:
                    raise AssertionError(
                        f"[{self.name}] {key}: expected {exp_val}, got {act_val}"
                    )
            elif isinstance(exp_val, dict) and isinstance(act_val, dict):
                self._assert_dict_matches(
                    actual=act_val,  # type: ignore[arg-type]
                    expected=exp_val,  # type: ignore[arg-type]
                    tolerance=tolerance,
                )
            elif act_val != exp_val:
                raise AssertionError(
                    f"[{self.name}] {key}: expected {exp_val}, got {act_val}"
                )


def load_known_results(category: str) -> list[KnownResult]:
    """Load known results from JSON file.

    Args:
        category: Category name (maps to {category}.json file).

    Returns:
        List of KnownResult objects.

    Raises:
        FileNotFoundError: If the known results file doesn't exist.
    """
    file_path = KNOWN_RESULTS_DIR / f"{category}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Known results file not found: {file_path}")

    with open(file_path) as f:
        data = json.load(f)

    results = []
    for case in data["cases"]:
        results.append(
            KnownResult(
                name=case["name"],
                input=case["input"],
                expected=case["expected"],
                tolerance=case.get("tolerance", 1e-10),
                description=case.get("description", ""),
                tags=case.get("tags", []),
            )
        )
    return results


def save_known_results(
    category: str,
    cases: list[KnownResult],
    description: str = "",
) -> None:
    """Save known results to JSON file.

    Useful for generating fixture files from verified calculations.

    Args:
        category: Category name (maps to {category}.json file).
        cases: List of KnownResult objects to save.
        description: Description for the entire category.
    """
    KNOWN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = KNOWN_RESULTS_DIR / f"{category}.json"

    data = {
        "description": description,
        "cases": [
            {
                "name": case.name,
                "input": case.input,
                "expected": case.expected,
                "tolerance": case.tolerance,
                "description": case.description,
                "tags": case.tags,
            }
            for case in cases
        ],
    }

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def known_result_parametrize(
    category: str,
    compute_fn: Callable[[dict[str, object]], object],
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """Decorator to parametrize a test with known results.

    Usage:
        @known_result_parametrize("sharpe_ratio", calculate_sharpe)
        def test_sharpe_ratio(case: KnownResult, result: object) -> None:
            case.assert_matches(result)

    Args:
        category: Category name for loading fixtures.
        compute_fn: Function that takes input dict and returns result.

    Returns:
        Decorator function.
    """
    import pytest

    def decorator(test_fn: Callable[..., None]) -> Callable[..., None]:
        cases = load_known_results(category)

        @pytest.mark.parametrize(
            "case",
            cases,
            ids=[c.name for c in cases],
        )
        def wrapped(case: KnownResult) -> None:
            result = compute_fn(case.input)
            test_fn(case, result)

        return wrapped

    return decorator
