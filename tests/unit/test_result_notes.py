"""The notes-column codec (bead vibe-quant-rabmo.8, T10 -- design 2.3, ruling R6).

One module owns every ``json.loads`` / ``json.dumps`` of a ``backtest_results.notes``
column. ``parse_notes`` must stay byte- and exception-identical to ``json.loads``
(readers keep their own ``try/except``), ``notes_mapping`` keeps the router's
"``{}`` on anything but a non-empty JSON object" rule, and ``dump_notes`` keeps
the default separators (stored notes bytes are compared across refactors).

Outcomes at the module edge only: the existing R21-R29 / R38 / R38c / R38d rows
through the flipped adapters are the readers' proof.
"""

from __future__ import annotations

import json

import pytest

from vibe_quant.db.result_notes import dump_notes, notes_mapping, parse_notes

# --- parse_notes: == json.loads, SAME exceptions -------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1, "b": [2, 3]}', {"a": 1, "b": [2, 3]}),
        ('["x", null, {"k": true}]', ["x", None, {"k": True}]),
        ("null", None),
        ("5", 5),
        ("", json.JSONDecodeError),
        ("{not json", json.JSONDecodeError),
    ],
    ids=["object", "list", "null", "number", "empty", "garbage"],
)
def test_parse_notes_is_json_loads(text: str, expected: object) -> None:
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            parse_notes(text)
        with pytest.raises(expected):
            json.loads(text)
        return
    assert parse_notes(text) == expected == json.loads(text)


def test_parse_notes_type_error_matches_json_loads() -> None:
    """A non-text value raises exactly what ``json.loads`` raises (TypeError)."""
    with pytest.raises(TypeError):
        parse_notes(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        json.loads(None)  # type: ignore[arg-type]


def test_parse_notes_bytes_and_bytearray() -> None:
    assert parse_notes(b'{"bytes": 1}') == {"bytes": 1}
    assert parse_notes(bytearray(b'["bytearray"]')) == ["bytearray"]


# --- notes_mapping: {} unless a non-empty str parses to a dict -------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, {}),
        ("", {}),
        (b"{}", {}),
        (5, {}),
        ("garbage", {}),
        ("[]", {}),
        ("null", {}),
        ("5", {}),
        ("{}", {}),
        ('{"a": 1}', {"a": 1}),
        ('{"nested": {"k": [1, 2]}}', {"nested": {"k": [1, 2]}}),
    ],
    ids=[
        "none",
        "empty",
        "bytes",
        "int",
        "garbage",
        "list",
        "null",
        "number",
        "empty_object",
        "object",
        "nested",
    ],
)
def test_notes_mapping_rows(value: object, expected: dict[str, object]) -> None:
    assert notes_mapping(value) == expected


# --- dump_notes: == json.dumps (default separators, escaping, key order) ---------------------


def test_dump_notes_bytes() -> None:
    payload: dict[str, object] = {"b": {"z": [1, 2], "a": "\u00e9"}, "a": 1}
    expected = '{"b": {"z": [1, 2], "a": "\\u00e9"}, "a": 1}'
    assert dump_notes(payload) == expected == json.dumps(payload)


def test_notes_mapping_round_trips_dump_notes() -> None:
    payload: dict[str, object] = {"x": [1, {"y": True}], "z": "s"}
    assert notes_mapping(dump_notes(payload)) == payload
