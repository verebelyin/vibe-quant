"""Mapper from LLM ``proposed_indicators`` JSON to ``IndicatorSpec`` args.

Slice 1 of 4 for bd-3p1k.1 — backend foundation only. The mapper takes a
single proposal dict (as emitted by the extractor under
``proposed_indicators_json``) and produces the kwargs that the
``IndicatorSpec`` constructor needs, plus a separate ``compute_fn``
synthesis hook that subsequent slices will fill in (LLM codegen).

The mapper is intentionally pure (no LLM calls, no I/O). It validates
the input shape, normalizes the indicator name, applies range
heuristics where the LLM didn't supply one, and reports per-parameter
provenance (``llm`` vs ``heuristic``) so the slice-2 file writer can
record it in the AUTO-GENERATED header.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Names must survive being used as a YAML ``type:`` value AND as a Python
# module filename — same rule as ``plugin_scaffold._NAME_RE``.
_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

# Heuristic param ranges applied when the LLM omits ``range`` on a
# parameter. The rule mirrors the design notes on bd-3p1k.1: period-like
# params (typical default 5-100) get (5, 50); threshold-like params
# (typical default 0-100) get (10, 90); otherwise no GA enrollment.
_PERIOD_RANGE: tuple[float, float] = (5.0, 50.0)
_THRESHOLD_RANGE: tuple[float, float] = (10.0, 90.0)
_PERIOD_DEFAULT_HI = 100
_THRESHOLD_DEFAULT_HI = 100


class InvalidProposalError(ValueError):
    """Raised when a proposal can't be mapped (missing ``formula``, bad name)."""


@dataclass
class IndicatorSpecArgs:
    """Plain dataclass mirror of ``IndicatorSpec.__init__`` kwargs.

    Slice 2 hands this to a rendering function that emits the actual
    ``register_spec(IndicatorSpec(...))`` call in the generated plugin
    file. ``range_provenance`` is *not* on ``IndicatorSpec`` itself — it
    is metadata for the header-comment block only.
    """

    name: str
    default_params: dict[str, Any]
    param_schema: dict[str, type]
    param_ranges: dict[str, tuple[float, float]]
    threshold_range: tuple[float, float] | None
    output_names: tuple[str, ...]
    display_name: str
    description: str
    category: str = "Custom"
    range_provenance: dict[str, str] = field(default_factory=dict)


def normalize_name(raw: Any) -> str:
    """Return the canonical UPPER name for an indicator proposal.

    Raises ``InvalidProposalError`` if the raw string can't be coerced
    into a valid identifier — the snake_case → UPPER coercion drops
    spaces and dashes the same way ``plugin_scaffold._normalize_name``
    does for the CLI path. Accepts ``Any`` because callers pass raw
    LLM JSON straight in; runtime checks the shape.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise InvalidProposalError("indicator name is empty")
    coerced = re.sub(r"[\s-]+", "_", raw.strip()).upper()
    if not _NAME_RE.match(coerced):
        raise InvalidProposalError(
            f"indicator name {raw!r} → {coerced!r} is not a valid identifier "
            "(must start with A-Z, contain only A-Z 0-9 _)"
        )
    return coerced


def suggest_alt_name(name: str, existing: set[str]) -> str:
    """Find the first ``<name>_V<n>`` (n >= 2) not present in ``existing``."""
    for n in range(2, 100):
        candidate = f"{name}_V{n}"
        if candidate not in existing:
            return candidate
    # Practically unreachable — caller has bigger problems if they have
    # 99 colliding versions. Fall back to a timestamped name.
    return f"{name}_VX"


def _map_output_range(raw: Any) -> tuple[float, float] | None:
    """Map the LLM ``output_range`` hint to a GA ``threshold_range``.

    The extractor prompt asks for one of ``"0..100"`` | ``"fraction"`` |
    ``"unbounded"``, but real outputs are noisy — anything we don't
    recognize falls through to ``None`` (no GA enrollment for
    thresholds) rather than raising, so a sloppy proposal still
    scaffolds.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip().lower()
    if s == "0..100" or s == "0-100" or s == "0 to 100":
        return (20.0, 80.0)
    if s == "fraction" or s == "0..1" or s == "0-1":
        return (0.2, 0.8)
    if s == "unbounded":
        return None
    return None


def _coerce_param_value(raw: Any) -> tuple[Any, type]:
    """Best-effort coerce an LLM-emitted default into (value, type).

    The LLM emits parameters as either bare values (``14``) or
    ``{default, range?, description?}`` dicts. We accept either; this
    function unwraps the ``default`` if present and infers the Python
    type. Strings that parse as numbers are kept as strings unless the
    surrounding ``range`` makes them clearly numeric — slice 2's
    codegen prompt can deal with quoted defaults.
    """
    if isinstance(raw, dict):
        raw = raw.get("default")
    if isinstance(raw, bool):  # bool is a subclass of int — check first
        return raw, bool
    if isinstance(raw, int):
        return raw, int
    if isinstance(raw, float):
        return raw, float
    if isinstance(raw, str):
        return raw, str
    # Fallback: drop unknown types to a string repr so the spec
    # registers something rather than crashing during codegen.
    return str(raw), str


_THRESHOLD_NAME_TOKENS = ("threshold", "overbought", "oversold", "level")
_PERIOD_NAME_TOKENS = ("period", "length", "window", "lookback", "bars")


def _heuristic_range(name: str, default: Any) -> tuple[float, float] | None:
    """Return a (lo, hi) range or None if we can't reasonably guess.

    Order matters: name-based hints win over value-based ones, because
    a default of ``70`` could be either a period or a threshold and the
    parameter name disambiguates (``overbought=70`` is clearly the
    latter, ``period=70`` clearly the former). Only when the name is
    opaque do we fall back to the value-range heuristic.
    """
    low = name.lower()
    if any(tok in low for tok in _THRESHOLD_NAME_TOKENS):
        return _THRESHOLD_RANGE
    if any(tok in low for tok in _PERIOD_NAME_TOKENS):
        return _PERIOD_RANGE
    # Value-based fallback. Periods tend to be small ints (5-50);
    # thresholds tend to land in 20-80. The split at 50 is rough but
    # matches typical default conventions across RSI/Stoch/etc.
    if isinstance(default, int) and 1 <= default <= _PERIOD_DEFAULT_HI:
        if default > 50:
            return _THRESHOLD_RANGE
        return _PERIOD_RANGE
    if isinstance(default, float) and 0 <= default <= _THRESHOLD_DEFAULT_HI:
        return _THRESHOLD_RANGE if default > 1.0 else None
    return None


def _map_parameters(
    raw: Any,
) -> tuple[
    dict[str, Any],
    dict[str, type],
    dict[str, tuple[float, float]],
    dict[str, str],
]:
    """Split a raw ``parameters`` dict into the four IndicatorSpec slots.

    Returns ``(default_params, param_schema, param_ranges,
    range_provenance)``. ``range_provenance`` is keyed by param name
    and is one of ``"llm"`` (LLM supplied an explicit range) or
    ``"heuristic"`` (range came from this module's fallback).
    Parameters with no LLM range AND no plausible heuristic are simply
    not entered into ``param_ranges`` — they are still registered (with
    a default), just not GA-enrolled.
    """
    if not isinstance(raw, dict):
        return {}, {}, {}, {}

    defaults: dict[str, Any] = {}
    schema: dict[str, type] = {}
    ranges: dict[str, tuple[float, float]] = {}
    provenance: dict[str, str] = {}

    for key, val in raw.items():
        if not isinstance(key, str) or not key.isidentifier():
            continue
        default, py_type = _coerce_param_value(val)
        defaults[key] = default
        schema[key] = py_type

        if isinstance(val, dict):
            llm_range = val.get("range")
            if (
                isinstance(llm_range, (list, tuple))
                and len(llm_range) == 2
                and all(isinstance(x, (int, float)) for x in llm_range)
                and llm_range[0] < llm_range[1]
            ):
                ranges[key] = (float(llm_range[0]), float(llm_range[1]))
                provenance[key] = "llm"
                continue

        guessed = _heuristic_range(key, default)
        if guessed is not None:
            ranges[key] = guessed
            provenance[key] = "heuristic"

    return defaults, schema, ranges, provenance


def _map_outputs(raw: Any) -> tuple[str, ...]:
    """Map the optional ``outputs`` array to ``IndicatorSpec.output_names``."""
    if isinstance(raw, list) and raw:
        clean = tuple(s for s in raw if isinstance(s, str) and s.isidentifier())
        if clean:
            return clean
    return ("value",)


def proposed_to_spec_args(proposed: dict[str, Any]) -> IndicatorSpecArgs:
    """Convert one ``proposed_indicators_json`` entry to spec kwargs.

    Refuses (``InvalidProposalError``) when ``formula`` is missing — the
    endpoint maps that refusal to status=invalid_input. All other fields
    have sensible fallbacks: a proposal with no ``parameters`` simply
    registers as a parameterless indicator.
    """
    if not isinstance(proposed, dict):
        raise InvalidProposalError("proposal is not a dict")
    formula = proposed.get("formula")
    if not isinstance(formula, str) or not formula.strip():
        raise InvalidProposalError(
            "proposal missing 'formula' — won't synthesize indicator without one"
        )

    name = normalize_name(proposed.get("name"))
    display = proposed.get("display_name")
    desc = proposed.get("description")
    defaults, schema, ranges, provenance = _map_parameters(
        proposed.get("parameters")
    )
    threshold_range = _map_output_range(proposed.get("output_range"))
    output_names = _map_outputs(proposed.get("outputs"))

    return IndicatorSpecArgs(
        name=name,
        default_params=defaults,
        param_schema=schema,
        param_ranges=ranges,
        threshold_range=threshold_range,
        output_names=output_names,
        display_name=(
            display.strip()
            if isinstance(display, str) and display.strip()
            else name.replace("_", " ").title()
        ),
        description=(
            desc.strip()
            if isinstance(desc, str) and desc.strip()
            else f"{name} — proposed indicator awaiting promotion."
        ),
        range_provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Slice 2: LLM codegen + AST safety + file write
# ---------------------------------------------------------------------------


CLAUDE_BIN = "claude"
CODEGEN_TIMEOUT_SECONDS = 60

# Where ``proposed_<name>.py`` files land. A module-level value (not a
# constant evaluated at function call time) so tests can monkeypatch
# ``indicator_scaffold.PLUGINS_DIR`` to a tmp_path without touching the
# real plugins directory.
PLUGINS_DIR: Path = (
    Path(__file__).resolve().parent.parent / "dsl" / "plugins"
)

# Banned top-level identifiers / call names. ``ast.walk`` over the body
# catches them no matter where they're hidden (inside a string-only
# regex-grep would false-positive on docstrings + comments).
_BANNED_IMPORTS = frozenset({"os", "subprocess", "socket", "sys", "pathlib"})
_BANNED_CALLS = frozenset({"exec", "eval", "__import__", "compile", "open"})


class CodegenError(Exception):
    """LLM codegen produced something we can't safely write to disk.

    The ``code`` attribute matches the bead spec's ``error`` field
    vocabulary (``timeout``, ``banned_import:<which>``, ``mypy_fail``,
    ``ruff_fail``, ``syntax_error``, ``non_function``,
    ``missing_signature``) so the endpoint can pass it through
    unchanged.
    """

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _build_codegen_prompt(
    spec: IndicatorSpecArgs, formula: str, source_quote: str | None
) -> str:
    """Build the locked-down prompt for claude-p compute_fn synthesis.

    The prompt has three jobs: pin the function signature, ban imports
    outside the function body, and inject the LLM's own formula so the
    response is grounded. We do NOT show the LLM any of our existing
    plugin source — we want behavior, not pattern-matching.
    """
    fn_name = f"compute_{spec.name.lower()}"
    param_keys = ", ".join(sorted(spec.default_params.keys())) or "(none)"
    quote_line = (
        f"Source quote: {source_quote.strip()[:600]!r}\n"
        if isinstance(source_quote, str) and source_quote.strip()
        else ""
    )
    return (
        "You write a single pure-Python function that computes a "
        "technical indicator from an OHLCV DataFrame. The function will "
        "be checked by `ruff check` and `mypy --strict` and rejected on "
        "any error, so write defensively.\n\n"
        "OUTPUT RULES — output ONLY the raw Python source of the "
        "function. No prose, no markdown fences, no import statements "
        "ANYWHERE (the scaffolder injects `pd` and `np` at module "
        "level for you), no extra definitions.\n\n"
        "REQUIRED SIGNATURE (verbatim, including type annotations):\n"
        f"    def {fn_name}(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:\n\n"
        "INSIDE THE FUNCTION BODY:\n"
        "  - `pd` (pandas) and `np` (numpy) are in scope — DO NOT "
        "`import` them. The scaffolder will detect uses and emit the "
        "right module-level imports for you.\n"
        "  - You MAY NOT import anything else, and you MAY NOT call "
        "exec / eval / __import__ / compile / open.\n\n"
        "INPUTS\n"
        "  - df has columns: open, high, low, close, volume (lowercase). "
        "Use `df['close']` etc. directly — they are `pd.Series`.\n"
        f"  - params keys: {param_keys}\n"
        "  - To coerce a param: `period = int(cast(int, params.get('period', 14)))` "
        "after `from typing import cast`. But `from typing import cast` "
        "is NOT allowed (no module-level imports outside the function). "
        "Instead, write it inline: e.g. `period_raw = params.get('period', 14); "
        "period = int(period_raw) if isinstance(period_raw, (int, float)) else 14`. "
        "Always handle the `object` static type with a runtime isinstance check.\n\n"
        "OUTPUT\n"
        "  - Return a pd.Series indexed exactly like df.index, same length as df.\n"
        "  - Use np.nan (after `import numpy as np`) for warmup bars where "
        "the indicator is undefined.\n\n"
        f"FORMULA TO IMPLEMENT (verbatim from a research extraction):\n{formula.strip()}\n\n"
        f"{quote_line}"
        "Now output ONLY the function definition. Begin with `def`."
    )


def _run_claude_codegen(
    prompt: str, *, timeout_seconds: int = CODEGEN_TIMEOUT_SECONDS
) -> str:
    """Shell out to ``claude -p --output-format json`` for a compute_fn body.

    Returns the unwrapped ``result`` text. Raises ``CodegenError`` for
    timeouts / non-zero exit / missing binary so the endpoint can map to
    ``codegen_failed`` with a precise error code. Intentionally narrow
    in scope vs ``extractor.ClaudePExtractor._run_claude`` — that one
    deals with extraction envelopes and per-finding parsing; here we
    just need one string back.
    """
    claude_path = shutil.which(CLAUDE_BIN)
    if claude_path is None:
        raise CodegenError("codegen_unavailable", f"{CLAUDE_BIN!r} CLI not on PATH")
    try:
        proc = subprocess.run(  # noqa: S603
            [claude_path, "-p", "--output-format", "json", prompt],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        raise CodegenError("timeout", f"{timeout_seconds}s") from e
    if proc.returncode != 0:
        raise CodegenError(
            "claude_exit", f"rc={proc.returncode} stderr={proc.stderr[:200]}"
        )
    raw = proc.stdout
    try:
        outer = json.loads(raw)
    except json.JSONDecodeError as e:
        raise CodegenError("non_json_envelope", str(e)) from e
    if not isinstance(outer, dict) or "result" not in outer:
        raise CodegenError("non_json_envelope", "missing 'result'")
    result = outer.get("result")
    if not isinstance(result, str):
        raise CodegenError("non_json_envelope", "result not a string")
    return _strip_code_fence(result).strip()


def _strip_code_fence(text: str) -> str:
    """Remove a leading ```python``` fence if the model emitted one anyway."""
    s = text.strip()
    if not s.startswith("```"):
        return text
    first_nl = s.find("\n")
    if first_nl == -1:
        return text
    body = s[first_nl + 1 :]
    if body.endswith("```"):
        body = body[: -3]
    return body.strip()


def _ast_safety_check(body: str, expected_fn_name: str) -> None:
    """Parse ``body`` and raise ``CodegenError`` if it's unsafe or wrong shape.

    Catches: syntax errors, missing top-level function, wrong function
    name, missing/incorrect signature annotations, banned imports
    anywhere in the tree, banned bare calls anywhere in the tree. We
    walk the entire AST (not just the first statement) so a hidden
    ``exec()`` inside a nested helper still gets caught.
    """
    try:
        tree = ast.parse(body)
    except SyntaxError as e:
        raise CodegenError("syntax_error", str(e)) from e

    if not tree.body:
        raise CodegenError("non_function", "empty body")

    top = tree.body[0]
    if not isinstance(top, ast.FunctionDef):
        raise CodegenError(
            "non_function", f"top-level is {type(top).__name__}, expected FunctionDef"
        )
    if top.name != expected_fn_name:
        raise CodegenError(
            "missing_signature",
            f"function is {top.name!r}, expected {expected_fn_name!r}",
        )
    if len(tree.body) > 1:
        raise CodegenError(
            "non_function",
            f"expected one top-level def, got {len(tree.body)} statements",
        )

    args = top.args.args
    if len(args) != 2 or [a.arg for a in args] != ["df", "params"]:
        raise CodegenError(
            "missing_signature", f"args must be (df, params), got {[a.arg for a in args]}"
        )
    if any(a.annotation is None for a in args) or top.returns is None:
        raise CodegenError("missing_signature", "missing type annotations")

    for node in ast.walk(top):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _BANNED_IMPORTS:
                    raise CodegenError("banned_import", root)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _BANNED_IMPORTS:
                raise CodegenError("banned_import", root)
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in _BANNED_CALLS:
                raise CodegenError("banned_call", fn.id)


def _strip_inner_imports_and_detect_np(body: str) -> tuple[str, bool]:
    """Strip ``import`` statements from inside the function body and
    detect whether the body references ``np``.

    Inner imports in the LLM-generated body trip ruff I001 (unsorted
    import block) because they live inside a ``def``. Pulling them up
    to module level — which we render ourselves — fixes the style.

    ``pd`` is always emitted at module level (the signature uses
    ``pd.DataFrame`` / ``pd.Series`` annotations, and a TC002 noqa
    silences the type-checking-only complaint). ``np`` is conditional:
    emitting it unconditionally would trip F401 when unused.
    """
    tree = ast.parse(body)
    func = tree.body[0]
    if not isinstance(func, ast.FunctionDef):
        return body, False
    func.body = [
        s for s in func.body if not isinstance(s, (ast.Import, ast.ImportFrom))
    ] or [ast.Pass()]
    uses_np = any(
        isinstance(node, ast.Name) and node.id == "np"
        for stmt in func.body
        for node in ast.walk(stmt)
    )
    return ast.unparse(tree), uses_np


def _indent(body: str, spaces: int = 0) -> str:
    if spaces == 0:
        return body
    pad = " " * spaces
    return "\n".join(pad + line if line else line for line in body.splitlines())


def _format_literal(value: Any) -> str:
    """Format a default-param value for inclusion in generated source."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return repr(value)
    return repr(value)


def _format_type(t: type) -> str:
    return {bool: "bool", int: "int", float: "float", str: "str"}.get(t, "object")


def render_plugin_file(
    spec: IndicatorSpecArgs,
    body: str,
    *,
    extraction_id: int,
    source_quote: str | None = None,
    now: datetime | None = None,
) -> str:
    """Render the full ``proposed_<name>.py`` plugin source from inputs.

    The output is mypy-strict / ruff clean by construction; the only
    variable surface area is the body — which is gated by
    ``_ast_safety_check`` before this is called.
    """
    ts = (now or datetime.now(UTC)).isoformat(timespec="seconds")
    ranges_line = (
        ", ".join(
            f"{k}={spec.range_provenance.get(k, 'unknown')}"
            for k in sorted(spec.param_ranges)
        )
        or "(no GA ranges)"
    )

    defaults_src = (
        "{"
        + ", ".join(
            f"{k!r}: {_format_literal(v)}" for k, v in spec.default_params.items()
        )
        + "}"
    )
    schema_src = (
        "{"
        + ", ".join(
            f"{k!r}: {_format_type(t)}" for k, t in spec.param_schema.items()
        )
        + "}"
    )
    ranges_src = (
        "{"
        + ", ".join(
            f"{k!r}: ({lo!r}, {hi!r})" for k, (lo, hi) in spec.param_ranges.items()
        )
        + "}"
    )
    threshold_src = (
        f"({spec.threshold_range[0]!r}, {spec.threshold_range[1]!r})"
        if spec.threshold_range is not None
        else "None"
    )
    quote_block = ""
    if isinstance(source_quote, str) and source_quote.strip():
        # Indent each line so the quote reads as a block inside the
        # docstring; strip triple-quotes that would terminate the
        # enclosing docstring early.
        cleaned = source_quote.strip().replace('"""', '"​""')[:400]
        indented = "\n".join("    " + line for line in cleaned.splitlines())
        quote_block = f"\nSource quote:\n{indented}\n"

    fn_name = f"compute_{spec.name.lower()}"

    cleaned_body, uses_np = _strip_inner_imports_and_detect_np(body)
    # Order matters for ruff isort: stdlib → third-party → first-party.
    # pandas is always a runtime import — most LLM bodies use pd at
    # runtime (pd.Series, pd.concat, etc.). TC002 noqa silences the
    # annotation-only-import complaint for the bodies that don't.
    np_line = "import numpy as np\n" if uses_np else ""

    return (
        f'"""AUTO-GENERATED FROM EXTRACTION {extraction_id} ON {ts} '
        f'— review before promoting.\n'
        f'\n'
        f'RANGES: {ranges_line}\n'
        f'Display: {spec.display_name}\n'
        f'Description: {spec.description}\n'
        f'{quote_block}'
        f'"""\n'
        f'\n'
        f'from __future__ import annotations\n'
        f'\n'
        f'{np_line}'
        f'import pandas as pd  # noqa: TC002\n'
        f'\n'
        f'from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry\n'
        f'\n'
        f'\n'
        f'{cleaned_body.rstrip()}\n'
        f'\n'
        f'\n'
        f'indicator_registry.register_spec(\n'
        f'    IndicatorSpec(\n'
        f'        name={spec.name!r},\n'
        f'        nt_class=None,\n'
        f'        pandas_ta_func=None,\n'
        f'        default_params={defaults_src},\n'
        f'        param_schema={schema_src},\n'
        f'        compute_fn={fn_name},\n'
        f'        display_name={spec.display_name!r},\n'
        f'        description={spec.description!r},\n'
        f'        category={spec.category!r},\n'
        f'        param_ranges={ranges_src},\n'
        f'        threshold_range={threshold_src},\n'
        f'    )\n'
        f')\n'
    )


def _run_tool(argv: list[str]) -> tuple[bool, str]:
    """Run a subprocess and return (passed, combined output, truncated)."""
    try:
        proc = subprocess.run(  # noqa: S603
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError as e:
        return False, f"tool not found: {e}"
    except subprocess.TimeoutExpired:
        return False, "tool timed out after 60s"
    combined = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, combined[:500]


def run_mypy(path: Path) -> tuple[bool, str]:
    """Type-check a single plugin file with the project's mypy config."""
    return _run_tool([sys.executable, "-m", "mypy", "--no-color-output", str(path)])


def run_ruff(path: Path) -> tuple[bool, str]:
    """Lint a single plugin file with the project's ruff config."""
    return _run_tool([sys.executable, "-m", "ruff", "check", "--no-fix", str(path)])


def plugin_path_for(name: str) -> Path:
    """Resolve where ``proposed_<name>.py`` lives. Tests monkeypatch PLUGINS_DIR."""
    return PLUGINS_DIR / f"proposed_{name.lower()}.py"


def write_plugin_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def synthesize_and_write(
    spec: IndicatorSpecArgs,
    *,
    formula: str,
    extraction_id: int,
    source_quote: str | None,
    runner: Any = None,
) -> Path:
    """End-to-end: prompt → claude-p → AST gate → render → write → mypy + ruff.

    Returns the written path on success. Raises ``CodegenError`` with a
    machine-readable ``code`` on any failure; the caller (the endpoint)
    surfaces that as ``status=codegen_failed``. On a post-write failure
    (mypy/ruff) the file IS deleted so a half-broken plugin can't
    poison the next ``load_builtin_plugins`` call.

    ``runner`` is injected so tests can swap in a callable that returns
    a canned body without spawning a real claude subprocess.
    """
    fn_name = f"compute_{spec.name.lower()}"
    prompt = _build_codegen_prompt(spec, formula, source_quote)
    call = runner if runner is not None else _run_claude_codegen
    body = call(prompt)
    if not isinstance(body, str) or not body.strip():
        raise CodegenError("empty_body", "runner returned empty string")

    _ast_safety_check(body, fn_name)

    rendered = render_plugin_file(
        spec, body, extraction_id=extraction_id, source_quote=source_quote
    )
    path = plugin_path_for(spec.name)
    write_plugin_file(path, rendered)

    try:
        ok, output = run_mypy(path)
        if not ok:
            raise CodegenError("mypy_fail", output)
        ok, output = run_ruff(path)
        if not ok:
            raise CodegenError("ruff_fail", output)
    except CodegenError:
        # Don't leave a broken plugin on disk — the next process restart
        # would import it and the registry would surface a load error.
        path.unlink(missing_ok=True)
        raise
    return path


# ---------------------------------------------------------------------------
# Slice 3: contract test gen + pytest run + auto-commit
# ---------------------------------------------------------------------------


TESTS_DIR: Path = (
    Path(__file__).resolve().parent.parent.parent
    / "tests"
    / "unit"
    / "test_plugins"
)

PYTEST_TIMEOUT_SECONDS = 30
COMMIT_MESSAGE_TEMPLATE = "chore: scaffold proposed indicator {name} (bd-3p1k)"
CO_AUTHOR_TRAILER = "Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"


class ScaffoldError(Exception):
    """Test- or commit-stage failure. The endpoint maps these to ``test_failed``.

    Carries a ``code`` (``test_failed`` | ``commit_failed``) and an
    ``output`` blob (first 2KB of pytest / git output) that the caller
    surfaces to the user verbatim — they're the only signal of what
    actually broke when a hand-written commit hook or a model-generated
    body misbehaves.
    """

    OUTPUT_LIMIT = 2048

    def __init__(self, code: str, output: str = "") -> None:
        super().__init__(f"{code}: {output[:200]}" if output else code)
        self.code = code
        self.output = output[: self.OUTPUT_LIMIT]


def test_path_for(name: str) -> Path:
    """Resolve where ``test_proposed_<name>.py`` lives. Monkeypatched in tests."""
    return TESTS_DIR / f"test_proposed_{name.lower()}.py"


_CONTRACT_TEST_SINGLE = '''"""Auto-generated contract test for proposed indicator {name_upper}.

Generated as part of the scaffold pipeline (bd-3p1k.1.3) — verifies the
synthesized compute_fn produces an output of the right shape and
isn't all-NaN past warmup. Re-run after editing the plugin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vibe_quant.dsl.indicators import indicator_registry, invoke_compute_fn


def _sample_ohlcv(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(42)
    close = 100.0 + rng.standard_normal(n).cumsum()
    return pd.DataFrame(
        {{
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.uniform(1000.0, 10000.0, n),
        }},
        index=idx,
    )


def test_{name_lower}_registered() -> None:
    spec = indicator_registry.get("{name_upper}")
    assert spec is not None, "{name_upper} plugin did not register"
    assert spec.compute_fn is not None


def test_{name_lower}_contract_length_and_index() -> None:
    spec = indicator_registry.get("{name_upper}")
    assert spec is not None
    df = _sample_ohlcv()
    out = invoke_compute_fn(spec, df, spec.default_params)
    assert isinstance(out, pd.Series)
    assert len(out) == len(df)
    assert out.index.equals(df.index)


def test_{name_lower}_not_all_nan_past_warmup() -> None:
    spec = indicator_registry.get("{name_upper}")
    assert spec is not None
    df = _sample_ohlcv()
    out = invoke_compute_fn(spec, df, spec.default_params)
    assert isinstance(out, pd.Series)
    # Past the second half of the series we expect at least one finite
    # value; a fully-NaN tail means the body computes nothing.
    tail = out.iloc[len(out) // 2 :]
    assert tail.notna().any(), "{name_upper} produced all-NaN past warmup"
'''


_CONTRACT_TEST_MULTI = '''"""Auto-generated contract test for proposed multi-output indicator {name_upper}.

Generated as part of the scaffold pipeline (bd-3p1k.1.3) — verifies the
compute_fn returns a dict keyed by every declared output_name, each
Series index-aligned to the input frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vibe_quant.dsl.indicators import indicator_registry, invoke_compute_fn


_OUTPUTS = {outputs_tuple}


def _sample_ohlcv(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(42)
    close = 100.0 + rng.standard_normal(n).cumsum()
    return pd.DataFrame(
        {{
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.uniform(1000.0, 10000.0, n),
        }},
        index=idx,
    )


def test_{name_lower}_registered() -> None:
    spec = indicator_registry.get("{name_upper}")
    assert spec is not None
    assert spec.compute_fn is not None
    assert tuple(spec.output_names) == _OUTPUTS


def test_{name_lower}_contract_dict_and_alignment() -> None:
    spec = indicator_registry.get("{name_upper}")
    assert spec is not None
    df = _sample_ohlcv()
    out = invoke_compute_fn(spec, df, spec.default_params)
    assert isinstance(out, dict)
    for key in _OUTPUTS:
        assert key in out, f"missing declared output {{key!r}}"
        series = out[key]
        assert len(series) == len(df)
        assert series.index.equals(df.index)


def test_{name_lower}_not_all_nan_past_warmup() -> None:
    spec = indicator_registry.get("{name_upper}")
    assert spec is not None
    df = _sample_ohlcv()
    out = invoke_compute_fn(spec, df, spec.default_params)
    assert isinstance(out, dict)
    for key, series in out.items():
        tail = series.iloc[len(series) // 2 :]
        assert tail.notna().any(), (
            f"{name_upper} output {{key!r}} all-NaN past warmup"
        )
'''


def render_contract_test(spec: IndicatorSpecArgs) -> str:
    """Render the auto-generated contract test for a scaffolded indicator.

    Picks the single-output or multi-output template based on
    ``spec.output_names``. The output is a Python source string the
    caller can write to ``test_path_for(spec.name)``.
    """
    name_lower = spec.name.lower()
    if len(spec.output_names) > 1:
        outputs_tuple = repr(tuple(spec.output_names))
        return _CONTRACT_TEST_MULTI.format(
            name_upper=spec.name,
            name_lower=name_lower,
            outputs_tuple=outputs_tuple,
        )
    return _CONTRACT_TEST_SINGLE.format(
        name_upper=spec.name, name_lower=name_lower
    )


def run_contract_test(
    test_path: Path, *, timeout_seconds: int = PYTEST_TIMEOUT_SECONDS
) -> tuple[bool, str]:
    """Run ``pytest -x`` on a single contract test file.

    Returns ``(passed, output_first_2kb)``. ``PYTHONDONTWRITEBYTECODE=1``
    keeps the plugin dir free of ``__pycache__`` clutter so the auto-
    commit step doesn't include stray byproducts in ``git status``.
    """
    import os
    import sys

    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "pytest", "-x", "-q", str(test_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        return False, f"pytest timed out after {timeout_seconds}s: {e}"
    combined = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, combined[: ScaffoldError.OUTPUT_LIMIT]


def _git(*args: str, cwd: Path | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(  # noqa: S603, S607
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd is not None else None,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def git_commit_scaffold(
    plugin_path: Path,
    test_path: Path,
    *,
    name: str,
    repo_root: Path | None = None,
) -> str:
    """Stage + commit the plugin and test files. Returns the commit SHA.

    Raises ``ScaffoldError("commit_failed", output)`` if any git step
    fails — the caller treats this identically to a test failure
    (delete both files, surface the output). NEVER pushes; the user
    decides when to publish.
    """
    rc, _, err = _git("add", str(plugin_path), str(test_path), cwd=repo_root)
    if rc != 0:
        raise ScaffoldError("commit_failed", f"git add failed: {err}")

    message = COMMIT_MESSAGE_TEMPLATE.format(name=name)
    full_message = f"{message}\n\n{CO_AUTHOR_TRAILER}\n"
    rc, _, err = _git("commit", "-m", full_message, cwd=repo_root)
    if rc != 0:
        # Pre-commit hook (or anything else) rejected the commit.
        # Unstage so the working tree is clean again for the caller's
        # cleanup pass.
        _git("restore", "--staged", str(plugin_path), str(test_path), cwd=repo_root)
        raise ScaffoldError("commit_failed", err or "git commit failed")

    rc, out, err = _git("rev-parse", "HEAD", cwd=repo_root)
    if rc != 0:
        raise ScaffoldError("commit_failed", f"rev-parse failed: {err}")
    return out.strip()


@dataclass
class ScaffoldResult:
    """Successful end-to-end scaffold outcome (plugin + test + commit)."""

    plugin_path: Path
    test_path: Path
    commit_sha: str


# ---------------------------------------------------------------------------
# bd-3p1k.3: promote indicator (rename to drop ``proposed_`` prefix)
# ---------------------------------------------------------------------------


PROMOTE_COMMIT_TEMPLATE = "chore: promote indicator {name} (bd-3p1k)"


class PromoteError(Exception):
    """Promotion failed. ``code`` is the machine vocabulary the endpoint surfaces.

    Codes: ``invalid_name``, ``not_found``, ``collision``, ``write_failed``,
    ``commit_failed``, ``bd_failed``.
    """

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def proposed_path_for(name: str) -> Path:
    """Resolve where the ``proposed_<name>.py`` plugin lives on disk."""
    return PLUGINS_DIR / f"proposed_{name.lower()}.py"


def promoted_path_for(name: str) -> Path:
    """Resolve where the promoted ``<name>.py`` plugin would live."""
    return PLUGINS_DIR / f"{name.lower()}.py"


def strip_auto_generated_header(source: str) -> str:
    """Remove the leading ``AUTO-GENERATED ...`` module docstring.

    The scaffolder always emits the file with the docstring as the very
    first triple-quoted block (``\"\"\"AUTO-GENERATED FROM EXTRACTION ...``).
    We find the opening ``\"\"\"`` and the next closing ``\"\"\"`` after it,
    and slice that span out. If the file doesn't start with such a header,
    return the source unchanged — a hand-edited plugin should round-trip.
    """
    s = source.lstrip("\n")
    if not s.startswith('"""'):
        return source
    end = s.find('"""', 3)
    if end < 0:
        return source
    body = s[end + 3 :].lstrip("\n")
    return body


def write_promoted_plugin(
    *, name: str, force: bool = False
) -> tuple[Path, Path]:
    """Atomically rename the proposed plugin file, stripping the header.

    Sequence:
    1. Read ``proposed_<name>.py`` (or raise ``not_found``).
    2. Refuse if ``<name>.py`` already exists (``collision``) unless
       ``force=True`` — keeps the user from clobbering a hand-promoted file.
    3. Write the cleaned source to ``<name>.py`` first.
    4. Only then delete ``proposed_<name>.py``.

    Returns ``(old_path, new_path)`` so the caller can hand both to git.
    Raises ``PromoteError`` with a precise ``code`` on any failure.
    """
    if not _NAME_RE.match(name):
        raise PromoteError("invalid_name", f"{name!r} not a valid uppercase identifier")
    old = proposed_path_for(name)
    new = promoted_path_for(name)
    if not old.exists():
        raise PromoteError("not_found", f"{old.name} does not exist")
    if new.exists() and not force:
        raise PromoteError("collision", f"{new.name} already exists")

    source = old.read_text(encoding="utf-8")
    stripped = strip_auto_generated_header(source)

    # Atomic write: write to a temp file in the same directory, then rename
    # over the target. Same-dir rename is atomic on POSIX. We only unlink
    # the old file after the new file is fully on disk.
    tmp = new.with_suffix(new.suffix + ".tmp")
    try:
        tmp.write_text(stripped, encoding="utf-8")
        tmp.replace(new)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise PromoteError("write_failed", str(e)) from e
    old.unlink()
    return old, new


def bd_remember_indicator(
    *,
    name: str,
    extraction_id: int | None,
    source_url: str | None,
    bd_bin: str = "bd",
    timeout_seconds: int = 10,
) -> tuple[bool, str]:
    """Record indicator provenance in beads memory. Returns (ok, stderr/output).

    Best-effort: failures (bd not installed, command error) are reported
    to the caller but do NOT abort the promotion. The plugin file move
    has already happened; this is just metadata.
    """
    parts = [f"indicator:{name.lower()}"]
    if extraction_id is not None:
        parts.append(f"from extraction {extraction_id}")
    if source_url:
        parts.append(f"source {source_url}")
    fact = " — ".join(parts)
    key = f"indicator:{name.lower()}"
    bd_path = shutil.which(bd_bin)
    if bd_path is None:
        return False, f"{bd_bin!r} CLI not on PATH"
    try:
        proc = subprocess.run(  # noqa: S603
            [bd_path, "remember", fact, "--key", key],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        return False, f"bd remember timed out: {e}"
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "bd remember failed")[:500]
    return True, (proc.stdout or "")[:500]


def git_commit_promotion(
    old_path: Path,
    new_path: Path,
    *,
    name: str,
    repo_root: Path | None = None,
) -> str:
    """Stage the (deleted, added) pair and commit. Returns the new commit SHA.

    ``git add -A`` on both paths captures both the deletion of the proposed
    file and the addition of the promoted one in a single index update.
    """
    rc, _, err = _git("add", "-A", str(old_path), str(new_path), cwd=repo_root)
    if rc != 0:
        raise PromoteError("commit_failed", f"git add failed: {err}")

    message = PROMOTE_COMMIT_TEMPLATE.format(name=name)
    full_message = f"{message}\n\n{CO_AUTHOR_TRAILER}\n"
    rc, _, err = _git("commit", "-m", full_message, cwd=repo_root)
    if rc != 0:
        # Best-effort unstage so the index is clean for the caller's recovery.
        _git("restore", "--staged", str(old_path), str(new_path), cwd=repo_root)
        raise PromoteError("commit_failed", err or "git commit failed")

    rc, out, err = _git("rev-parse", "HEAD", cwd=repo_root)
    if rc != 0:
        raise PromoteError("commit_failed", f"rev-parse failed: {err}")
    return out.strip()


@dataclass
class PromoteResult:
    """Successful outcome of ``promote_indicator``."""

    old_path: Path
    new_path: Path
    commit_sha: str
    bd_remember_ok: bool
    bd_remember_output: str


def promote_indicator(
    name: str,
    *,
    extraction_id: int | None,
    source_url: str | None,
    repo_root: Path | None = None,
    force: bool = False,
) -> PromoteResult:
    """End-to-end: rename + strip header + git commit + bd remember.

    Order matters:
    1. Rename first (so a git_commit failure leaves the user with a clean
       file system — the new plugin works, just isn't committed yet).
    2. Commit the rename.
    3. ``bd remember`` last — provenance is informational, never blocks.

    Raises ``PromoteError`` on stages 1-2. Stage 3 (bd) failures are
    surfaced in the returned ``bd_remember_ok`` flag instead.
    """
    old, new = write_promoted_plugin(name=name, force=force)
    try:
        sha = git_commit_promotion(old, new, name=name, repo_root=repo_root)
    except PromoteError:
        # Roll back the rename so the user is in a consistent state and
        # can retry. Best-effort — if the unlink itself fails, the
        # commit_failed surface area is still what they care about.
        import contextlib

        with contextlib.suppress(OSError):
            new.unlink(missing_ok=True)
        raise
    bd_ok, bd_out = bd_remember_indicator(
        name=name, extraction_id=extraction_id, source_url=source_url
    )
    return PromoteResult(
        old_path=old,
        new_path=new,
        commit_sha=sha,
        bd_remember_ok=bd_ok,
        bd_remember_output=bd_out,
    )


def scaffold_full(
    spec: IndicatorSpecArgs,
    *,
    formula: str,
    extraction_id: int,
    source_quote: str | None,
    runner: Any = None,
    repo_root: Path | None = None,
) -> ScaffoldResult:
    """End-to-end scaffold: codegen → file → contract test → pytest → commit.

    Returns ``ScaffoldResult`` on success. Raises ``CodegenError`` if
    codegen / AST / mypy / ruff fails (file is already cleaned up by
    ``synthesize_and_write``). Raises ``ScaffoldError`` if the contract
    test fails or the git commit fails — in that case BOTH the plugin
    file AND the test file are deleted so a broken pair never lingers
    on disk.

    ``runner`` and ``repo_root`` are injection points for tests; in
    production both are ``None`` and the module defaults apply.
    """
    plugin_path = synthesize_and_write(
        spec,
        formula=formula,
        extraction_id=extraction_id,
        source_quote=source_quote,
        runner=runner,
    )

    test_path = test_path_for(spec.name)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text(render_contract_test(spec), encoding="utf-8")

    try:
        ok, output = run_contract_test(test_path)
        if not ok:
            raise ScaffoldError("test_failed", output)
        sha = git_commit_scaffold(
            plugin_path, test_path, name=spec.name, repo_root=repo_root
        )
    except ScaffoldError:
        plugin_path.unlink(missing_ok=True)
        test_path.unlink(missing_ok=True)
        raise

    return ScaffoldResult(
        plugin_path=plugin_path, test_path=test_path, commit_sha=sha
    )
