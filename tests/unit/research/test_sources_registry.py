"""Auto-discovery + registry behaviour for research sources."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest

from vibe_quant.research import sources as sources_pkg
from vibe_quant.research.sources import (
    _reset_for_tests,
    get_load_errors,
    get_source,
    list_sources,
    load_builtin_sources,
    register_source,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate_registry() -> None:
    _reset_for_tests()
    yield
    _reset_for_tests()


def _write_source_file(name: str, body: str) -> "Path":
    """Drop a source file alongside the package and return its path."""
    pkg_dir = sources_pkg.__path__[0]
    p = __import__("pathlib").Path(pkg_dir) / f"{name}.py"
    p.write_text(textwrap.dedent(body))
    return p


def test_load_then_list(tmp_path: "Path") -> None:
    p = _write_source_file(
        "_test_dummy_source",
        '''
        from vibe_quant.research.sources import register_source

        @register_source("dummy")
        class DummySource:
            name = "dummy"
            def fetch(self, since, limit):
                yield from ()
        ''',
    )
    # Files with leading underscore are skipped — verify that path
    load_builtin_sources(force=True)
    assert "dummy" not in list_sources()
    p.unlink()


def test_drop_source_in_dir_is_discovered(tmp_path: "Path") -> None:
    p = _write_source_file(
        "fake_test_source_for_unit",
        '''
        from vibe_quant.research.sources import register_source

        @register_source("fake_unit")
        class FakeUnitSource:
            name = "fake_unit"
            def fetch(self, since, limit):
                yield from ()
        ''',
    )
    try:
        load_builtin_sources(force=True)
        assert "fake_unit" in list_sources()
    finally:
        p.unlink()
        # remove from sys.modules so subsequent loads don't re-register
        import sys
        sys.modules.pop("vibe_quant.research.sources.fake_test_source_for_unit", None)


def test_get_source_unknown_raises_with_message() -> None:
    with pytest.raises(KeyError) as exc:
        get_source("nonexistent")
    assert "nonexistent" in str(exc.value)
    assert "Available" in str(exc.value)


def test_duplicate_register_raises_value_error() -> None:
    @register_source("dupname")
    class A:
        name = "dupname"
        def fetch(self, since, limit):  # noqa: ARG002
            yield from ()

    with pytest.raises(ValueError, match="dupname"):
        @register_source("dupname")
        class B:
            name = "dupname"
            def fetch(self, since, limit):  # noqa: ARG002
                yield from ()

    # registry still intact, A still resolvable
    assert get_source("dupname") is A


def test_class_missing_fetch_raises_type_error_at_registration() -> None:
    with pytest.raises(TypeError, match="fetch"):

        @register_source("nofetch")
        class NoFetch:
            name = "nofetch"


def test_broken_source_file_does_not_break_others() -> None:
    broken = _write_source_file(
        "broken_source_for_unit",
        '''
        raise ImportError("intentionally broken for unit test")
        ''',
    )
    good = _write_source_file(
        "good_source_for_unit",
        '''
        from vibe_quant.research.sources import register_source

        @register_source("good_unit")
        class GoodUnitSource:
            name = "good_unit"
            def fetch(self, since, limit):
                yield from ()
        ''',
    )
    try:
        load_builtin_sources(force=True)
        assert "good_unit" in list_sources()
        errs = get_load_errors()
        assert any("broken_source_for_unit" in e.module for e in errs)
    finally:
        broken.unlink()
        good.unlink()
        import sys
        sys.modules.pop("vibe_quant.research.sources.broken_source_for_unit", None)
        sys.modules.pop("vibe_quant.research.sources.good_source_for_unit", None)


def test_load_is_idempotent_when_not_forced() -> None:
    @register_source("idem")
    class Idem:
        name = "idem"
        def fetch(self, since, limit):  # noqa: ARG002
            yield from ()

    first = load_builtin_sources()
    second = load_builtin_sources()
    assert first == second
