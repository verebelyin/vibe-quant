"""Reload-safety: a registry reset must not leave the builtins unloadable."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import pytest

from vibe_quant.research.sources import (
    _reset_for_tests,
    get_load_errors,
    get_source,
    list_sources,
    load_builtin_sources,
    register_source,
)

if TYPE_CHECKING:
    from collections.abc import Generator

_BUILTIN_MODULES = (
    "vibe_quant.research.sources.hackernews",
    "vibe_quant.research.sources.reddit",
)


@pytest.fixture(autouse=True)
def _isolate_registry() -> Generator[None]:
    _reset_for_tests()
    yield
    _reset_for_tests()


def _cache_builtin_modules() -> None:
    """Put the source submodules in `sys.modules`, as a startup load does."""
    for qualified in _BUILTIN_MODULES:
        importlib.import_module(qualified)


def test_forced_load_re_registers_builtins_after_reset() -> None:
    """`tests/unit/api/test_research_router*.py` reset the registry mid-process.

    The submodules stay in `sys.modules`, so a forced load has to re-run their
    `@register_source` decorators instead of returning the cached modules.
    """
    _cache_builtin_modules()

    _reset_for_tests()
    assert list_sources() == []

    load_builtin_sources(force=True)
    assert {"hackernews", "reddit"} <= set(list_sources())


def test_forced_load_twice_re_registers_without_load_errors() -> None:
    _cache_builtin_modules()

    load_builtin_sources(force=True)
    assert {"hackernews", "reddit"} <= set(list_sources())

    before = list_sources()
    load_builtin_sources(force=True)
    assert list_sources() == before
    assert get_load_errors() == []


def test_non_forced_load_after_reset_keeps_cached_registrations() -> None:
    """A non-forced load must not re-run the cached decorators.

    `tests/unit/api/test_research_router*.py` register a fake `reddit` and then
    call an endpoint whose handler runs `load_builtin_sources()`; a re-import
    there would collide with that stub instead of leaving it alone.
    """
    _cache_builtin_modules()
    _reset_for_tests()

    @register_source("reddit")
    class FakeReddit:
        name = "reddit"

        def fetch(self, since, limit):  # noqa: ARG002
            yield from ()

    load_builtin_sources()
    assert get_source("reddit") is FakeReddit
    assert "hackernews" not in list_sources()
    assert get_load_errors() == []
