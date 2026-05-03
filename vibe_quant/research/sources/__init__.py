"""Research source plugin system.

Drop a `.py` file in this directory that defines a class implementing
`ResearchSource` and decorate it with `@register_source("name")`. The class
will be auto-loaded at startup and exposed via `list_sources()` /
`get_source(name)`.

See `README.md` in this directory for a worked example.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from datetime import datetime

    from vibe_quant.research.schema import RawItem

logger = logging.getLogger(__name__)


@runtime_checkable
class ResearchSource(Protocol):
    """Protocol every research source must satisfy.

    The `name` class attribute is the registry key. `fetch` yields RawItems
    one at a time so callers can stream them into the archive without
    buffering an entire response.
    """

    name: ClassVar[str]

    def fetch(self, since: datetime | None, limit: int) -> Iterable[RawItem]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SourceLoadError:
    module: str
    error_type: str
    message: str


_registry: dict[str, type[ResearchSource]] = {}
_load_errors: list[SourceLoadError] = []
_loaded_modules: list[str] = []
_loaded: bool = False

T = TypeVar("T", bound=type[ResearchSource])


def register_source(name: str) -> Callable[[T], T]:
    """Decorator: register a class as a named research source.

    Raises:
        ValueError: If `name` is already registered.
        TypeError:  If the class doesn't define `fetch`.
    """

    def decorator(cls: T) -> T:
        if not hasattr(cls, "fetch") or not callable(cls.fetch):
            raise TypeError(
                f"ResearchSource {cls.__name__!r} must define a callable `fetch` method"
            )
        if name in _registry:
            existing = _registry[name].__name__
            raise ValueError(
                f"Research source {name!r} already registered (existing: {existing!r}, "
                f"attempted: {cls.__name__!r})"
            )
        cls.name = name  # type: ignore[misc]
        _registry[name] = cls
        return cls

    return decorator


def get_source(name: str) -> type[ResearchSource]:
    if name not in _registry:
        available = sorted(_registry.keys()) or ["<none>"]
        raise KeyError(
            f"No research source registered for {name!r}. Available: {', '.join(available)}"
        )
    return _registry[name]


def list_sources() -> list[str]:
    return sorted(_registry.keys())


def get_load_errors() -> list[SourceLoadError]:
    return list(_load_errors)


def load_builtin_sources(*, force: bool = False) -> list[str]:
    """Import every `.py` under this package so its `@register_source` runs.

    Idempotent: subsequent calls return the cached load list unless `force=True`.
    A failing module is logged and recorded in `_load_errors` — other sources
    still load.
    """
    global _loaded
    if _loaded and not force:
        return list(_loaded_modules)

    _load_errors.clear()
    _loaded_modules.clear()

    package = importlib.import_module(__name__)
    for module_info in pkgutil.iter_modules(package.__path__):
        name = module_info.name
        if name.startswith("_"):
            continue
        qualified = f"{__name__}.{name}"
        try:
            importlib.import_module(qualified)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load research source %s: %s", qualified, exc)
            _load_errors.append(
                SourceLoadError(
                    module=qualified,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            continue
        _loaded_modules.append(qualified)
        logger.info("Loaded research source: %s", qualified)

    _loaded = True
    return list(_loaded_modules)


def _reset_for_tests() -> None:
    """Test-only hook: clear the registry + loaded flag."""
    global _loaded
    _registry.clear()
    _load_errors.clear()
    _loaded_modules.clear()
    _loaded = False
