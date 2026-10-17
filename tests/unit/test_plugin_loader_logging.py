"""Plugin-loader log-level hygiene: per-plugin "Loaded ..." lines are DEBUG only.

Every ``python -m vibe_quant.*`` invocation loads the built-in indicator plugins and
used to print one ``Loaded indicator plugin: ...`` line per plugin at INFO. Those are
diagnostics, not operational events — agents had to filter them out by hand. The two
``Loaded ... indicator plugin`` calls in ``vibe_quant/dsl/plugin_loader.py`` are now
``logger.debug``; failure (``_record_failure``) and override notices stay at
warning/error/info.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vibe_quant.dsl import plugin_loader

if TYPE_CHECKING:
    import pytest


def test_loaded_plugin_lines_are_debug_not_info(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # DEBUG captures both levels, so a single load proves what moved.
    with caplog.at_level(logging.DEBUG, logger="vibe_quant.dsl.plugin_loader"):
        loaded = plugin_loader.load_builtin_plugins()

    assert loaded, "expected the built-in plugin directory to load at least one plugin"

    loaded_records = [r for r in caplog.records if "Loaded" in r.getMessage()]
    at_info_or_above = [r for r in loaded_records if r.levelno >= logging.INFO]
    at_debug = [r for r in loaded_records if r.levelno == logging.DEBUG]

    assert at_info_or_above == [], (
        "plugin-load success lines must not reach INFO: "
        f"{[r.getMessage() for r in at_info_or_above]}"
    )
    assert at_debug, "expected a DEBUG record for each successfully loaded plugin"
