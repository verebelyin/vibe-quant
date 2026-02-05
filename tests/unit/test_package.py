"""Basic package tests."""

import vibe_quant


def test_version() -> None:
    """Test package version is set."""
    assert vibe_quant.__version__ == "0.1.0"
