"""Basic package tests."""

import vibe_quant


def test_version() -> None:
    """Test package version is set."""
    assert vibe_quant.__version__ == "0.1.0"


def test_screening_main_importable() -> None:
    """vibe_quant.screening.__main__ should be importable."""
    import importlib
    mod = importlib.import_module("vibe_quant.screening.__main__")
    assert hasattr(mod, "main")


def test_validation_main_importable() -> None:
    """vibe_quant.validation.__main__ should be importable."""
    import importlib
    mod = importlib.import_module("vibe_quant.validation.__main__")
    assert hasattr(mod, "main")
