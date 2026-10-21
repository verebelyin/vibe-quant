"""Fast-tier errors."""

from __future__ import annotations


class FastTierUnsupported(ValueError):
    """A genome or run shape the fast tier cannot reproduce exactly (never approximated)."""
