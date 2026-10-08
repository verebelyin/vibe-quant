"""Canonical DSL identity keys.

Discovery promote dedupe and validation consistency reference matching must
agree byte-for-byte on what "the same strategy" means: canonical JSON of a DSL
dict minus its ``name``. These two call paths used to duplicate the formula
privately (``_dsl_body`` / ``_dsl_body_key``); if they ever diverge, matching
silently fails. This module is the single source of truth for that key.
"""

import json


def dsl_body_key(dsl: dict[str, object]) -> str:
    """Canonical JSON of a DSL dict minus ``name`` (the strategy's identity)."""
    body = {k: v for k, v in dsl.items() if k != "name"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
