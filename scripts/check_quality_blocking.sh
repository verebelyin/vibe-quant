#!/usr/bin/env bash
set -euo pipefail

echo "[quality:blocking] ruff (runtime package)"
uv run --extra dev ruff check vibe_quant

echo "[quality:blocking] mypy (runtime package)"
uv run --extra dev mypy vibe_quant

echo "[quality:blocking] complete"
