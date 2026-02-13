#!/usr/bin/env bash
set -euo pipefail

echo "[quality:style-debt] collecting full-repo lint/type debt (non-blocking)"

set +e
uv run --extra dev ruff check .
ruff_status=$?

uv run --extra dev mypy .
mypy_status=$?
set -e

echo "[quality:style-debt] ruff exit code: ${ruff_status}"
echo "[quality:style-debt] mypy exit code: ${mypy_status}"
echo "[quality:style-debt] done (always exits 0)"
