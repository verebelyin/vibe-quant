#!/usr/bin/env bash
# Start backend + frontend for local development
# Usage: ./dev.sh

trap 'kill 0' EXIT

echo "Starting backend on :8000..."
.venv/bin/uvicorn vibe_quant.api.app:create_app --factory --port 8000 --reload &

echo "Starting frontend on :5173..."
cd frontend && pnpm dev --port 5173 &

wait
