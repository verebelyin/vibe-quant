#!/bin/sh
# Start the local Vercel Board (boardctl.py up): idempotent, prints its URL.
exec python3 "$(dirname "$0")/boardctl.py" up "$@"
