#!/usr/bin/env bash
# Run NilzCoin core node
# Usage: ./run/node.sh [host] [port]
# Env overrides:
#   NILZ_BLOCK_TIME_S        Target seconds per block (default 30)
#   NILZ_RETARGET_WINDOW     Blocks averaged for difficulty retarget (default 30)
#
# Example:
#   NILZ_BLOCK_TIME_S=20 ./run/node.sh 127.0.0.1 5001
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Auto-load unified env file if present
ENV_FILE="$ROOT/run/.env"
if [[ -f "$ENV_FILE" ]]; then
	set -a
	source "$ENV_FILE"
	set +a
fi
cd "$ROOT"
HOST="${1:-0.0.0.0}"; PORT="${2:-5000}";
if [[ -d .venv ]]; then source .venv/bin/activate; fi
exec python node/server.py --host "$HOST" --port "$PORT" 2>&1 | sed "s/^/[node] /" || true
