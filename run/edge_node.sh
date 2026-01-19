#!/usr/bin/env bash
# Run Public Relay (edge node)
# Usage: ./run/edge_node.sh [upstream] [port]
# Env vars:
#   NILZ_NODE_UPSTREAM   Upstream private node (default http://127.0.0.1:5000)
#   NILZ_PUBLIC_HOST     Bind host (default 0.0.0.0)
#   NILZ_PUBLIC_PORT     Bind port (default 5010)
#   NILZ_PUBLIC_ID       Identifier label
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
export NILZ_NODE_UPSTREAM="${1:-${NILZ_NODE_UPSTREAM:-http://127.0.0.1:5000}}"
export NILZ_PUBLIC_PORT="${2:-${NILZ_PUBLIC_PORT:-5010}}"
export NILZ_PUBLIC_HOST="${NILZ_PUBLIC_HOST:-0.0.0.0}"
export NILZ_PUBLIC_ID="${NILZ_PUBLIC_ID:-public-relay}";
if [[ -d .venv ]]; then source .venv/bin/activate; fi
exec python node/edge_node.py 2>&1 | sed "s/^/[relay] /" || true
