#!/usr/bin/env bash
# Run Mining Edge gateway
# Usage: ./run/edge_pool.sh [upstream_pool] [port]
# Env vars:
#   NILZ_POOL_UPSTREAM  Private pool URL (default http://127.0.0.1:8000)
#   NILZ_EDGE_HOST      Bind host (default 0.0.0.0)
#   NILZ_EDGE_PORT      Port (default 8010)
#   NILZ_EDGE_ID        Edge identifier
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
export NILZ_POOL_UPSTREAM="${1:-${NILZ_POOL_UPSTREAM:-http://127.0.0.1:8000}}"
export NILZ_EDGE_PORT="${2:-${NILZ_EDGE_PORT:-8010}}"
export NILZ_EDGE_HOST="${NILZ_EDGE_HOST:-0.0.0.0}"; export NILZ_EDGE_ID="${NILZ_EDGE_ID:-edge-1}";
if [[ -d .venv ]]; then source .venv/bin/activate; fi
exec python pool/edge_pool.py 2>&1 | sed "s/^/[edge-pool] /" || true
