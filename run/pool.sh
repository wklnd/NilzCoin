#!/usr/bin/env bash
# Run NilzCoin mining pool
# Usage: ./run/pool.sh [config_json]
# Env overrides:
#   NILZ_NODE_URL            Private node URL
#   NILZ_POOL_REWARD_ADDRESS Pool coinbase reward address
#   NILZ_POOL_FEE_PERCENT    Operator fee percent
#   NILZ_POOL_CONFIG         Alternate config path
#   NILZ_POOL_RELOAD=1       Enable uvicorn reload
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
CONFIG_ARG="${1:-}"
if [[ -n "$CONFIG_ARG" ]]; then export NILZ_POOL_CONFIG="$CONFIG_ARG"; fi
if [[ -d .venv ]]; then source .venv/bin/activate; fi
exec python pool/server.py 2>&1 | sed "s/^/[pool] /" || true
