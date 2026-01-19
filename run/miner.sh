#!/usr/bin/env bash
# Run CPU miner
# Usage: ./run/miner.sh [config_path] [log_level]
# Example:
#   ./run/miner.sh miner/miner_config.json INFO
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
CONFIG="${1:-miner/miner_config.json}"; LEVEL="${2:-INFO}";
if [[ ! -f "$CONFIG" ]]; then echo "Config not found: $CONFIG" >&2; exit 1; fi
if [[ -d .venv ]]; then source .venv/bin/activate; fi
exec python miner/miner.py --config "$CONFIG" --log-level "$LEVEL" 2>&1 | sed "s/^/[miner] /" || true
