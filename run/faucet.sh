#!/usr/bin/env bash
# Run Faucet service
# Usage: ./run/faucet.sh <passphrase> [relay_url] [amount]
# Required:
#   NILZ_FAUCET_PASSPHRASE (or pass first arg)
# Optional envs:
#   NILZ_FAUCET_NODE_URL      Public relay (default http://127.0.0.1:5010)
#   NILZ_FAUCET_WALLET_FILE   Path to wallet (default ./nilz.wallet)
#   NILZ_FAUCET_FROM_LABEL    Key label (default first key)
#   NILZ_FAUCET_AMOUNT        Amount per claim (default 1)
#   NILZ_FAUCET_INTERVAL_S    Cooldown seconds (default 3600)
#   NILZ_FAUCET_DAILY_CAP     Daily total cap (default 100)
#   NILZ_FAUCET_ALLOW_ORIGINS CORS origins (default *)
#   RECAPTCHA_SECRET / HCAPTCHA_SECRET
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
PASSPHRASE="${1:-${NILZ_FAUCET_PASSPHRASE:-}}"
if [[ -z "$PASSPHRASE" ]]; then echo "Passphrase required" >&2; exit 1; fi
export NILZ_FAUCET_PASSPHRASE="$PASSPHRASE"
export NILZ_FAUCET_NODE_URL="${2:-${NILZ_FAUCET_NODE_URL:-http://127.0.0.1:5010}}"
export NILZ_FAUCET_AMOUNT="${3:-${NILZ_FAUCET_AMOUNT:-1}}"
if [[ -d .venv ]]; then source .venv/bin/activate; fi
exec python faucet/server.py 2>&1 | sed "s/^/[faucet] /" || true
