# NilzCoin

An end-to-end, approachable cryptocurrency proof‑of‑concept: wallet + node + pool + miner, with optional public relays and a website faucet. It demonstrates a full transaction lifecycle (create → broadcast → mine → confirm) with clear, hackable Python services.

This README covers what each service does, how to configure it, run it locally, and how to host it.

## Quick Start

Requirements:
- Python 3.10+
- Linux/macOS recommended

Setup:
```
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Start the core node (default http://127.0.0.1:5000):
```
python node/server.py
```

Open the wallet GUI (talks to the node):
```
python -m wallet.gui
```

Optional services (pool, public relays, faucet) are documented below.

## Architecture Overview

- Node (`node/server.py`): canonical blockchain state, mempool, PoW validation, dynamic difficulty retargeting, and REST API for wallets and miners.
- Pool (`pool/server.py`): issues work to miners, validates shares, submits full blocks to the node, keeps simple round/earnings accounting, and exposes a dashboard + explorer feed.
- Miner (`miner/miner.py`): CPU miner that requests work from the pool, hashes, and submits shares/blocks; includes stats and temperature readout.
- Public Relay (`node/edge_node.py`): wallet/API gateway that forwards read endpoints and `/tx` to a private node; safe to expose publicly (no mining endpoints).
- Mining Edge (`pool/edge_pool.py`): worker‑only gateway that forwards `/register`, `/work`, `/submit` to a private pool.
- Faucet (`faucet/server.py`): website faucet with rate limiting, optional CAPTCHA, and wallet‑signed payouts broadcast via the public relay.
- Wallet (`wallet/wallet.py`, `wallet/gui.py`): CLI + Tkinter GUI for managing addresses and sending transactions.

Data flow: Wallets broadcast to Node (or a Public Relay) → transactions pile into mempool → miners fetch work from the Pool → when a valid block is found, the Pool submits it to the Node → Node validates and appends the block, clearing confirmed mempool entries.

## Core Concepts

- Proof‑of‑Work: miners find a header hash under the target. Coinbase pays a fixed block reward to the configured address.
- Dynamic difficulty: target adjusts to aim for a desired block time using a moving average window.
- Persistence: Node stores chain and mempool as JSON. Wallets store encrypted keys in a `.wallet` file.

Difficulty/retarget envs (Node):
- `NILZ_BLOCK_TIME_S` (default 30): target seconds per block.
- `NILZ_RETARGET_WINDOW` (default 30): blocks averaged for retarget.

You can examine current network pace with:
```
curl -s http://127.0.0.1:5000/chain | python scripts/time_to_block.py
```

---

## Services

### Node (blockchain)
File: `node/server.py`

What it does:
- Maintains the canonical chain and mempool.
- Validates transactions and blocks, retargets difficulty after each block.
- Issues mining work templates and accepts submitted blocks.

HTTP endpoints:
- `GET /health` – summary stats.
- `GET /chain?limit=N` – recent blocks.
- `GET /mempool` – pending transactions.
- `GET /address/{address}` – balances and tx history.
- `POST /tx` – submit a signed transaction.
- `GET /work?address=…` – mining block template for a reward address.
- `POST /block` – submit a mined block.

Run locally:
```
python node/server.py
```

Env vars:
- `NILZ_BLOCK_TIME_S` – desired block time (seconds).
- `NILZ_RETARGET_WINDOW` – moving average window (blocks).

Hosting tips:
- Keep the node private if you run a public relay. Bind to `127.0.0.1` and let the relay sit in front.
- Persist data under `node/data/` (JSON). Back up for long‑running testnets.

### Public Relay (wallet/API gateway)
File: `node/edge_node.py`

What it does:
- Publicly exposes safe wallet endpoints and forwards them to your private node.
- Allows browser wallets and websites (CORS enabled) without exposing mining APIs.

Run:
```
NILZ_NODE_UPSTREAM=http://127.0.0.1:5000 \
python node/edge_node.py
```

Env vars:
- `NILZ_NODE_UPSTREAM` – base URL of the private node.
- `NILZ_PUBLIC_HOST` (default `0.0.0.0`), `NILZ_PUBLIC_PORT` (default `5010`).
- `NILZ_PUBLIC_ID` – optional instance label.

Hosting tips:
- Put behind HTTPS (e.g., Caddy/Nginx). Optionally add basic auth or IP rate limiting.

### Pool (mining coordinator)
File: `pool/server.py`

What it does:
- Issues jobs to miners from a private node, checks shares, and submits blocks.
- Tracks per‑miner stats and rounds; provides basic earnings/payout previews.
- Serves a simple dashboard (`/`) and an explorer JSON feed (`/blockchain.json`).

Config file: `pool/pool_config.json` (see `pool/pool_config.example.json`). Key options:
- `node_url`: private node base URL.
- `host`, `port`: bind address.
- `job_ttl_seconds`, `job_refresh_seconds`: job expiry and polling interval.
- `share_target_multiplier`: ease target for shares vs full blocks.
- `reward_address`: pool coinbase address controlled by the operator.
- `pool_fee_percent`: operator fee retained from each block reward.
- `operator_wallet_path`, `operator_wallet_label`: wallet/key used for settlements.
- `payout_min_amount`, `payout_interval_seconds`: settlement thresholds/schedule (informational + API to trigger).

Run:
```
python pool/server.py
```

Helpful endpoints:
- `GET /` – dashboard HTML.
- `GET /stats` – miners, shares, blocks, history.
- `GET /earnings` – pooled earnings summary.
- `GET /payouts` – unsettled allocations.
- `POST /settle` – build and broadcast payouts from the operator wallet.
- `GET /next_payout` – countdown helper for UI.

Hosting tips:
- Keep the pool private; expose a Mining Edge to the public Internet.
- Persist `pool_rounds.json`, `pool_settlements.json`, and `pool_miners.json`.

### Mining Edge (worker‑only gateway)
File: `pool/edge_pool.py`

What it does:
- Public gateway for miners, forwarding only `/register`, `/work`, `/submit` to the private pool.

Run:
```
NILZ_POOL_UPSTREAM=http://127.0.0.1:8000 \
python pool/edge_pool.py
```

Env vars:
- `NILZ_POOL_UPSTREAM` – base URL of the private pool.
- `NILZ_EDGE_HOST` (default `0.0.0.0`), `NILZ_EDGE_PORT` (default `8010`).
- `NILZ_EDGE_ID` – optional label in `/health`.

Hosting tips:
- Put behind HTTPS and rate‑limit POSTs. Consider an IP allowlist for your miners.

### Miner (CPU)
File: `miner/miner.py`

What it does:
- Registers with a pool, fetches work, hashes on CPU, submits shares/blocks.
- Logs interval and moving‑average hashrate; attempts to read CPU temperature (via `psutil`).

Config file: `miner/miner_config.json`
```
{
	"wallet_address": "nilzYourMinerAddress...",
	"pool_address": "http://127.0.0.1:8010",
	"worker_name": "rig-01",
	"threads": 2,
	"stats_interval": 30
}
```

Run:
```
python miner/miner.py --config miner/miner_config.json
```

Flags:
- `--log-level DEBUG|INFO|WARNING|...` or `-q/--quiet` for fewer logs.

### Wallet (CLI + GUI)
Files: `wallet/wallet.py`, `wallet/gui.py`

CLI usage:
```
python -m wallet.wallet init --label default --wallet ./nilz.wallet
python -m wallet.wallet list --wallet ./nilz.wallet
python -m wallet.wallet new-address --label MINER --wallet ./nilz.wallet
python -m wallet.wallet send --from-label default --to nilz... --amount 1.0 \
	--wallet ./nilz.wallet --node-url http://127.0.0.1:5010
```

GUI:
```
python -m wallet.gui
```

### Faucet (website payouts)
File: `faucet/server.py` (details also in `faucet/README.md`)

What it does:
- Web API to send small payouts from a hot wallet, with per‑IP/address cooldown, daily cap, and optional CAPTCHA.
- Broadcasts via the Public Relay so your node can stay private.

Run:
```
export NILZ_FAUCET_NODE_URL=http://127.0.0.1:5010
export NILZ_FAUCET_WALLET_FILE=./nilz.wallet
export NILZ_FAUCET_PASSPHRASE=changeme
python faucet/server.py
```

Key env vars:
- `NILZ_FAUCET_NODE_URL` – public relay base URL.
- `NILZ_FAUCET_WALLET_FILE` – faucet wallet path.
- `NILZ_FAUCET_PASSPHRASE` – passphrase to unlock the faucet key.
- `NILZ_FAUCET_FROM_LABEL` – specific wallet label (optional; defaults to first key).
- `NILZ_FAUCET_AMOUNT` – amount per claim (e.g., `1`).
- `NILZ_FAUCET_INTERVAL_S` – cooldown seconds per IP/address.
- `NILZ_FAUCET_DAILY_CAP` – max NILZ per day.
- `NILZ_FAUCET_ALLOW_ORIGINS` – CORS comma list (default `*`).
- `RECAPTCHA_SECRET` or `HCAPTCHA_SECRET` – enable CAPTCHA verification (optional).

Endpoints:
- `GET /health` – current faucet status and budget.
- `POST /claim` – request a payout `{ address, captcha_token? }`.

Hosting tips:
- Treat the faucet wallet as a hot wallet with limited balance. Secure with CAPTCHA and a reverse proxy with rate limits.

---

## Hosting & Deployment

Recommended topology for Internet exposure:
- Keep `node/server.py` and `pool/server.py` private (bind to localhost or a private network).
- Expose `node/edge_node.py` (Public Relay) for wallets and your website.
- Expose `pool/edge_pool.py` (Mining Edge) for miners.
- Optionally expose `faucet/server.py` to power your website faucet.

Reverse proxy quickstart (Caddy):
```
example.com {
	reverse_proxy 127.0.0.1:5010
}

miners.example.com {
	reverse_proxy 127.0.0.1:8010
}
```

Operational tips:
- Enable HTTPS and basic DDoS protections at the edge (rate limits, request size caps).
- Monitor logs for `Block rejected … Index mismatch` which typically indicates a stale job; ensure miners request fresh work frequently and the pool enforces job TTL.
- Back up `node/data/` and pool JSON state files for continuity.

## Troubleshooting

- Index mismatch on block submit: a newer block landed; miners must refresh work. Ensure the pool’s `job_ttl_seconds` and `job_refresh_seconds` are small (defaults are sane).
- Slow confirmations: reduce `NILZ_BLOCK_TIME_S` or increase hashrate; consider a smaller `NILZ_RETARGET_WINDOW` while bootstrapping.
- Wallet broadcast fails: point GUI/CLI to the Public Relay (`http://127.0.0.1:5010`) or directly to the node if on a trusted LAN.

## Development

Run all services with reload in separate terminals during dev. Typical ports:
- Node: `5000`
- Public Relay: `5010`
- Pool: `8000`
- Mining Edge: `8010`
- Faucet: `5020`

Install/update dependencies:
```
pip install -r requirements.txt
```

# Final Words
Please remember this is a proof‑of‑concept for educational purposes. It is not secure or production‑ready, not even close.