import os
import time
import random
import hashlib
from pathlib import Path
from typing import List, Dict

import requests

# Reuse wallet utilities
from wallet.wallet import WalletFile, canonical_json, now_iso  # type: ignore

"""NilzCoin Automated Transaction Bot

Purpose:
    Periodically create, sign and broadcast small NilzCoin transactions to
    the node, cycling between configured recipient addresses. Useful for
    load testing mempool, confirmation flow, and wallet activity.

Configuration (environment variables):
    NILZ_BOT_WALLET_PATH   Path to wallet file (default: ./nilz.wallet)
    NILZ_BOT_WALLET_LABEL  Wallet key label to use (default: "default")
    NILZ_BOT_PASSPHRASE    Passphrase to unlock signing key (required)
    NILZ_NODE_URL          Node base URL (default: http://127.0.0.1:9001)
    NILZ_BOT_RECIPIENTS    Comma-separated wallet addresses to receive funds.
                           If empty, bot will send to its own address (no-op transfers).
    NILZ_BOT_MIN_AMOUNT    Minimum amount per tx (float, default: 0.01)
    NILZ_BOT_MAX_AMOUNT    Maximum amount per tx (float, default: 0.05)
    NILZ_BOT_INTERVAL_SEC  Interval between transactions (default: 30)

Safety:
    - Keep amounts small; do not drain the wallet.
    - Ensure the node runs locally; do not expose passphrase.

Run:
    export NILZ_BOT_PASSPHRASE="your_passphrase"
    python bot/tx_bot.py
"""

class TxBot:
    def __init__(self) -> None:
        self.wallet_path = Path(os.getenv("NILZ_BOT_WALLET_PATH", "./nilz.wallet")).expanduser().resolve()
        self.wallet_label = os.getenv("NILZ_BOT_WALLET_LABEL", "default")
        self.passphrase = os.getenv("NILZ_BOT_PASSPHRASE")
        self.node_url = os.getenv("NILZ_NODE_URL", "http://127.0.0.1:9001").rstrip("/")
        self.recipients: List[str] = [r.strip() for r in os.getenv("NILZ_BOT_RECIPIENTS", "").split(",") if r.strip()]
        self.min_amount = float(os.getenv("NILZ_BOT_MIN_AMOUNT", "0.01"))
        self.max_amount = float(os.getenv("NILZ_BOT_MAX_AMOUNT", "0.05"))
        self.interval = int(os.getenv("NILZ_BOT_INTERVAL_SEC", "30"))
        if not self.passphrase:
            raise RuntimeError("NILZ_BOT_PASSPHRASE env missing")
        try:
            self.wallet = WalletFile.load(self.wallet_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load wallet {self.wallet_path}: {exc}") from exc
        try:
            self.key_meta = self.wallet.get_key_metadata(self.wallet_label)
            self.signing_key = self.wallet.unlock_signing_key(self.wallet_label, self.passphrase)  # type: ignore[arg-type]
        except Exception as exc:
            raise RuntimeError(f"Failed to unlock wallet key '{self.wallet_label}': {exc}") from exc
        if not self.recipients:
            # Fallback to self-transfer list (harmless churn for testing)
            self.recipients = [self.key_meta.address]

    def build_tx(self, to_addr: str, amount: float) -> Dict:
        core = {
            "from": self.key_meta.address,
            "to": to_addr,
            "amount": f"{amount:.8f}",
            "timestamp": now_iso(),
            "nonce": os.urandom(8).hex(),
            "public_key": self.key_meta.public_key,
        }
        payload = canonical_json(core)
        signature = self.signing_key.sign_deterministic(payload, hashfunc=hashlib.sha256).hex()  # type: ignore
        tx = dict(core)
        tx["signature"] = signature
        tx["hash"] = hashlib.sha256(payload + bytes.fromhex(signature)).hexdigest()
        return tx

    def broadcast(self, tx: Dict) -> Dict:
        endpoint = f"{self.node_url}/tx"
        try:
            r = requests.post(endpoint, json=tx, timeout=10)
            return {"status_code": r.status_code, "accepted": r.status_code < 400, "response": r.text[:120]}
        except requests.RequestException as exc:
            return {"status_code": None, "accepted": False, "error": str(exc)}

    def loop(self) -> None:
        print("TxBot starting. Press Ctrl+C to stop.")
        print(f"Node: {self.node_url}")
        print(f"Recipients: {', '.join(self.recipients)}")
        while True:
            to_addr = random.choice(self.recipients)
            amt = random.uniform(self.min_amount, self.max_amount)
            tx = self.build_tx(to_addr, amt)
            result = self.broadcast(tx)
            status = "ACCEPTED" if result.get("accepted") else "REJECTED"
            print(f"[{now_iso()}] {status} tx {tx['hash'][:12]} -> {to_addr[:12]} amt={tx['amount']} code={result.get('status_code')} detail={result.get('response') or result.get('error')}")
            time.sleep(self.interval)


def main():
    bot = TxBot()
    bot.loop()


if __name__ == "__main__":
    main()
