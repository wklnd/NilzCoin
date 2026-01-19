"""NilzCoin Faucet Service

A small FastAPI service to power a web faucet. It:
- Verifies optional CAPTCHA
- Enforces per-IP and per-address cooldowns and daily budget
- Signs a payout from a faucet wallet and broadcasts to a public relay node

Environment variables:
  NILZ_FAUCET_NODE_URL       Public relay URL for broadcasting (default: http://127.0.0.1:5010)
  NILZ_FAUCET_WALLET_FILE    Path to faucet wallet file (default: ./nilz.wallet)
  NILZ_FAUCET_PASSPHRASE     Passphrase for faucet wallet (required)
  NILZ_FAUCET_FROM_LABEL     Label of faucet key (default: first key)
  NILZ_FAUCET_AMOUNT         Amount per claim, e.g. "1.0" (default: 1)
  NILZ_FAUCET_INTERVAL_S     Cooldown seconds per IP/address (default: 3600)
  NILZ_FAUCET_DAILY_CAP      Max total NILZ per day, e.g. "100" (default: 100)
  NILZ_FAUCET_ALLOW_ORIGINS  CORS comma list (default: *)
  RECAPTCHA_SECRET           Google reCAPTCHA secret (optional)
  HCAPTCHA_SECRET            hCaptcha secret (optional)

Run:
  python faucet/server.py
"""

from __future__ import annotations

import os
import json
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
# Ensure repository root is on sys.path when running directly inside faucet/ directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    # When running from repo
    from wallet.wallet import WalletFile, parse_amount
    from types import SimpleNamespace
except Exception:  # pragma: no cover - allow running if package layout differs
    from wallet import WalletFile, parse_amount  # type: ignore
    from types import SimpleNamespace  # type: ignore


log = logging.getLogger("nilz.faucet")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

ROOT_DIR = Path(__file__).resolve().parent
STATE_DIR = ROOT_DIR / "data"
STATE_PATH = STATE_DIR / "state.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class FaucetConfig:
    node_url: str
    wallet_path: Path
    passphrase: str
    from_label: Optional[str]
    amount: Decimal
    interval_s: int
    daily_cap: Decimal
    allow_origins: list[str]
    recaptcha_secret: Optional[str]
    hcaptcha_secret: Optional[str]


def load_config() -> FaucetConfig:
    node_url = os.getenv("NILZ_FAUCET_NODE_URL", "http://127.0.0.1:5010").rstrip("/")
    wallet_path = Path(os.getenv("NILZ_FAUCET_WALLET_FILE", str(Path.cwd() / "nilz.wallet"))).expanduser().resolve()
    passphrase = os.getenv("NILZ_FAUCET_PASSPHRASE", "")
    if not passphrase:
        raise RuntimeError("NILZ_FAUCET_PASSPHRASE must be set")
    from_label = os.getenv("NILZ_FAUCET_FROM_LABEL") or None
    amount = parse_amount(os.getenv("NILZ_FAUCET_AMOUNT", "1"))
    interval_s = int(os.getenv("NILZ_FAUCET_INTERVAL_S", "3600"))
    daily_cap = parse_amount(os.getenv("NILZ_FAUCET_DAILY_CAP", "100"))
    allow_origins = [o.strip() for o in os.getenv("NILZ_FAUCET_ALLOW_ORIGINS", "*").split(",")]
    recaptcha_secret = os.getenv("RECAPTCHA_SECRET")
    hcaptcha_secret = os.getenv("HCAPTCHA_SECRET")
    return FaucetConfig(
        node_url=node_url,
        wallet_path=wallet_path,
        passphrase=passphrase,
        from_label=from_label,
        amount=amount,
        interval_s=interval_s,
        daily_cap=daily_cap,
        allow_origins=allow_origins,
        recaptcha_secret=recaptcha_secret,
        hcaptcha_secret=hcaptcha_secret,
    )


class FaucetState:
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, Any] = {
            "last_by_ip": {},
            "last_by_address": {},
            "daily_totals": {},  # date->str Decimal
        }
        self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                self.data = json.loads(self.path.read_text())
        except Exception:
            log.warning("Failed to load state; starting fresh")

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)

    @staticmethod
    def _today() -> str:
        return datetime.utcnow().strftime("%Y-%m-%d")

    def can_claim(self, ip: str, address: str, interval_s: int) -> tuple[bool, Optional[int]]:
        now = int(time.time())
        last_ip = int(self.data.get("last_by_ip", {}).get(ip, 0))
        last_addr = int(self.data.get("last_by_address", {}).get(address, 0))
        delta = now - max(last_ip, last_addr)
        if delta < interval_s:
            return False, interval_s - delta
        return True, None

    def record_claim(self, ip: str, address: str, amount: Decimal) -> None:
        now = int(time.time())
        self.data.setdefault("last_by_ip", {})[ip] = now
        self.data.setdefault("last_by_address", {})[address] = now
        today = self._today()
        totals = self.data.setdefault("daily_totals", {})
        totals[today] = str(Decimal(totals.get(today, "0")) + amount)
        self._save()

    def today_total(self) -> Decimal:
        return Decimal(self.data.get("daily_totals", {}).get(self._today(), "0"))


class ClaimRequest(BaseModel):
    address: str
    captcha_token: Optional[str] = None


class Faucet:
    def __init__(self, cfg: FaucetConfig, state: FaucetState):
        self.cfg = cfg
        self.state = state
        self.session = requests.Session()
        self.wallet = WalletFile.load(cfg.wallet_path)
        self.key = self.wallet.get_key_metadata(cfg.from_label)
        log.info("Faucet address: %s (label=%s)", self.key.address, self.key.label)

    def _verify_captcha(self, token: Optional[str], remote_ip: Optional[str]) -> None:
        if self.cfg.recaptcha_secret:
            if not token:
                raise HTTPException(status_code=400, detail="Missing CAPTCHA token")
            url = "https://www.google.com/recaptcha/api/siteverify"
            data = {"secret": self.cfg.recaptcha_secret, "response": token}
            if remote_ip:
                data["remoteip"] = remote_ip
            try:
                r = requests.post(url, data=data, timeout=5)
                ok = r.json().get("success", False)
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"reCAPTCHA verify failed: {exc}")
            if not ok:
                raise HTTPException(status_code=400, detail="CAPTCHA verification failed")
        elif self.cfg.hcaptcha_secret:
            if not token:
                raise HTTPException(status_code=400, detail="Missing CAPTCHA token")
            url = "https://hcaptcha.com/siteverify"
            data = {"secret": self.cfg.hcaptcha_secret, "response": token}
            try:
                r = requests.post(url, data=data, timeout=5)
                ok = r.json().get("success", False)
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"hCaptcha verify failed: {exc}")
            if not ok:
                raise HTTPException(status_code=400, detail="CAPTCHA verification failed")

    def _faucet_spendable(self) -> Decimal:
        try:
            r = self.session.get(f"{self.cfg.node_url}/address/{self.key.address}", timeout=10)
            r.raise_for_status()
            spendable = Decimal(r.json().get("spendable", "0"))
            return spendable
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"Failed to query faucet balance: {exc}")

    def _broadcast(self, to_addr: str, amount: Decimal) -> Dict[str, Any]:
        # Build a transaction using the wallet module utilities
        args = SimpleNamespace(from_label=self.cfg.from_label, to=to_addr, amount=amount)
        # craft_transaction returns the dict with signature/hash
        from wallet.wallet import craft_transaction  # local import to avoid circular typing
        tx = craft_transaction(self.wallet, args, self.cfg.passphrase)
        try:
            resp = self.session.post(f"{self.cfg.node_url}/tx", json=tx, timeout=10)
            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return resp.json()
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"Broadcast failed: {exc}")

    def claim(self, address: str, client_ip: str, captcha_token: Optional[str]) -> Dict[str, Any]:
        if not address.startswith("nilz") or len(address) < 10:
            raise HTTPException(status_code=400, detail="Invalid address")
        if address == self.key.address:
            raise HTTPException(status_code=400, detail="Faucet address cannot claim")

        # CAPTCHA if configured
        self._verify_captcha(captcha_token, client_ip)

        # Cooldown checks
        allowed, retry = self.state.can_claim(client_ip, address, self.cfg.interval_s)
        if not allowed:
            raise HTTPException(status_code=429, detail={"retry_after": retry})

        # Daily budget
        today_total = self.state.today_total()
        if today_total + self.cfg.amount > self.cfg.daily_cap:
            raise HTTPException(status_code=429, detail="Faucet daily budget exhausted; try tomorrow")

        # Balance
        spendable = self._faucet_spendable()
        if spendable < self.cfg.amount:
            raise HTTPException(status_code=503, detail="Faucet is empty; please try later")

        # Broadcast
        res = self._broadcast(address, self.cfg.amount)

        # Record claim if broadcast accepted
        self.state.record_claim(client_ip, address, self.cfg.amount)
        return {"accepted": True, "amount": str(self.cfg.amount), "tx_submit": res, "timestamp": now_iso()}


cfg = load_config()
state = FaucetState(STATE_PATH)
faucet = Faucet(cfg, state)

app = FastAPI(title="NilzCoin Faucet", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.allow_origins if cfg.allow_origins != ["*"] else ["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"]
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "address": faucet.key.address,
        "amount": str(cfg.amount),
        "interval_s": cfg.interval_s,
        "daily_cap": str(cfg.daily_cap),
        "today_total": str(state.today_total()),
        "node": cfg.node_url,
    }


@app.post("/claim")
async def claim(request: ClaimRequest, raw: Request):
    client_ip = raw.client.host if raw.client else "unknown"
    return faucet.claim(request.address, client_ip, request.captcha_token)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("NILZ_FAUCET_HOST", "0.0.0.0")
    port = int(os.getenv("NILZ_FAUCET_PORT", "5020"))
    uvicorn.run("faucet.server:app", host=host, port=port, reload=True)
