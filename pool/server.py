"""NilzCoin mining pool service."""

from __future__ import annotations

import logging
import secrets
import threading
from collections import deque
from dataclasses import dataclass, field
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Deque, Dict, List, Optional, Any

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


log = logging.getLogger("nilz.pool")
#logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")


@dataclass
class PoolConfig:
    node_url: str = "http://127.0.0.1:5000"
    host: str = "0.0.0.0"
    port: int = 8000
    job_ttl_seconds: int = 90
    job_refresh_seconds: int = 5
    share_target_multiplier: int = 16  # shares may be easier than full target
    reward_address: str = "nilzPoolOperatorXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    pool_fee_percent: float = 2.0  # percent of block reward retained by operator
    payout_min_amount: float = 0.1  # minimum owed before showing in payouts
    payout_interval_seconds: int = 3600  # informational (not enforced yet)
    operator_wallet_path: str = "nilz.wallet"  # path to wallet file containing operator key
    operator_wallet_label: str = "POOL"  # label of operator key within wallet
    operator_passphrase_env: str = "NILZ_POOL_OPERATOR_PASSPHRASE"  # env var holding passphrase

    @classmethod
    def load(cls, path: Optional[Path]) -> "PoolConfig":
        if path and path.exists():
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        # Allow env overrides for quick Docker usage
        env_node = os.getenv("NILZ_NODE_URL")
        env_reward = os.getenv("NILZ_POOL_REWARD_ADDRESS")
        env_fee = os.getenv("NILZ_POOL_FEE_PERCENT")
        cfg = cls()
        if env_node:
            cfg.node_url = env_node
        if env_reward:
            cfg.reward_address = env_reward
        if env_fee:
            try:
                cfg.pool_fee_percent = float(env_fee)
            except ValueError:
                pass
        return cfg


class NodeClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def fetch_work(self, reward_address: str) -> Dict:
        try:
            response = requests.get(
                f"{self.base_url}/work",
                params={"address": reward_address},
                timeout=5,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise HTTPException(status_code=503, detail=f"Node unavailable: {exc}") from exc

    def submit_block(self, payload: Dict) -> Dict:
        try:
            response = requests.post(f"{self.base_url}/block", json=payload, timeout=5)
            if response.status_code >= 400:
                # Try to extract structured detail; fall back to raw text
                try:
                    err_json = response.json()
                    detail = err_json.get("detail") or err_json
                except Exception:
                    detail = response.text.strip()
                return {
                    "accepted": False,
                    "status": response.status_code,
                    "detail": detail,
                }
            try:
                data = response.json()
            except Exception:
                data = {"raw": response.text.strip()}
            # Promote block height/index for convenience if present
            height = None
            if isinstance(data, dict):
                height = data.get("index") or data.get("height")
            return {"accepted": True, "status": response.status_code, "height": height, "data": data}
        except requests.RequestException as exc:
            raise HTTPException(status_code=503, detail=f"Failed to submit block: {exc}") from exc

    def fetch_chain(self, limit: Optional[int] = None) -> List[Dict]:
        try:
            params = {"limit": limit} if limit else None
            response = requests.get(f"{self.base_url}/chain", params=params, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise HTTPException(status_code=503, detail=f"Node unavailable: {exc}") from exc


@dataclass
class Miner:
    miner_id: str
    name: str
    wallet_address: str
    registered_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Job:
    job_id: str
    template: Dict
    miner_id: str
    issued_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def target(self) -> int:
        return int(self.template["target"], 16)


@dataclass
class ShareStats:
    shares: int = 0
    blocks: int = 0
    last_share_at: Optional[datetime] = None
    events: Deque[datetime] = field(default_factory=deque)


class PoolState:
    HISTORY_RETENTION_MINUTES = 360
    ROUNDS_FILE = Path("pool_rounds.json")
    SETTLEMENT_FILE = Path("pool_settlements.json")
    MINERS_FILE = Path("pool_miners.json")

    def __init__(self, config: PoolConfig):
        self.config = config
        self.node = NodeClient(config.node_url)
        self.miners: Dict[str, Miner] = {}
        self.jobs: Dict[str, Job] = {}
        self.stats: Dict[str, ShareStats] = {}
        # Round accounting: list of dicts capturing block reward distribution snapshots
        self.rounds: List[Dict[str, Any]] = []
        self._last_snapshot: Dict[str, int] = {}  # miner_id -> cumulative shares at last round boundary
        self.lock = threading.Lock()
        self.settlements: List[Dict[str, Any]] = []
        self._load_miners()
        self._load_persisted_rounds()
        self._load_settlements()
        # Auto settlement scheduling
        self._auto_thread: Optional[threading.Thread] = None
        self._auto_stop = threading.Event()
        self._last_auto_settlement_at: Optional[datetime] = None

    # ---------------- Auto Settlement -----------------

    def start_auto_settler(self) -> None:
        """Start background thread that settles payouts automatically based on config interval.

        Requirements for triggering a live settlement:
        - Interval elapsed (config.payout_interval_seconds)
        - There are miners with owed amounts (compute_payouts miners list not empty)
        - There exist unsettled rounds
        - Operator passphrase env variable present (needed for signing)
        """
        if self._auto_thread and self._auto_thread.is_alive():
            return
        interval = max(10, int(self.config.payout_interval_seconds))  # guard minimum
        if interval <= 0:
            log.info("Auto settler disabled (interval <= 0)")
            return

        def loop():
            log.info("Auto settler thread started interval=%ss", interval)
            while not self._auto_stop.is_set():
                try:
                    now = datetime.utcnow()
                    # Only proceed if interval elapsed
                    if (
                        self._last_auto_settlement_at is None
                        or (now - self._last_auto_settlement_at).total_seconds() >= interval
                    ):
                        payouts = self.compute_payouts()
                        miners = payouts.get("miners", [])
                        has_unsettled = any(not r.get("settled") for r in self.rounds)
                        passphrase_present = bool(os.getenv(self.config.operator_passphrase_env))
                        if miners and has_unsettled and passphrase_present:
                            log.info("Auto settlement criteria met (miners=%d rounds=%d)", len(miners), len(self.rounds))
                            result = self.settle(dry_run=False)
                            if result.get("settled"):
                                log.info("Auto settlement completed txs=%d", len(result.get("transactions", [])))
                                self._last_auto_settlement_at = now
                            else:
                                log.warning("Auto settlement failed: %s", result.get("reason"))
                        else:
                            # Provide debug context occasionally
                            log.debug(
                                "Auto settlement skipped miners=%d unsettled=%s passphrase=%s", 
                                len(miners), has_unsettled, passphrase_present,
                            )
                    # Sleep in short increments to allow responsive shutdown
                except Exception as exc:
                    log.warning("Auto settler loop error: %s", exc)
                finally:
                    self._auto_stop.wait(5)
            log.info("Auto settler thread stopped")

        self._auto_thread = threading.Thread(target=loop, name="auto-settler", daemon=True)
        self._auto_thread.start()

    def stop_auto_settler(self) -> None:
        if self._auto_thread and self._auto_thread.is_alive():
            self._auto_stop.set()
            self._auto_thread.join(timeout=5)

    # Persistence ---------------------------------------------------------

    def _load_persisted_rounds(self) -> None:
        if self.ROUNDS_FILE.exists():
            try:
                with self.ROUNDS_FILE.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                    if isinstance(data, list):
                        self.rounds = data
                log.info("Loaded %d persisted rounds", len(self.rounds))
            except Exception as exc:
                log.warning("Failed to load rounds file: %s", exc)

    def _persist_rounds(self) -> None:
        try:
            with self.ROUNDS_FILE.open("w", encoding="utf-8") as handle:
                json.dump(self.rounds, handle, indent=2)
        except Exception as exc:
            log.error("Failed to persist rounds: %s", exc)

    def _load_settlements(self) -> None:
        if self.SETTLEMENT_FILE.exists():
            try:
                with self.SETTLEMENT_FILE.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                    if isinstance(data, list):
                        self.settlements = data
                log.info("Loaded %d settlements", len(self.settlements))
            except Exception as exc:
                log.warning("Failed to load settlements file: %s", exc)

    def _persist_settlements(self) -> None:
        try:
            with self.SETTLEMENT_FILE.open("w", encoding="utf-8") as handle:
                json.dump(self.settlements, handle, indent=2)
        except Exception as exc:
            log.error("Failed to persist settlements: %s", exc)

    def _load_miners(self) -> None:
        if self.MINERS_FILE.exists():
            try:
                with self.MINERS_FILE.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                    if isinstance(data, list):
                        for entry in data:
                            try:
                                miner = Miner(
                                    miner_id=entry["miner_id"],
                                    name=entry.get("name", "miner"),
                                    wallet_address=entry.get("wallet_address", ""),
                                    registered_at=datetime.fromisoformat(entry.get("registered_at")) if entry.get("registered_at") else datetime.utcnow(),
                                )
                                self.miners[miner.miner_id] = miner
                            except Exception:
                                continue
                        if self.miners:
                            log.info("Loaded %d persisted miners", len(self.miners))
            except Exception as exc:
                log.warning("Failed to load miners file: %s", exc)

    def _persist_miners(self) -> None:
        try:
            payload = []
            for miner in self.miners.values():
                payload.append({
                    "miner_id": miner.miner_id,
                    "name": miner.name,
                    "wallet_address": miner.wallet_address,
                    "registered_at": miner.registered_at.isoformat(),
                })
            with self.MINERS_FILE.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception as exc:
            log.error("Failed to persist miners: %s", exc)

    def register_miner(self, name: str, wallet_address: str) -> Miner:
        miner_id = secrets.token_hex(8)
        miner = Miner(miner_id=miner_id, name=name, wallet_address=wallet_address)
        with self.lock:
            self.miners[miner_id] = miner
            self._persist_miners()
        log.info("Registered miner %s (%s)", miner_id, name)
        return miner

    def _validate_miner(self, miner_id: str) -> Miner:
        miner = self.miners.get(miner_id)
        if not miner:
            raise HTTPException(status_code=404, detail="Unknown miner")
        return miner

    def issue_job(self, miner_id: str) -> Dict:
        miner = self._validate_miner(miner_id)
        # Always use pool reward_address for coinbase so pool can later distribute payouts.
        template = self.node.fetch_work(reward_address=self.config.reward_address)
        job_id = secrets.token_hex(6)
        job = Job(job_id=job_id, template=template, miner_id=miner_id)
        with self.lock:
            self.jobs[job_id] = job
            self._purge_expired_jobs()
        response = dict(template)
        response["job_id"] = job_id
        response["miner_address"] = miner.wallet_address
        response["share_target"] = self._share_target(job.target)
        return response

    def _share_target(self, block_target: int) -> str:
        adjusted = min(block_target * self.config.share_target_multiplier, (1 << 256) - 1)
        return f"{adjusted:064x}"

    def _purge_expired_jobs(self) -> None:
        cutoff = datetime.utcnow() - timedelta(seconds=self.config.job_ttl_seconds)
        expired = [job_id for job_id, job in self.jobs.items() if job.issued_at < cutoff]
        for job_id in expired:
            self.jobs.pop(job_id, None)

    def get_job(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found or expired")
        return job

    def submit_share(self, miner_id: str, job_id: str, nonce: int, block_hash: str, header: Dict) -> Dict:
        miner = self._validate_miner(miner_id)
        job = self.get_job(job_id)
        if job.miner_id != miner_id:
            raise HTTPException(status_code=400, detail="Job does not belong to miner")

        hash_int = int(block_hash, 16)
        share_target = int(self._share_target(job.target), 16)
        is_block = hash_int <= job.target
        is_share = hash_int <= share_target

        if not is_share:
            raise HTTPException(status_code=400, detail="Hash does not meet share target")

        log.info(
            "Accepted share from %s job=%s hash=%s block=%s",
            miner_id,
            job_id,
            block_hash[:16],
            is_block,
        )

        result: Dict[str, object] = {"share_accepted": True, "block_submitted": False}
        event_time = datetime.utcnow()
        self._record_share(miner_id, is_block, event_time)

        if is_block:
            payload = {
                "miner_id": miner_id,
                "job_id": job_id,
                "nonce": nonce,
                "header": header,
                "template": job.template,
                "hash": block_hash,
            }
            node_response = self.node.submit_block(payload)
            if node_response.get("accepted"):
                result["block_submitted"] = True
                result["node_response"] = node_response
                self._record_round(block_hash, node_response.get("height"), job)
            else:
                result["block_submitted"] = False
                result["block_rejection"] = node_response
                log.warning(
                    "Block rejected job=%s miner=%s status=%s detail=%s",
                    job_id,
                    miner_id,
                    node_response.get("status"),
                    node_response.get("detail"),
                )
        return result
    def _record_round(self, block_hash: str, height: Optional[int], job: Job) -> None:
        # Compute delta shares since last round for each miner
        with self.lock:
            snapshot: Dict[str, int] = {m_id: (self.stats.get(m_id) or ShareStats()).shares for m_id in self.miners.keys()}
            delta: Dict[str, int] = {}
            total_delta = 0
            for m_id, shares in snapshot.items():
                prev = self._last_snapshot.get(m_id, 0)
                d = max(shares - prev, 0)
                if d > 0:
                    delta[m_id] = d
                    total_delta += d
            # Coinbase reward amount from template
            try:
                reward_amount = float(job.template["transactions"][0]["amount"])
            except Exception:
                reward_amount = 0.0
            if height is None:
                # Derive expected height from template previous index if available
                prev_index = job.template.get("index")
                if isinstance(prev_index, int):
                    height = prev_index + 1
            # Pre-compute operator fee amount for later earnings aggregation
            fee_percent = self.config.pool_fee_percent
            operator_fee_amount = reward_amount * (fee_percent / 100.0) if reward_amount > 0 else 0.0
            round_entry = {
                "height": height,
                "block_hash": block_hash,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "reward_amount": reward_amount,
                "shares_delta": delta,
                "total_delta": total_delta,
                "fee_percent": fee_percent,
                "operator_fee_amount": operator_fee_amount,
                "settled": False,
            }
            self.rounds.append(round_entry)
            # Update last snapshot
            self._last_snapshot = snapshot
            self._persist_rounds()

    def compute_payouts(self) -> Dict[str, Any]:
        # Aggregate unsettled rounds and compute owed amounts per miner.
        with self.lock:
            owed: Dict[str, float] = {}
            rounds_detail: List[Dict[str, Any]] = []
            for rnd in self.rounds:
                if rnd.get("settled"):
                    continue
                reward = rnd.get("reward_amount", 0.0)
                fee_percent = rnd.get("fee_percent", 0.0)
                operator_fee = reward * (fee_percent / 100.0)
                distributable = max(reward - operator_fee, 0.0)
                total_delta = rnd.get("total_delta", 0) or 0
                miner_allocations: List[Dict[str, Any]] = []
                if total_delta > 0 and distributable > 0:
                    for m_id, shares in rnd.get("shares_delta", {}).items():
                        portion = shares / total_delta
                        amount = distributable * portion
                        if amount >= self.config.payout_min_amount:
                            owed[m_id] = owed.get(m_id, 0.0) + amount
                        miner_allocations.append({
                            "miner_id": m_id,
                            "shares": shares,
                            "portion": round(portion, 8),
                            "amount": round(amount, 8),
                        })
                rounds_detail.append({
                    "height": rnd.get("height"),
                    "block_hash": rnd.get("block_hash"),
                    "reward": reward,
                    "operator_fee": round(operator_fee, 8),
                    "distributable": round(distributable, 8),
                    "total_shares_delta": total_delta,
                    "allocations": miner_allocations,
                })
            # Do not mark settled automatically; operator would settle externally.
            miner_details = []
            for m_id, amount in owed.items():
                miner = self.miners.get(m_id)
                miner_details.append({
                    "miner_id": m_id,
                    "wallet_address": miner.wallet_address if miner else None,
                    "owed_amount": round(amount, 8),
                })
            return {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "rounds": rounds_detail,
                "miners": miner_details,
                "fee_percent": self.config.pool_fee_percent,
                "reward_address": self.config.reward_address,
            }

    # Payout schedule (informational) -----------------------------------

    def next_payout_info(self) -> Dict[str, Any]:
        """Return the next expected payout timestamp and remaining seconds.

        Uses the configured interval (payout_interval_seconds). The anchor time
        is the most recent settlement timestamp if any exist; otherwise the
        timestamp of the latest round; falling back to current time.
        This does not trigger settlement automatically; it is purely
        informational for UI countdown purposes.
        """
        interval = max(60, int(self.config.payout_interval_seconds))  # enforce minimum 1 min
        with self.lock:
            anchor_iso: Optional[str] = None
            if self.settlements:
                anchor_iso = self.settlements[-1].get("timestamp")
            elif self.rounds:
                anchor_iso = self.rounds[-1].get("timestamp")
        now = datetime.utcnow()
        if anchor_iso and isinstance(anchor_iso, str):
            # Remove trailing Z for fromisoformat compatibility if present
            ts_clean = anchor_iso.rstrip("Z")
            try:
                anchor_dt = datetime.fromisoformat(ts_clean)
            except ValueError:
                anchor_dt = now
        else:
            anchor_dt = now
        next_dt = anchor_dt + timedelta(seconds=interval)
        remaining = max(int((next_dt - now).total_seconds()), 0)
        return {
            "now": now.isoformat() + "Z",
            "anchor_timestamp": anchor_dt.isoformat() + "Z",
            "next_payout_timestamp": next_dt.isoformat() + "Z",
            "seconds_remaining": remaining,
            "interval_seconds": interval,
            "overdue": remaining == 0 and now >= next_dt,
        }

    # Settlement ----------------------------------------------------------

    def settle(self, dry_run: bool = False, passphrase_override: Optional[str] = None) -> Dict[str, Any]:
        """Create and optionally broadcast payout transactions to the node.

        Strategy:
        - Aggregate owed amounts for all unsettled rounds (same logic as compute_payouts)
        - Build one transaction per miner meeting min amount
        - Sign using operator wallet key identified by label
        - Broadcast to node /tx endpoint
        - If all broadcasts succeed and not dry_run, mark rounds settled & persist
        """
        payouts = self.compute_payouts()
        miners = payouts["miners"]
        if not miners:
            return {"settled": False, "reason": "No miners with owed amounts", "dry_run": dry_run}

        # Load operator wallet lazily only if we need to sign
        from wallet.wallet import WalletFile, canonical_json, now_iso  # type: ignore
        import hashlib
        from ecdsa import SECP256k1, SigningKey

        passphrase = passphrase_override or os.getenv(self.config.operator_passphrase_env)
        if not passphrase and not dry_run:
            return {"settled": False, "reason": f"Operator passphrase env {self.config.operator_passphrase_env} missing", "dry_run": dry_run}

        wallet_path = Path(self.config.operator_wallet_path).expanduser().resolve()
        try:
            wallet = WalletFile.load(wallet_path)
        except Exception as exc:
            return {"settled": False, "reason": f"Failed to load operator wallet: {exc}", "dry_run": dry_run}

        try:
            key_meta = wallet.get_key_metadata(self.config.operator_wallet_label)
            if not dry_run:
                signing_key = wallet.unlock_signing_key(self.config.operator_wallet_label, passphrase)  # type: ignore[arg-type]
            operator_address = key_meta.address
        except Exception as exc:
            return {"settled": False, "reason": f"Operator key unavailable: {exc}", "dry_run": dry_run}

        txs: List[Dict[str, Any]] = []
        for miner in miners:
            miner_id = miner.get("miner_id")
            to_addr = miner.get("wallet_address")
            amount = miner.get("owed_amount")
            if not to_addr or amount is None:
                continue
            # Build unsigned core
            tx_core = {
                "from": operator_address,
                "to": to_addr,
                "amount": str(amount),
                "timestamp": now_iso(),
                "nonce": secrets.token_hex(8),
                "public_key": key_meta.public_key,
            }
            payload = canonical_json(tx_core)
            if not dry_run:
                signature = signing_key.sign_deterministic(payload, hashfunc=hashlib.sha256).hex()  # type: ignore[name-defined]
                tx = dict(tx_core)
                tx["signature"] = signature
                tx["hash"] = hashlib.sha256(payload + bytes.fromhex(signature)).hexdigest()
            else:
                # Simulate hash for preview
                fake_hash = hashlib.sha256(payload + b"dryrun").hexdigest()
                tx = dict(tx_core)
                tx["signature"] = None
                tx["hash"] = fake_hash
            txs.append(tx)

        # Broadcast if not dry_run
        results: List[Dict[str, Any]] = []
        if not dry_run:
            endpoint = self.config.node_url.rstrip("/") + "/tx"
            for tx in txs:
                try:
                    response = requests.post(endpoint, json=tx, timeout=10)
                    ok = response.status_code < 400
                    results.append({"hash": tx["hash"], "status_code": response.status_code, "accepted": ok})
                    if not ok:
                        log.warning("Settlement tx rejected %s %s", response.status_code, response.text[:120])
                except requests.RequestException as exc:
                    results.append({"hash": tx["hash"], "accepted": False, "error": str(exc)})
            # If any failed, abort marking rounds settled
            if not results or any(not r.get("accepted") for r in results):
                return {"settled": False, "transactions": results, "dry_run": dry_run, "reason": "One or more transactions failed"}
            # Mark rounds settled
            with self.lock:
                for rnd in self.rounds:
                    if not rnd.get("settled"):
                        rnd["settled"] = True
                self._persist_rounds()
                # Record settlement entries
                ts = datetime.utcnow().isoformat() + "Z"
                for tx in txs:
                    self.settlements.append({
                        "timestamp": ts,
                        "tx_hash": tx["hash"],
                        "from": tx["from"],
                        "to": tx["to"],
                        "amount": tx["amount"],
                    })
                self._persist_settlements()
            return {"settled": True, "transactions": results, "dry_run": dry_run}
        else:
            return {"settled": False, "dry_run": True, "transactions": txs}

    def _record_share(self, miner_id: str, is_block: bool, event_time: datetime) -> None:
        with self.lock:
            stats = self.stats.setdefault(miner_id, ShareStats())
            stats.shares += 1
            stats.last_share_at = event_time
            if is_block:
                stats.blocks += 1
            stats.events.append(event_time)
            cutoff = event_time - timedelta(minutes=self.HISTORY_RETENTION_MINUTES)
            while stats.events and stats.events[0] < cutoff:
                stats.events.popleft()

    def stats_snapshot(self, wallet_address: Optional[str] = None, history_minutes: int = 60) -> Dict:
        history_minutes = max(1, history_minutes)
        wallet_filter = wallet_address.lower() if wallet_address else None
        with self.lock:
            now = datetime.utcnow()
            miners_payload: List[Dict[str, object]] = []
            totals = {"miners": 0, "shares": 0, "blocks": 0}
            for miner_id, miner in self.miners.items():
                if wallet_filter and miner.wallet_address.lower() != wallet_filter:
                    continue
                stats = self.stats.get(miner_id) or ShareStats()
                uptime_seconds = max((now - miner.registered_at).total_seconds(), 0.0)
                share_rate = self._share_rate_per_min(stats, now)
                history = self._history_points(stats, now, history_minutes)
                last_share_at = (
                    stats.last_share_at.isoformat() + "Z" if stats.last_share_at else None
                )
                miner_payload = {
                    "miner_id": miner_id,
                    "name": miner.name,
                    "wallet_address": miner.wallet_address,
                    "shares": stats.shares,
                    "blocks": stats.blocks,
                    "last_share_at": last_share_at,
                    "uptime_seconds": int(uptime_seconds),
                    "share_rate_per_min": share_rate,
                    "history": history,
                }
                miners_payload.append(miner_payload)
                totals["miners"] += 1
                totals["shares"] += stats.shares
                totals["blocks"] += stats.blocks
            return {
                "generated_at": now.isoformat() + "Z",
                "totals": totals,
                "miners": miners_payload,
            }

    @staticmethod
    def _share_rate_per_min(stats: ShareStats, now: datetime, window_minutes: int = 10) -> float:
        if not stats.events:
            return 0.0
        window = timedelta(minutes=max(1, window_minutes))
        cutoff = now - window
        recent = [event for event in stats.events if event >= cutoff]
        if not recent:
            return 0.0
        elapsed_minutes = max(window.total_seconds() / 60.0, 1.0)
        return len(recent) / elapsed_minutes

    @staticmethod
    def _history_points(stats: ShareStats, now: datetime, minutes: int) -> List[Dict[str, object]]:
        if minutes <= 0:
            return []
        start = (now - timedelta(minutes=minutes - 1)).replace(second=0, microsecond=0)
        buckets: Dict[datetime, int] = {}
        for event in list(stats.events):
            if event < start - timedelta(minutes=1):
                continue
            bucket = event.replace(second=0, microsecond=0)
            buckets[bucket] = buckets.get(bucket, 0) + 1
        points: List[Dict[str, object]] = []
        for i in range(minutes):
            bucket_start = start + timedelta(minutes=i)
            share_count = buckets.get(bucket_start, 0)
            points.append(
                {
                    "timestamp": (bucket_start + timedelta(minutes=1)).isoformat() + "Z",
                    "shares": share_count,
                }
            )
        return points


class RegisterRequest(BaseModel):
    miner_name: str = Field(..., min_length=1, max_length=40)
    wallet_address: str = Field(..., min_length=8, max_length=80)


class RegisterResponse(BaseModel):
    miner_id: str
    poll_interval: int


class SubmitRequest(BaseModel):
    miner_id: str
    job_id: str
    nonce: int
    block_hash: str = Field(..., min_length=64, max_length=64)
    header: Dict


ROOT_DIR = Path(__file__).resolve().parent
# Load configuration from file if present (env NILZ_POOL_CONFIG or default pool_config.json)
config_path_env = os.getenv("NILZ_POOL_CONFIG")
default_cfg = ROOT_DIR / "pool_config.json"
cfg_path = Path(config_path_env).expanduser() if config_path_env else default_cfg
config = PoolConfig.load(cfg_path if cfg_path.exists() else None)
state = PoolState(config)
app = FastAPI(title="NilzCoin Pool", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _startup_event() -> None:
    try:
        state.start_auto_settler()
        log.info("Startup: auto settler initiated")
    except Exception as exc:
        log.warning("Startup: failed to start auto settler: %s", exc)

@app.on_event("shutdown")
def _shutdown_event() -> None:
    try:
        state.stop_auto_settler()
        log.info("Shutdown: auto settler stopped")
    except Exception as exc:
        log.warning("Shutdown: failed to stop auto settler: %s", exc)


@app.get("/", response_class=HTMLResponse)
def dashboard():
    html_path = ROOT_DIR / "pool.html"
    try:
        return html_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="pool.html missing") from exc


@app.get("/explorer", response_class=HTMLResponse)
def explorer_page():
    html_path = ROOT_DIR / "blockchainexplorer.html"
    try:
        return html_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="blockchainexplorer.html missing") from exc


@app.post("/register", response_model=RegisterResponse)
def register(request: RegisterRequest):
    miner = state.register_miner(request.miner_name, request.wallet_address)
    return RegisterResponse(miner_id=miner.miner_id, poll_interval=config.job_refresh_seconds)


@app.get("/work")
def work(miner_id: str):
    return state.issue_job(miner_id)


@app.post("/submit")
def submit(request: SubmitRequest):
    result = state.submit_share(
        miner_id=request.miner_id,
        job_id=request.job_id,
        nonce=request.nonce,
        block_hash=request.block_hash,
        header=request.header,
    )
    return result


@app.get("/health")
def health():
    return {"status": "ok", "miners": len(state.miners)}


@app.get("/stats")
def stats(wallet: Optional[str] = None, history_minutes: int = 60):
    return state.stats_snapshot(wallet_address=wallet, history_minutes=history_minutes)


@app.get("/payouts")
def payouts():
    return state.compute_payouts()


class SettleRequest(BaseModel):
    dry_run: bool = False
    passphrase: Optional[str] = None


@app.post("/settle")
def settle(request: SettleRequest):
    return state.settle(dry_run=request.dry_run, passphrase_override=request.passphrase)


@app.get("/next_payout")
def next_payout():
    return state.next_payout_info()


@app.get("/blockchain.json")
def blockchain_json(limit: Optional[int] = None):
    """Serve a normalized blockchain feed that the explorer expects.

    Transforms the node's snake_case fields into the explorer's expected keys.
    Returns an object with `chain` (array of blocks) and `height`.
    """
    chain = state.node.fetch_chain(limit=limit)

    def map_block(b: Dict) -> Dict:
        return {
            "index": b.get("index"),
            "timestamp": b.get("timestamp"),
            "previousHash": b.get("previous_hash"),
            "hash": b.get("hash"),
            "nonce": b.get("nonce"),
            "difficulty": b.get("difficulty"),
            # keep any additional fields if needed in the future
            "transactions": [
                {
                    # explorer supports id/hash/txid, provide hash for readability
                    "id": tx.get("hash") or tx.get("id") or tx.get("txid"),
                    "hash": tx.get("hash"),
                    "from": tx.get("from"),
                    "to": tx.get("to"),
                    "amount": tx.get("amount"),
                    "type": tx.get("type", "transfer"),
                    "timestamp": tx.get("timestamp"),
                }
                for tx in b.get("transactions", [])
            ],
        }

    normalized = [map_block(b) for b in chain]
    height = normalized[-1]["index"] if normalized else 0
    return {"chain": normalized, "height": height}


@app.get("/earnings")
def earnings():
    """Aggregate pool vs miner earnings (settled & pending).

    Returns totals for:
    - pool_fee_settled: sum of operator fees for settled rounds
    - pool_fee_pending: operator fees for unsettled rounds
    - miners_payout_settled: sum of distributable rewards for settled rounds
    - miners_payout_pending: distributable rewards for unsettled rounds
    - settled_rounds / pending_rounds counts
    - latest_settlement_timestamp if any
    """
    with state.lock:
        pool_fee_settled = 0.0
        pool_fee_pending = 0.0
        miners_payout_settled = 0.0
        miners_payout_pending = 0.0
        settled_rounds = 0
        pending_rounds = 0
        for rnd in state.rounds:
            reward = float(rnd.get("reward_amount", 0.0) or 0.0)
            fee_percent = float(rnd.get("fee_percent", state.config.pool_fee_percent) or 0.0)
            op_fee_val = rnd.get("operator_fee_amount")
            if op_fee_val is not None:
                try:
                    operator_fee = float(op_fee_val)
                except (ValueError, TypeError):
                    operator_fee = reward * (fee_percent / 100.0)
            else:
                operator_fee = reward * (fee_percent / 100.0)
            distributable = max(reward - operator_fee, 0.0)
            if rnd.get("settled"):
                settled_rounds += 1
                pool_fee_settled += operator_fee
                miners_payout_settled += distributable
            else:
                pending_rounds += 1
                pool_fee_pending += operator_fee
                miners_payout_pending += distributable
        latest_settlement_ts = state.settlements[-1]["timestamp"] if state.settlements else None
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "pool_fee_settled": round(pool_fee_settled, 8),
        "pool_fee_pending": round(pool_fee_pending, 8),
        "miners_payout_settled": round(miners_payout_settled, 8),
        "miners_payout_pending": round(miners_payout_pending, 8),
        "settled_rounds": settled_rounds,
        "pending_rounds": pending_rounds,
        "latest_settlement_timestamp": latest_settlement_ts,
        "fee_percent": state.config.pool_fee_percent,
    }


def run():
    import uvicorn
    uvicorn.run(
        "pool.server:app",
        host=config.host,
        port=config.port,
        reload=bool(os.getenv("NILZ_POOL_RELOAD", "")),
    )


if __name__ == "__main__":
    run()
