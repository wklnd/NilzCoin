"""NilzCoin blockchain node service."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import hashlib
import os
import secrets
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from ecdsa import BadSignatureError, SECP256k1, VerifyingKey


log = logging.getLogger("nilz.node")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

getcontext().prec = 28

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
BLOCKCHAIN_FILE = DATA_DIR / "blockchain.json"
MEMPOOL_FILE = DATA_DIR / "mempool.json"

BLOCK_REWARD = Decimal("50")
# Initial target (harder than before). Lower value => harder.
# You can further adjust via retargeting below.
TARGET_HEX = "000000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
DIFFICULTY = 5  # cosmetic, please implement. 
MAX_TRANSACTIONS_PER_BLOCK = 50
MAX_NONCE = 2**32 - 1
GENESIS_ADDRESS = "nilzFoundation0000000000000000000000000000"
ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

# Difficulty retargeting configuration
DESIRED_BLOCK_TIME_S = int(os.getenv("NILZ_BLOCK_TIME_S", "30"))
RETARGET_WINDOW = int(os.getenv("NILZ_RETARGET_WINDOW", "30"))  # number of blocks to average (>= 2)
# Clamp how fast difficulty can change per retarget step
RETARGET_MIN_FACTOR = 0.25  # at most 4x harder per step
RETARGET_MAX_FACTOR = 4.0   # at most 4x easier per step
UINT256_MAX = (1 << 256) - 1


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checksum(payload: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]


def b58encode(data: bytes) -> str:
    value = int.from_bytes(data, "big")
    encoded = ""
    while value:
        value, remainder = divmod(value, 58)
        encoded = ALPHABET[remainder] + encoded
    prefix = ""
    for byte in data:
        if byte == 0:
            prefix += ALPHABET[0]
        else:
            break
    return prefix + encoded or ALPHABET[0]


def public_key_to_address(public_key_bytes: bytes) -> str:
    sha = hashlib.sha256(public_key_bytes).digest()
    ripe = hashlib.new("ripemd160", sha).digest()
    payload = b"\x35" + ripe
    return "nilz" + b58encode(payload + checksum(payload))


def merkle_root(transactions: List[Dict[str, Any]]) -> str:
    if not transactions:
        return sha256_hex(b"")
    layer = [bytes.fromhex(tx["hash"]) for tx in transactions]
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        next_layer = []
        for i in range(0, len(layer), 2):
            next_layer.append(hashlib.sha256(layer[i] + layer[i + 1]).digest())
        layer = next_layer
    return layer[0].hex()


def parse_amount(value: str) -> Decimal:
    amount = Decimal(value)
    if amount <= 0:
        raise ValueError("Amount must be positive")
    return amount.quantize(Decimal("0.00000001"))


def block_header_dict(block_like: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "index": block_like["index"],
        "previous_hash": block_like["previous_hash"],
        "merkle_root": block_like["merkle_root"],
        "timestamp": block_like["timestamp"],
        "nonce": block_like["nonce"],
        "difficulty": block_like["difficulty"],
        "target": block_like["target"],
    }


def hash_block_header(block_like: Dict[str, Any]) -> str:
    return sha256_hex(canonical_json(block_header_dict(block_like)))


def verify_transaction_signature(tx: Dict[str, Any]) -> None:
    if tx.get("type") == "coinbase":
        return
    required = ["from", "to", "amount", "timestamp", "nonce", "signature", "public_key", "hash"]
    for key in required:
        if key not in tx:
            raise ValueError(f"Transaction missing field '{key}'")
    pub_bytes = bytes.fromhex(tx["public_key"])
    if not pub_bytes or pub_bytes[0] != 0x04:
        raise ValueError("Public key must be uncompressed and start with 0x04")
    verifying_key = VerifyingKey.from_string(pub_bytes[1:], curve=SECP256k1)
    core = {
        "amount": tx["amount"],
        "from": tx["from"],
        "nonce": tx["nonce"],
        "public_key": tx["public_key"],
        "timestamp": tx["timestamp"],
        "to": tx["to"],
    }
    payload = canonical_json(core)
    signature_bytes = bytes.fromhex(tx["signature"])
    try:
        verifying_key.verify(signature_bytes, payload, hashfunc=hashlib.sha256)
    except BadSignatureError as exc:
        raise ValueError("Invalid signature") from exc
    expected_hash = sha256_hex(payload + signature_bytes)
    if expected_hash != tx["hash"]:
        raise ValueError("Transaction hash mismatch")
    derived_address = public_key_to_address(pub_bytes)
    if derived_address != tx["from"]:
        raise ValueError("Public key does not match from address")


@dataclass
class NodeStats:
    blocks: int
    transactions: int
    mempool_size: int
    last_block_time: str


class BlockchainNode:
    def __init__(self, chain_path: Path, mempool_path: Path):
        self.chain_path = chain_path
        self.mempool_path = mempool_path
        self.lock = threading.RLock()
        self.chain: List[Dict[str, Any]] = []
        self.mempool: List[Dict[str, Any]] = []
        # Current working target as int (dynamic difficulty)
        self.current_target: int = int(TARGET_HEX, 16)
        self._load()
        # Initialize current_target from the latest block on disk
        try:
            self.current_target = int(self.latest_block().get("target", TARGET_HEX), 16)
        except Exception:
            self.current_target = int(TARGET_HEX, 16)

    # Persistence helpers -------------------------------------------------

    def _load(self) -> None:
        if self.chain_path.exists():
            with self.chain_path.open("r", encoding="utf-8") as handle:
                self.chain = json.load(handle)
        if self.mempool_path.exists():
            with self.mempool_path.open("r", encoding="utf-8") as handle:
                self.mempool = json.load(handle)
        if not self.chain:
            self._create_genesis_block()
        self._persist_chain()
        self._persist_mempool()

    def _persist_chain(self) -> None:
        with self.chain_path.open("w", encoding="utf-8") as handle:
            json.dump(self.chain, handle, indent=2)

    def _persist_mempool(self) -> None:
        with self.mempool_path.open("w", encoding="utf-8") as handle:
            json.dump(self.mempool, handle, indent=2)

    # Genesis --------------------------------------------------------------

    def _create_genesis_block(self) -> None:
        coinbase = self._build_coinbase_tx(GENESIS_ADDRESS)
        block = {
            "index": 0,
            "previous_hash": "0" * 64,
            "timestamp": now_iso(),
            "nonce": 0,
            "difficulty": DIFFICULTY,
            "target": TARGET_HEX,
            "transactions": [coinbase],
        }
        block["merkle_root"] = merkle_root(block["transactions"])
        block["hash"] = hash_block_header(block)
        self.chain = [block]
        self.mempool = []
        self._persist_chain()
        self._persist_mempool()
        log.info("Created genesis block")

    # Chain utilities ------------------------------------------------------

    def latest_block(self) -> Dict[str, Any]:
        return self.chain[-1]

    def balances(self) -> Dict[str, Decimal]:
        ledger: Dict[str, Decimal] = {}
        for block in self.chain:
            for tx in block["transactions"]:
                amount = Decimal(tx["amount"])
                if tx.get("type") == "coinbase":
                    ledger[tx["to"]] = ledger.get(tx["to"], Decimal("0")) + amount
                else:
                    sender = tx["from"]
                    ledger[sender] = ledger.get(sender, Decimal("0")) - amount
                    ledger[tx["to"]] = ledger.get(tx["to"], Decimal("0")) + amount
        return ledger

    def pending_outgoing(self, address: str) -> Decimal:
        total = Decimal("0")
        for tx in self.mempool:
            if tx.get("from") == address and tx.get("type") != "coinbase":
                total += Decimal(tx["amount"])
        return total

    def address_summary(self, address: str, limit: int = 100) -> Dict[str, Any]:
        if not address.startswith("nilz"):
            raise ValueError("Invalid address")
        with self.lock:
            balances = self.balances()
            confirmed = balances.get(address, Decimal("0"))
            pending_out = self.pending_outgoing(address)
            history = self._collect_transactions(address, limit)
            return {
                "address": address,
                "balance": str(confirmed),
                "spendable": str(max(confirmed - pending_out, Decimal("0"))),
                "pending_outgoing": str(pending_out),
                "transactions": history,
            }

    def _collect_transactions(self, address: str, limit: int) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        for tx in self.mempool:
            participants = {tx.get("from"), tx.get("to")}
            if address not in participants:
                continue
            history.append(
                {
                    "hash": tx["hash"],
                    "type": tx.get("type", "transfer"),
                    "from": tx.get("from"),
                    "to": tx.get("to"),
                    "amount": tx["amount"],
                    "timestamp": tx["timestamp"],
                    "status": "pending",
                    "block_index": None,
                    "block_hash": None,
                }
            )
        for block in reversed(self.chain):
            for tx in block["transactions"]:
                if tx.get("type") == "coinbase" and tx.get("to") != address:
                    continue
                participants = {tx.get("from"), tx.get("to")}
                if address not in participants:
                    continue
                history.append(
                    {
                        "hash": tx["hash"],
                        "type": tx.get("type", "transfer"),
                        "from": tx.get("from"),
                        "to": tx.get("to"),
                        "amount": tx["amount"],
                        "timestamp": tx["timestamp"],
                        "status": "confirmed",
                        "block_index": block["index"],
                        "block_hash": block["hash"],
                    }
                )
        history.sort(key=lambda tx: tx["timestamp"], reverse=True)
        return history[:limit]

    # Transactions ---------------------------------------------------------

    def add_transaction(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            self._validate_transaction(tx)
            self.mempool.append(tx)
            self._persist_mempool()
            log.info("Accepted transaction %s", tx["hash"])
            return {"accepted": True, "mempool_size": len(self.mempool)}

    def _validate_transaction(self, tx: Dict[str, Any]) -> None:
        if tx.get("type") == "coinbase":
            raise ValueError("Client cannot submit coinbase transactions")
        verify_transaction_signature(tx)
        amount = parse_amount(tx["amount"])
        balances = self.balances()
        sender_balance = balances.get(tx["from"], Decimal("0"))
        pending = self.pending_outgoing(tx["from"])
        if sender_balance - pending < amount:
            raise ValueError("Insufficient balance")
        duplicate = any(existing["hash"] == tx["hash"] for existing in self.mempool)
        if duplicate or any(tx["hash"] == chain_tx["hash"] for block in self.chain for chain_tx in block["transactions"]):
            raise ValueError("Duplicate transaction")

    def _basic_tx_checks(self, tx: Dict[str, Any]) -> None:
        """Perform signature & structural checks without mempool balance/duplicate logic.

        Used when validating transactions already selected from the mempool for a
        candidate block. At block validation time we must not treat presence in
        the mempool as duplication, nor subtract pending outgoing twice.
        """
        if tx.get("type") == "coinbase":
            return  # coinbase handled separately in block validation
        verify_transaction_signature(tx)
        # Ensure not already confirmed in chain
        if any(tx["hash"] == chain_tx["hash"] for block in self.chain for chain_tx in block["transactions"]):
            raise ValueError("Duplicate transaction (already confirmed)")
        # Positive amount constraint
        parse_amount(tx["amount"])  # will raise if invalid

    # Work -----------------------------------------------------------------

    def build_block_template(self, reward_address: str) -> Dict[str, Any]:
        with self.lock:
            tx_candidates = self.mempool[:MAX_TRANSACTIONS_PER_BLOCK]
            transactions = [self._build_coinbase_tx(reward_address)] + tx_candidates
            target_hex = f"{self.current_target:064x}"
            template = {
                "index": len(self.chain),
                "previous_hash": self.latest_block()["hash"],
                "timestamp": now_iso(),
                "difficulty": DIFFICULTY,
                "target": target_hex,
                "nonce": 0,
                "transactions": transactions,
                "merkle_root": merkle_root(transactions),
                "max_nonce": MAX_NONCE,
            }
            return template

    def _build_coinbase_tx(self, address: str) -> Dict[str, Any]:
        tx = {
            "type": "coinbase",
            "to": address,
            "amount": str(BLOCK_REWARD),
            "timestamp": now_iso(),
            "nonce": secrets.token_hex(8),
        }
        payload = canonical_json({k: tx[k] for k in sorted(tx)})
        tx["hash"] = sha256_hex(payload)
        return tx

    # Blocks ---------------------------------------------------------------

    def submit_block(self, submission: Dict[str, Any]) -> Dict[str, Any]:
        header = submission.get("header") or {}
        template = submission.get("template") or {}
        provided_hash = submission.get("hash")
        if not header or not template or not provided_hash:
            raise HTTPException(status_code=400, detail="Submission missing header/template/hash")
        with self.lock:
            block = self._build_block_from_submission(header, template, provided_hash)
            self.chain.append(block)
            self._remove_confirmed_transactions(block)
            self._persist_chain()
            self._persist_mempool()
            # Retarget difficulty for subsequent work based on recent block times
            self._retarget()
            log.info("Block %s appended with %d tx", block["index"], len(block["transactions"]))
            return {"accepted": True, "height": block["index"], "hash": block["hash"]}

    def _build_block_from_submission(self, header: Dict[str, Any], template: Dict[str, Any], provided_hash: str) -> Dict[str, Any]:
        if header.get("index") != len(self.chain):
            raise HTTPException(status_code=400, detail="Index mismatch")
        if header.get("previous_hash") != self.latest_block()["hash"]:
            raise HTTPException(status_code=400, detail="Previous hash mismatch")
        expected_target_hex = f"{self.current_target:064x}"
        if header.get("target") != expected_target_hex:
            raise HTTPException(status_code=400, detail="Unexpected target (stale or mismatched)")
        if not (0 <= header.get("nonce", -1) <= MAX_NONCE):
            raise HTTPException(status_code=400, detail="Nonce out of range")
        expected_merkle = merkle_root(template.get("transactions", []))
        if header.get("merkle_root") != expected_merkle:
            raise HTTPException(status_code=400, detail="Merkle root mismatch")
        computed_hash = hash_block_header(header)
        if computed_hash != provided_hash:
            raise HTTPException(status_code=400, detail="Header hash mismatch")
        if int(provided_hash, 16) > self.current_target:
            raise HTTPException(status_code=400, detail="Hash does not meet target")
        self._validate_block_transactions(template.get("transactions", []))
        block = dict(header)
        block["hash"] = provided_hash
        block["transactions"] = template.get("transactions", [])
        return block

    # --- Difficulty retargeting -----------------------------------------

    @staticmethod
    def _parse_time(ts: str) -> datetime:
        # Support the Z suffix used by now_iso()
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)

    def _retarget(self) -> None:
        # Need at least 2 blocks to compute an interval
        n = len(self.chain)
        if n < 2 or DESIRED_BLOCK_TIME_S <= 0:
            return
        window = max(2, min(RETARGET_WINDOW, n))
        recent = self.chain[-window:]
        t_first = self._parse_time(recent[0]["timestamp"])
        t_last = self._parse_time(recent[-1]["timestamp"])
        elapsed = max(1.0, (t_last - t_first).total_seconds())
        avg_interval = elapsed / (len(recent) - 1)
        # Compute adjustment ratio and clamp
        ratio = avg_interval / float(DESIRED_BLOCK_TIME_S)
        ratio = max(RETARGET_MIN_FACTOR, min(RETARGET_MAX_FACTOR, ratio))
        new_target = int(self.current_target * ratio)
        # Clamp target into valid uint256 range (at least 1)
        new_target = max(1, min(UINT256_MAX, new_target))
        if new_target != self.current_target:
            self.current_target = new_target
            log.info(
                "Retarget: avg=%.2fs desired=%ss ratio=%.3f new_target=%s",
                avg_interval,
                DESIRED_BLOCK_TIME_S,
                ratio,
                f"{self.current_target:064x}",
            )

    def _validate_block_transactions(self, transactions: List[Dict[str, Any]]) -> None:
        if not transactions:
            raise HTTPException(status_code=400, detail="Block must contain coinbase transaction")
        coinbase = transactions[0]
        if coinbase.get("type") != "coinbase":
            raise HTTPException(status_code=400, detail="First transaction must be coinbase")
        if Decimal(coinbase.get("amount", "0")) != BLOCK_REWARD:
            raise HTTPException(status_code=400, detail="Coinbase amount incorrect")
        seen_hashes = set()
        ledger = self.balances()
        for tx in transactions:
            if tx["hash"] in seen_hashes:
                raise HTTPException(status_code=400, detail="Duplicate transaction in block")
            seen_hashes.add(tx["hash"])
            if tx.get("type") == "coinbase":
                ledger[tx["to"]] = ledger.get(tx["to"], Decimal("0")) + Decimal(tx["amount"])
                continue
            # Perform signature & structural validation excluding mempool duplication & pending checks
            try:
                self._basic_tx_checks(tx)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            amount = Decimal(tx["amount"])
            sender = tx["from"]
            balance = ledger.get(sender, Decimal("0"))
            # At block validation time we use rolling ledger so only current balance matters
            if balance < amount:
                raise HTTPException(status_code=400, detail="Block contains overspent transaction")
            ledger[sender] = balance - amount
            ledger[tx["to"]] = ledger.get(tx["to"], Decimal("0")) + amount

    def _remove_confirmed_transactions(self, block: Dict[str, Any]) -> None:
        confirmed_hashes = {tx["hash"] for tx in block["transactions"] if tx.get("type") != "coinbase"}
        self.mempool = [tx for tx in self.mempool if tx["hash"] not in confirmed_hashes]

    # Stats ----------------------------------------------------------------

    def stats(self) -> NodeStats:
        last_time = self.latest_block()["timestamp"]
        tx_count = sum(len(block["transactions"]) for block in self.chain)
        return NodeStats(
            blocks=len(self.chain),
            transactions=tx_count,
            mempool_size=len(self.mempool),
            last_block_time=last_time,
        )


class TransactionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    from_address: Optional[str] = Field(None, alias="from")
    to: str
    amount: str
    timestamp: str
    nonce: str
    signature: Optional[str] = None
    public_key: Optional[str] = None
    hash: str
    type: str = "transfer"

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True)


class BlockSubmissionRequest(BaseModel):
    miner_id: Optional[str] = None
    job_id: Optional[str] = None
    nonce: int
    header: Dict[str, Any]
    template: Dict[str, Any]
    hash: str


node = BlockchainNode(BLOCKCHAIN_FILE, MEMPOOL_FILE)
app = FastAPI(title="NilzCoin Node", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    stats = node.stats()
    return {
        "status": "ok",
        "blocks": stats.blocks,
        "mempool": stats.mempool_size,
        "last_block_time": stats.last_block_time,
    }


@app.get("/chain")
def chain(limit: Optional[int] = None):
    chain_data = node.chain[-limit:] if limit else node.chain
    return chain_data


@app.get("/mempool")
def mempool():
    return node.mempool


@app.get("/address/{address}")
def address_details(address: str, limit: int = 100):
    try:
        return node.address_summary(address, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/tx")
def submit_transaction(request: TransactionRequest):
    tx = request.to_dict()
    result = node.add_transaction(tx)
    return result


@app.get("/work")
def get_work(address: str):
    if not address.startswith("nilz"):
        raise HTTPException(status_code=400, detail="Invalid reward address")
    template = node.build_block_template(address)
    return template


@app.post("/block")
def submit_block(request: BlockSubmissionRequest):
    result = node.submit_block(request.model_dump())
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("node.server:app", host="0.0.0.0", port=5000, reload=True)
