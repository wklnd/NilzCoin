"""NilzCoin wallet CLI for creating addresses and signing transactions."""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import secrets
import sys
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Dict, List, Optional

import hashlib
import requests
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from ecdsa import SECP256k1, SigningKey


ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
DEFAULT_WALLET = Path.cwd() / "nilz.wallet"
KDF_ITERATIONS = 200_000


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


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


def checksum(payload: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]


def public_key_to_address(public_key: bytes) -> str:
    sha = hashlib.sha256(public_key).digest()
    ripe = hashlib.new("ripemd160", sha).digest()
    payload = b"\x35" + ripe  # version byte chosen for NilzCoin
    return "nilz" + b58encode(payload + checksum(payload))


def derive_cipher(passphrase: str, salt: bytes, iterations: int) -> Fernet:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
    )
    key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))
    return Fernet(key)


def canonical_json(payload: Dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def parse_amount(raw: str) -> Decimal:
    amount = Decimal(raw)
    if amount <= 0:
        raise argparse.ArgumentTypeError("Amount must be positive")
    return amount.quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)


def prompt_passphrase(provided: Optional[str], confirm: bool = False) -> str:
    if provided:
        return provided
    first = getpass.getpass("Enter wallet passphrase: ")
    if confirm:
        again = getpass.getpass("Confirm passphrase: ")
        if first != again:
            raise ValueError("Passphrases do not match")
    return first


@dataclass
class WalletKey:
    label: str
    address: str
    public_key: str
    encrypted_private_key: str
    created: str


class WalletFile:
    def __init__(self, path: Path, data: Dict):
        self.path = path
        self.data = data

    @classmethod
    def create(cls, path: Path, passphrase: str, label: str) -> "WalletFile":
        if path.exists():
            raise FileExistsError(f"Wallet file {path} already exists")
        salt = secrets.token_bytes(16)
        data = {
            "version": 1,
            "created": now_iso(),
            "kdf": {
                "salt": base64.b64encode(salt).decode(),
                "iterations": KDF_ITERATIONS,
            },
            "keys": [],
        }
        wallet = cls(path, data)
        wallet.add_key(label=label, passphrase=passphrase)
        wallet.save()
        return wallet

    @classmethod
    def load(cls, path: Path) -> "WalletFile":
        if not path.exists():
            raise FileNotFoundError(f"Wallet file {path} does not exist")
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(path, data)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(self.data, handle, indent=2)
        tmp_path.replace(self.path)

    def list_keys(self) -> List[WalletKey]:
        return [WalletKey(**entry) for entry in self.data.get("keys", [])]

    def _cipher(self, passphrase: str) -> Fernet:
        kdf_info = self.data["kdf"]
        salt = base64.b64decode(kdf_info["salt"])
        return derive_cipher(passphrase, salt, kdf_info["iterations"])

    def add_key(self, label: str, passphrase: str) -> WalletKey:
        if any(entry["label"] == label for entry in self.data["keys"]):
            raise ValueError(f"A key named '{label}' already exists")
        signing_key = SigningKey.generate(curve=SECP256k1)
        private_hex = signing_key.to_string().hex()
        verifying_key = signing_key.get_verifying_key()
        pub_bytes = b"\x04" + verifying_key.to_string()
        pub_hex = pub_bytes.hex()
        address = public_key_to_address(pub_bytes)
        cipher = self._cipher(passphrase)
        encrypted = cipher.encrypt(private_hex.encode()).decode()
        key_entry = {
            "label": label,
            "address": address,
            "public_key": pub_hex,
            "encrypted_private_key": encrypted,
            "created": now_iso(),
        }
        self.data.setdefault("keys", []).append(key_entry)
        self.save()
        return WalletKey(**key_entry)

    def _get_key_entry(self, label: Optional[str]) -> Dict:
        keys = self.data.get("keys", [])
        if not keys:
            raise ValueError("Wallet has no keys")
        if label is None:
            return keys[0]
        for entry in keys:
            if entry["label"] == label:
                return entry
        raise KeyError(f"No key labeled '{label}'")

    def unlock_signing_key(self, label: Optional[str], passphrase: str) -> SigningKey:
        entry = self._get_key_entry(label)
        cipher = self._cipher(passphrase)
        private_hex = cipher.decrypt(entry["encrypted_private_key"].encode()).decode()
        return SigningKey.from_string(bytes.fromhex(private_hex), curve=SECP256k1)

    def get_key_metadata(self, label: Optional[str]) -> WalletKey:
        entry = self._get_key_entry(label)
        return WalletKey(**entry)


def cmd_init(args: argparse.Namespace) -> None:
    passphrase = prompt_passphrase(args.passphrase, confirm=True)
    wallet = WalletFile.create(args.wallet, passphrase, label=args.label)
    print(f"Created wallet at {wallet.path} with address {wallet.list_keys()[0].address}")


def cmd_new_address(args: argparse.Namespace) -> None:
    passphrase = prompt_passphrase(args.passphrase)
    wallet = WalletFile.load(args.wallet)
    key = wallet.add_key(label=args.label, passphrase=passphrase)
    print(f"Added address {key.address} (label='{key.label}') to {wallet.path}")


def cmd_list(args: argparse.Namespace) -> None:
    wallet = WalletFile.load(args.wallet)
    keys = wallet.list_keys()
    if not keys:
        print("Wallet has no addresses yet")
        return
    for key in keys:
        print(f"{key.label:12} {key.address} (created {key.created})")


def craft_transaction(wallet: WalletFile, args: argparse.Namespace, passphrase: str) -> Dict:
    key_meta = wallet.get_key_metadata(args.from_label)
    signing_key = wallet.unlock_signing_key(args.from_label, passphrase)
    tx_core = {
        "from": key_meta.address,
        "to": args.to,
        "amount": str(args.amount),
        "timestamp": now_iso(),
        "nonce": secrets.token_hex(8),
        "public_key": key_meta.public_key,
    }
    payload = canonical_json(tx_core)
    signature = signing_key.sign_deterministic(payload, hashfunc=hashlib.sha256).hex()
    tx = dict(tx_core)
    tx["signature"] = signature
    tx["hash"] = hashlib.sha256(payload + bytes.fromhex(signature)).hexdigest()
    return tx


def broadcast_transaction(tx: Dict, node_url: str) -> None:
    endpoint = node_url.rstrip("/") + "/tx"
    try:
        response = requests.post(endpoint, json=tx, timeout=10)
        if response.status_code >= 400:
            print(f"Node rejected transaction: {response.status_code} {response.text}")
        else:
            print(f"Transaction submitted to {endpoint}")
    except requests.RequestException as exc:
        print(f"Failed to reach node: {exc}")


def cmd_send(args: argparse.Namespace) -> None:
    passphrase = prompt_passphrase(args.passphrase)
    wallet = WalletFile.load(args.wallet)
    tx = craft_transaction(wallet, args, passphrase)
    print("Signed transaction:")
    print(json.dumps(tx, indent=2))
    if args.node_url:
        broadcast_transaction(tx, args.node_url)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NilzCoin wallet CLI")
    parser.set_defaults(func=None)
    sub = parser.add_subparsers(dest="command")

    def add_common_wallet(subparser):
        subparser.add_argument(
            "--wallet",
            type=lambda value: Path(value).expanduser().resolve(),
            default=DEFAULT_WALLET,
            help=f"Wallet file path (default: {DEFAULT_WALLET})",
        )

    init_parser = sub.add_parser("init", help="Create a new wallet file")
    add_common_wallet(init_parser)
    init_parser.add_argument("--label", default="default", help="Label for the first address")
    init_parser.add_argument("--passphrase", help="Passphrase to encrypt the wallet")
    init_parser.set_defaults(func=cmd_init)

    new_parser = sub.add_parser("new-address", help="Add an additional address")
    add_common_wallet(new_parser)
    new_parser.add_argument("--label", required=True, help="Label for the new address")
    new_parser.add_argument("--passphrase", help="Wallet passphrase")
    new_parser.set_defaults(func=cmd_new_address)

    list_parser = sub.add_parser("list", help="List wallet addresses")
    add_common_wallet(list_parser)
    list_parser.set_defaults(func=cmd_list)

    send_parser = sub.add_parser("send", help="Create and broadcast a transaction")
    add_common_wallet(send_parser)
    send_parser.add_argument("--from-label", dest="from_label", help="Label of the sending address")
    send_parser.add_argument("--to", required=True, help="Destination address")
    send_parser.add_argument("--amount", required=True, type=parse_amount, help="Amount of nilz to send")
    send_parser.add_argument("--passphrase", help="Wallet passphrase")
    send_parser.add_argument("--node-url", default="http://127.0.0.1:5000", help="Node endpoint for broadcasting")
    send_parser.set_defaults(func=cmd_send)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1
    try:
        args.func(args)
    except (FileExistsError, FileNotFoundError, ValueError, KeyError) as err:
        print(f"Error: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
