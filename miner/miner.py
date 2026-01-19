"""NilzCoin CPU miner that talks to the pool service."""

from __future__ import annotations

import argparse
import json
import logging
import random
import threading
import time
from math import log10
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List

import hashlib
import requests

log = logging.getLogger("nilz.miner")

def configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    # Clear root handlers to avoid duplicates if reconfigured
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logging.basicConfig(
        level=numeric,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def canonical_json(payload: Dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_scheme(url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return f"http://{url}"


@dataclass
class MinerConfig:
    wallet_address: str
    pool_address: str
    worker_name: str = "worker"
    threads: int = 1
    stats_interval: int = 30

    @classmethod
    def load(cls, path: Path) -> "MinerConfig":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(**data)


class PoolSession:
    def __init__(self, base_url: str):
        self.base_url = ensure_scheme(base_url.rstrip("/"))
        self.session = requests.Session()
        self.miner_id: Optional[str] = None
        self.poll_interval = 5

    def register(self, miner_name: str, wallet_address: str) -> None:
        payload = {"miner_name": miner_name, "wallet_address": wallet_address}
        resp = self.session.post(f"{self.base_url}/register", json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        self.miner_id = data["miner_id"]
        self.poll_interval = data.get("poll_interval", self.poll_interval)
        log.info("Registered with pool %s as miner_id=%s", self.base_url, self.miner_id)

    def get_work(self) -> Dict:
        if not self.miner_id:
            raise RuntimeError("Miner not registered")
        resp = self.session.get(
            f"{self.base_url}/work",
            params={"miner_id": self.miner_id},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return data

    def submit_share(self, job_id: str, nonce: int, block_hash: str, header: Dict) -> Dict:
        if not self.miner_id:
            raise RuntimeError("Miner not registered")
        payload = {
            "miner_id": self.miner_id,
            "job_id": job_id,
            "nonce": nonce,
            "block_hash": block_hash,
            "header": header,
        }
        resp = self.session.post(f"{self.base_url}/submit", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()


class MiningWorker(threading.Thread):
    def __init__(self, session: PoolSession, stop_event: threading.Event, worker_id: int):
        super().__init__(daemon=True)
        self.session = session
        self.stop_event = stop_event
        self.worker_id = worker_id
        self.shares = 0
        self.blocks = 0
        self.hashes = 0  # total hash attempts
        self.last_target_int: Optional[int] = None
        self.last_hashes = 0  # for per-worker instantaneous rate if desired

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                job = self.session.get_work()
            except requests.RequestException as exc:
                log.warning("Worker %s failed to get work: %s", self.worker_id, exc)
                time.sleep(self.session.poll_interval)
                continue
            self.mine_job(job)

    def mine_job(self, job: Dict) -> None:
        template = job
        header = {
            "index": template["index"],
            "previous_hash": template["previous_hash"],
            "merkle_root": template["merkle_root"],
            "timestamp": template["timestamp"],
            "difficulty": template["difficulty"],
            "target": template["target"],
            "nonce": 0,
        }
        max_nonce = template.get("max_nonce") or (2**32 - 1)
        share_target = int(job.get("share_target", job["target"]), 16)
        block_target = int(job["target"], 16)
        self.last_target_int = block_target
        start_nonce = random.randrange(0, max_nonce)
        deadline = time.time() + 60

        for attempt in range(max_nonce):
            if self.stop_event.is_set() or time.time() > deadline:
                break
            nonce = (start_nonce + attempt) % max_nonce
            header["nonce"] = nonce
            block_hash = sha256_hex(canonical_json(header))
            hash_val = int(block_hash, 16)
            self.hashes += 1
            if hash_val <= share_target:
                try:
                    result = self.session.submit_share(job["job_id"], nonce, block_hash, header.copy())
                except requests.RequestException as exc:
                    log.warning("Worker %s failed to submit share: %s", self.worker_id, exc)
                    break
                self.shares += 1
                is_block = hash_val <= block_target
                if is_block:
                    self.blocks += 1
                log.info(
                    "Worker %s share accepted (job=%s nonce=%s hash=%s block=%s)",
                    self.worker_id,
                    job["job_id"],
                    nonce,
                    block_hash[:16],
                    is_block,
                )
                if result.get("block_submitted"):
                    log.info("Worker %s submitted full block!", self.worker_id)
                break


def run_miner(config_path: Path) -> None:
    config = MinerConfig.load(config_path)
    session = PoolSession(config.pool_address)
    session.register(config.worker_name, config.wallet_address)
    stop_event = threading.Event()
    workers = [MiningWorker(session, stop_event, idx) for idx in range(1, config.threads + 1)]
    for worker in workers:
        worker.start()
    log.info("Started %s worker threads", len(workers))
    GENESIS_TARGET_INT = int("000000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", 16)
    prev_time = time.time()
    prev_hashes = 0
    # simple moving average over last N intervals
    avg_window: List[float] = []
    AVG_WINDOW_SIZE = 10

    def format_hashrate(hps: float) -> str:
        units = ["H/s", "kH/s", "MH/s", "GH/s", "TH/s", "PH/s"]
        if hps <= 0:
            return "0 H/s"
        idx = int(min(len(units) - 1, max(0, log10(hps) // 3)))
        scale = 1000 ** idx
        return f"{hps/scale:.2f} {units[idx]}"

    def read_cpu_temp() -> Optional[float]:
        # Try psutil first
        try:
            import psutil  # type: ignore
            temps = psutil.sensors_temperatures()
            for key in ("coretemp", "k10temp", "cpu-thermal", "acpitz"):
                if key in temps and temps[key]:
                    readings = [t.current for t in temps[key] if t.current is not None]
                    if readings:
                        return sum(readings) / len(readings)
            # Fallback: average all
            all_readings = []
            for arr in temps.values():
                all_readings.extend([t.current for t in arr if t.current is not None])
            if all_readings:
                return sum(all_readings) / len(all_readings)
        except Exception:
            pass
        # /sys/class/thermal fallback
        try:
            thermal = Path("/sys/class/thermal")
            vals = []
            for zone in thermal.glob("thermal_zone*/temp"):
                try:
                    raw = zone.read_text().strip()
                    val = float(raw) / (1000 if len(raw) > 3 else 1)
                    if 0 < val < 200:
                        vals.append(val)
                except Exception:
                    continue
            if vals:
                return sum(vals) / len(vals)
        except Exception:
            pass
        return None
    try:
        while True:
            time.sleep(config.stats_interval)
            total_shares = sum(worker.shares for worker in workers)
            total_blocks = sum(worker.blocks for worker in workers)
            total_hashes = sum(worker.hashes for worker in workers)
            now = time.time()
            interval = max(1e-6, now - prev_time)
            interval_hashes = total_hashes - prev_hashes
            hashrate = interval_hashes / interval
            prev_time = now
            prev_hashes = total_hashes
            avg_window.append(hashrate)
            if len(avg_window) > AVG_WINDOW_SIZE:
                avg_window.pop(0)
            avg_hashrate = sum(avg_window) / len(avg_window)
            # derive difficulty ratio from most recent target (use smallest target seen for accuracy)
            latest_targets = [w.last_target_int for w in workers if w.last_target_int]
            current_target = min(latest_targets) if latest_targets else GENESIS_TARGET_INT
            difficulty_ratio = GENESIS_TARGET_INT / current_target if current_target else 0
            cpu_temp = read_cpu_temp()
            log.info(
                "Stats: shares=%s blocks=%s hr=%s avg=%s diff=%.4f target=%s%s",
                total_shares,
                total_blocks,
                format_hashrate(hashrate),
                format_hashrate(avg_hashrate),
                difficulty_ratio,
                f"{current_target:064x}",
                f" temp={cpu_temp:.1f}C" if cpu_temp is not None else "",
            )
    except KeyboardInterrupt:
        log.info("Stopping miners...")
        stop_event.set()
        for worker in workers:
            worker.join()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NilzCoin CPU miner")
    parser.add_argument(
        "--config",
        type=lambda value: Path(value).expanduser().resolve(),
        default=Path.cwd() / "miner_config.json",
        help="Path to miner_config.json",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default INFO",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Quiet mode; equivalent to --log-level WARNING"
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    level = "WARNING" if args.quiet else args.log_level
    configure_logging(level)
    try:
        run_miner(args.config)
    except FileNotFoundError:
        parser.error(f"Config file not found: {args.config}")
    except (requests.RequestException, RuntimeError) as exc:
        log.error("Fatal error: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
