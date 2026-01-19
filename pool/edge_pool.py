"""Mining Edge Node (Worker-only gateway)

This FastAPI app exposes only the endpoints miners need (/register, /work, /submit)
and forwards them to an upstream pool. Use it as a public-facing gateway while
keeping your core pool private.

Env vars:
  NILZ_EDGE_PORT         Port to bind (default: 8010)
  NILZ_EDGE_HOST         Host to bind (default: 0.0.0.0)
  NILZ_POOL_UPSTREAM     Upstream pool base URL (e.g., http://127.0.0.1:8000)
  NILZ_EDGE_ID           Optional identifier for this edge (included in /health)
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict

import requests
from fastapi import FastAPI, HTTPException, Request

log = logging.getLogger("nilz.pool.edge")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

UPSTREAM = os.getenv("NILZ_POOL_UPSTREAM", "http://127.0.0.1:8000").rstrip("/")
EDGE_ID = os.getenv("NILZ_EDGE_ID", "edge-1")

session = requests.Session()

app = FastAPI(title="NilzCoin Mining Edge", version="0.1.0")


def _http_error(resp: requests.Response) -> HTTPException:
    try:
        detail = resp.json()
    except Exception:
        detail = resp.text
    return HTTPException(status_code=resp.status_code, detail=detail)


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        r = session.get(f"{UPSTREAM}/health", timeout=5)
        r.raise_for_status()
        upstream = r.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail=f"Upstream unreachable: {exc}")
    return {
        "status": "ok",
        "edge_id": EDGE_ID,
        "upstream": UPSTREAM,
        "upstream_health": upstream,
    }


@app.post("/register")
async def register(request: Request) -> Dict[str, Any]:
    try:
        data = await request.json()
        r = session.post(f"{UPSTREAM}/register", json=data, timeout=10)
        if r.status_code >= 400:
            raise _http_error(r)
        return r.json()
    except HTTPException:
        raise
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")


@app.get("/work")
def get_work(miner_id: str) -> Dict[str, Any]:
    try:
        r = session.get(f"{UPSTREAM}/work", params={"miner_id": miner_id}, timeout=10)
        if r.status_code >= 400:
            raise _http_error(r)
        return r.json()
    except HTTPException:
        raise
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")


@app.post("/submit")
async def submit(request: Request) -> Dict[str, Any]:
    try:
        payload = await request.json()
        r = session.post(f"{UPSTREAM}/submit", json=payload, timeout=10)
        if r.status_code >= 400:
            raise _http_error(r)
        return r.json()
    except HTTPException:
        raise
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")


# Optional: simple root page
@app.get("/")
def root() -> Dict[str, Any]:
    return {"service": "nilz-edge", "edge_id": EDGE_ID, "upstream": UPSTREAM}
