"""Public Relay Node (Wallet/API gateway)

This FastAPI app exposes public read endpoints and a transaction submission
endpoint, forwarding them to a private upstream NilzCoin node. It does NOT
expose mining endpoints.

Env vars:
  NILZ_PUBLIC_PORT       Port to bind (default: 5010)
  NILZ_PUBLIC_HOST       Host to bind (default: 0.0.0.0)
  NILZ_NODE_UPSTREAM     Upstream node base URL (e.g., http://127.0.0.1:5000)
  NILZ_PUBLIC_ID         Optional identifier for this relay (included in /)
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

log = logging.getLogger("nilz.node.edge")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

UPSTREAM = os.getenv("NILZ_NODE_UPSTREAM", "http://127.0.0.1:5000").rstrip("/")
EDGE_ID = os.getenv("NILZ_PUBLIC_ID", "public-node-1")

session = requests.Session()

app = FastAPI(title="NilzCoin Public Relay", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _http_error(resp: requests.Response) -> HTTPException:
    try:
        detail = resp.json()
    except Exception:
        detail = resp.text
    return HTTPException(status_code=resp.status_code, detail=detail)


@app.get("/")
def root() -> Dict[str, Any]:
    return {"service": "nilz-public-relay", "id": EDGE_ID, "upstream": UPSTREAM}


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        r = session.get(f"{UPSTREAM}/health", timeout=5)
        r.raise_for_status()
        upstream = r.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail=f"Upstream unreachable: {exc}")
    return {"status": "ok", "id": EDGE_ID, "upstream": UPSTREAM, "upstream_health": upstream}


@app.get("/chain")
def chain(limit: int | None = None):
    try:
        params = {"limit": limit} if limit else None
        r = session.get(f"{UPSTREAM}/chain", params=params, timeout=10)
        if r.status_code >= 400:
            raise _http_error(r)
        return r.json()
    except HTTPException:
        raise
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")


@app.get("/mempool")
def mempool():
    try:
        r = session.get(f"{UPSTREAM}/mempool", timeout=10)
        if r.status_code >= 400:
            raise _http_error(r)
        return r.json()
    except HTTPException:
        raise
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")


@app.get("/address/{address}")
def address_details(address: str, limit: int = 100):
    try:
        r = session.get(f"{UPSTREAM}/address/{address}", params={"limit": limit}, timeout=10)
        if r.status_code >= 400:
            raise _http_error(r)
        return r.json()
    except HTTPException:
        raise
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")


@app.post("/tx")
async def submit_transaction(request: Request) -> Dict[str, Any]:
    try:
        payload = await request.json()
        r = session.post(f"{UPSTREAM}/tx", json=payload, timeout=10)
        if r.status_code >= 400:
            raise _http_error(r)
        return r.json()
    except HTTPException:
        raise
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("NILZ_PUBLIC_HOST", "0.0.0.0")
    port = int(os.getenv("NILZ_PUBLIC_PORT", "5010"))
    uvicorn.run("node.edge_node:app", host=host, port=port, reload=True)
