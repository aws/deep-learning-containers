"""Minimal disaggregated prefill/decode proxy for the NIXL EFA test.

Modeled after upstream vLLM's tests/v1/kv_connector/nixl_integration/toy_proxy_server.py
but stripped to the minimum needed for the EFA validation: route a /v1/completions
request to the prefill server (max_tokens=1, returns a single sampling step), then
forward the same request body to the decode server, which receives the KV cache
over NIXL+LIBFABRIC+EFA and produces the rest of the tokens.
"""

import argparse
import logging

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="[proxy] %(message)s")
log = logging.getLogger(__name__)

app = FastAPI()
clients: dict[str, httpx.AsyncClient] = {}
prefill_url = ""
decode_url = ""


@app.on_event("startup")
async def _startup() -> None:
    clients["prefill"] = httpx.AsyncClient(base_url=prefill_url, timeout=300.0)
    clients["decode"] = httpx.AsyncClient(base_url=decode_url, timeout=300.0)


@app.on_event("shutdown")
async def _shutdown() -> None:
    for c in clients.values():
        await c.aclose()


@app.get("/healthcheck")
async def healthcheck() -> dict:
    return {"prefill": prefill_url, "decode": decode_url}


@app.post("/v1/completions")
async def completions(req: Request) -> JSONResponse:
    body = await req.json()
    prompt = body.get("prompt", "")

    # Step 1: prefill (max_tokens=1) — populates KV cache and triggers NIXL push.
    prefill_body = {**body, "max_tokens": 1}
    pf = await clients["prefill"].post("/v1/completions", json=prefill_body)
    log.info("prefill status=%s", pf.status_code)
    if pf.status_code != 200:
        return JSONResponse(status_code=pf.status_code, content={"error": pf.text})

    # Step 2: decode — same request; server should pull KV from prefill via NIXL.
    dc = await clients["decode"].post("/v1/completions", json=body)
    log.info("decode status=%s prompt=%r", dc.status_code, prompt[:40])
    return JSONResponse(status_code=dc.status_code, content=dc.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8192)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    args = parser.parse_args()
    prefill_url = args.prefill_url
    decode_url = args.decode_url
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
