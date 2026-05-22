"""Disaggregated prefill/decode proxy for the NIXL EFA test.

Modeled after upstream vLLM's
tests/v1/kv_connector/nixl_integration/toy_proxy_server.py: do the
two-step kv_transfer_params handshake so the decode side actually pulls
the prefilled KV cache from the prefill side over NIXL+LIBFABRIC instead
of silently re-prefilling locally.

NixlConnector (vLLM 0.21.0) ignores `kv_role` at the engine level — the
per-request `kv_transfer_params` dict is what determines remote-fetch
behavior:

  Step 1 (P): inject placeholder params {do_remote_decode: True, ...}
              and force max_tokens=1. Prefill computes KV, fills in
              remote_engine_id/remote_block_ids/remote_host/remote_port
              in its response.
  Step 2 (D): read those populated params from P's response and put them
              in D's request body. D's scheduler sees do_remote_prefill=
              True + populated remotes and pulls KV blocks instead of
              recomputing.
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

# vLLM treats kv_transfer_params as a forward-compat dict; min_tokens isn't
# supported on the prefill side, so we strip it temporarily.
_DROP_FIELDS_FOR_PREFILL = ("min_tokens", "min_completion_tokens")


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

    # ---- Step 1: prefill request ----
    pf_body = dict(body)
    pf_body["max_tokens"] = 1
    if "max_completion_tokens" in pf_body:
        pf_body["max_completion_tokens"] = 1
    saved_min: dict[str, object] = {}
    for k in _DROP_FIELDS_FOR_PREFILL:
        if k in pf_body:
            saved_min[k] = pf_body.pop(k)
    pf_body["stream"] = False
    # Placeholder kv_transfer_params: tells prefill it's a P-node and to
    # populate the remote_* fields in its response.
    pf_body["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }

    pf = await clients["prefill"].post("/v1/completions", json=pf_body)
    log.info("prefill status=%s", pf.status_code)
    if pf.status_code != 200:
        return JSONResponse(status_code=pf.status_code, content={"error": pf.text})

    pf_json = pf.json()
    kv_params = pf_json.get("kv_transfer_params") or {}
    if not kv_params.get("remote_engine_id"):
        log.warning("prefill returned no remote_engine_id; KV handoff will fail")
    log.info(
        "prefill kv_transfer_params: engine_id=%s blocks=%s host=%s port=%s",
        kv_params.get("remote_engine_id"),
        len(kv_params.get("remote_block_ids") or []),
        kv_params.get("remote_host"),
        kv_params.get("remote_port"),
    )

    # ---- Step 2: decode request ----
    dc_body = dict(body)
    dc_body["kv_transfer_params"] = kv_params
    # Restore stripped fields, if any.
    for k, v in saved_min.items():
        dc_body[k] = v

    dc = await clients["decode"].post("/v1/completions", json=dc_body)
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
