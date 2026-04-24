# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SageMaker HTTP proxy for llama.cpp llama-server.

SageMaker invokes POST /invocations and GET /ping on port 8080. llama-server
speaks OpenAI-style routes (e.g. /v1/chat/completions) and does not expose
/invocations.

Behavior mirrors scripts/vllm/omni_sagemaker_serve.py routing:

- GET /ping is proxied to GET {backend}/health.
- POST /invocations: if ``X-Amzn-SageMaker-Custom-Attributes`` contains
  ``route=/some/path``, the request is forwarded to that path on llama-server.
  Otherwise the target path is inferred from the JSON body (messages ->
  /v1/chat/completions, prompt -> /v1/completions, input+model -> /v1/embeddings),
  defaulting to /v1/chat/completions.

For routes that require multipart/form-data (parity with vLLM-Omni), JSON bodies
are converted when ``route=`` targets those paths.

Environment:

- LLAMACPP_SAGEMAKER_BACKEND_URL: upstream base URL (default http://127.0.0.1:8081)
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections.abc import AsyncIterator

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

logger = logging.getLogger("llamacpp_sagemaker")

BACKEND = os.environ.get("LLAMACPP_SAGEMAKER_BACKEND_URL", "http://127.0.0.1:8081").rstrip("/")

FORM_DATA_ROUTES = frozenset({"/v1/videos", "/v1/videos/sync"})

_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)

_RESP_DROP = frozenset({"transfer-encoding", "content-length", "connection"})


def _parse_route_from_header(raw: str | None) -> str | None:
    if not raw:
        return None
    m = re.search(r"route=(/[^\s,]+)", raw)
    return m.group(1) if m else None


def _parse_route(request: Request) -> str | None:
    h = request.headers
    v = h.get("x-amzn-sagemaker-custom-attributes")
    return _parse_route_from_header(v)


def _build_multipart_body(data: dict, boundary: str) -> bytes:
    parts: list[str] = []
    for key, value in data.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'
        )
    parts.append(f"--{boundary}--\r\n")
    return "".join(parts).encode()


def _default_path_for_invocation(content_type: str, body: bytes) -> str:
    ct = (content_type or "").lower()
    if "json" not in ct:
        return "/v1/chat/completions"
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "/v1/chat/completions"
    if not isinstance(data, dict):
        return "/v1/chat/completions"
    if "messages" in data:
        return "/v1/chat/completions"
    if "prompt" in data:
        return "/v1/completions"
    if "input" in data and "model" in data:
        return "/v1/embeddings"
    return "/v1/chat/completions"


def _forward_request_headers(request: Request, body_len: int, content_type: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in request.headers.items():
        lk = key.lower()
        if lk in _HOP_BY_HOP or lk == "x-amzn-sagemaker-custom-attributes":
            continue
        out[key] = value
    out["content-length"] = str(body_len)
    if content_type is not None:
        out["content-type"] = content_type
    return out


def _response_headers_from_httpx(resp: httpx.Response) -> dict[str, str]:
    h: dict[str, str] = {}
    for key, value in resp.headers.items():
        lk = key.lower()
        if lk in _RESP_DROP:
            continue
        h[key] = value
    return h


async def ping(request: Request) -> Response:
    url = f"{BACKEND}/health"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=2.0)) as client:
            r = await client.get(url)
    except httpx.RequestError as e:
        logger.warning("Backend health request failed: %s", e)
        return Response(status_code=503, content=b'{"error":"backend_unavailable"}')
    return Response(
        status_code=r.status_code,
        content=r.content,
        headers=_response_headers_from_httpx(r),
    )


async def invocations(request: Request) -> Response:
    if request.method != "POST":
        return Response(status_code=405, content=b"Method Not Allowed")

    body = await request.body()
    route = _parse_route(request)
    content_type = request.headers.get("content-type")

    if route:
        target = route
        logger.info("Rerouting /invocations -> %s", target)
        ct = (content_type or "").lower()
        if target in FORM_DATA_ROUTES and "json" in ct:
            try:
                data = json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                data = None
            if isinstance(data, dict):
                boundary = uuid.uuid4().hex
                body = _build_multipart_body(data, boundary)
                content_type = f"multipart/form-data; boundary={boundary}"
                logger.info("Converted JSON to form-data for %s", target)
    else:
        target = _default_path_for_invocation(content_type or "", body)
        logger.info("Inferred /invocations -> %s", target)

    url = f"{BACKEND}{target}"
    fwd_headers = _forward_request_headers(request, len(body), content_type)

    timeout = httpx.Timeout(600.0, connect=30.0)
    client = httpx.AsyncClient(timeout=timeout)
    try:
        req = client.build_request("POST", url, headers=fwd_headers, content=body)
        r = await client.send(req, stream=True)
    except httpx.RequestError as e:
        await client.aclose()
        logger.exception("Upstream request failed: %s", e)
        return Response(status_code=502, content=json.dumps({"error": "upstream_error"}).encode())

    async def stream_body() -> AsyncIterator[bytes]:
        try:
            async for chunk in r.aiter_bytes():
                yield chunk
        finally:
            await r.aclose()
            await client.aclose()

    return StreamingResponse(
        stream_body(),
        status_code=r.status_code,
        headers=_response_headers_from_httpx(r),
        media_type=r.headers.get("content-type"),
    )


routes = [
    Route("/ping", ping, methods=["GET"]),
    Route("/invocations", invocations, methods=["POST"]),
]

app = Starlette(routes=routes)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    force=True,
)
