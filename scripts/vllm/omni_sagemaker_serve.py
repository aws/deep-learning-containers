"""SageMaker serving proxy for vLLM-Omni.

Sits on port 8080 (SageMaker's expected port), proxies to vllm-omni on
port 8081. Routes /invocations to the correct vllm-omni endpoint using:

  1. X-Amzn-SageMaker-Custom-Attributes header with route=<path>
  2. Payload inspection as fallback:
     - has "input", no "messages" -> /v1/audio/speech
     - has "messages"             -> /v1/chat/completions
     - has "prompt"               -> /v1/completions
"""

import json
import logging
import re

import httpx
from fastapi import FastAPI, Request, Response

logger = logging.getLogger("omni_serve")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BACKEND = "http://127.0.0.1:8081"
app = FastAPI()


def _parse_route(custom_attrs: str | None) -> str | None:
    """Extract route=<path> from SageMaker custom attributes header."""
    if not custom_attrs:
        return None
    m = re.search(r"route=(/\S+)", custom_attrs)
    return m.group(1) if m else None


def _infer_route(data: dict) -> str:
    """Infer the target endpoint from payload content."""
    if "input" in data and "messages" not in data:
        return "/v1/audio/speech"
    if "messages" in data:
        return "/v1/chat/completions"
    if "prompt" in data:
        return "/v1/completions"
    return "/v1/chat/completions"


@app.get("/ping")
async def ping():
    """SageMaker health check — proxy to vllm-omni /health."""
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{BACKEND}/health", timeout=5)
            return Response(status_code=r.status_code)
        except httpx.ConnectError:
            return Response(status_code=503)


@app.post("/invocations")
async def invocations(request: Request):
    """Route /invocations to the correct vllm-omni endpoint."""
    body = await request.body()

    # 1. Explicit route from custom attributes header
    custom_attrs = request.headers.get("X-Amzn-SageMaker-Custom-Attributes")
    path = _parse_route(custom_attrs)

    # 2. Fallback: infer from payload
    if not path:
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return Response(content='{"error": "invalid JSON"}', status_code=400,
                            media_type="application/json")
        path = _infer_route(data)

    logger.info("Routing /invocations -> %s", path)

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{BACKEND}{path}",
            content=body,
            headers={"Content-Type": "application/json"},
            timeout=300,
        )

    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
