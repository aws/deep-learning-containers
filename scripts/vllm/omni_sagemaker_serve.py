"""SageMaker routing middleware for vLLM-Omni.

Routes /invocations requests based on the X-Amzn-SageMaker-Custom-Attributes
header. Clients specify the target endpoint via route=<path>, e.g.:

  CustomAttributes="route=/v1/audio/speech"

If no route is specified, falls through to vLLM's built-in /invocations
handler (chat/completion/embed).

For routes that require multipart/form-data (e.g. /v1/videos), JSON payloads
are automatically converted to form data since SageMaker async inference only
supports JSON/binary payloads.

Usage: vllm serve --omni --middleware omni_sagemaker_serve.SageMakerRouteMiddleware
"""

import json
import logging
import re
import uuid

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger("omni_sagemaker")

# Routes that require multipart/form-data
FORM_DATA_ROUTES = {"/v1/videos"}


def _parse_route(headers: list[tuple[bytes, bytes]]) -> str | None:
    """Extract route=<path> from SageMaker custom attributes header."""
    for key, value in headers:
        if key.lower() == b"x-amzn-sagemaker-custom-attributes":
            m = re.search(r"route=(/[^\s,]+)", value.decode())
            return m.group(1) if m else None
    return None


def _get_content_type(headers: list[tuple[bytes, bytes]]) -> str:
    for key, value in headers:
        if key.lower() == b"content-type":
            return value.decode().lower()
    return ""


def _build_multipart_body(data: dict, boundary: str) -> bytes:
    """Build a multipart/form-data body from a dict."""
    parts = []
    for key, value in data.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
            f"{value}\r\n"
        )
    parts.append(f"--{boundary}--\r\n")
    return "".join(parts).encode()


class SageMakerRouteMiddleware:
    """ASGI middleware that reroutes /invocations based on CustomAttributes.

    Explicit route via header -> rewrites path to that endpoint.
    No route specified -> falls through to vLLM's built-in /invocations handler.
    For form-data routes with JSON input, converts JSON to multipart/form-data.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"] == "/invocations":
            route = _parse_route(scope.get("headers", []))
            if route:
                logger.info("Rerouting /invocations -> %s", route)
                scope = dict(scope)
                scope["path"] = route
                scope["raw_path"] = route.encode()

                # Convert JSON to form-data for routes that require it
                content_type = _get_content_type(scope.get("headers", []))
                if route in FORM_DATA_ROUTES and "json" in content_type:
                    body = await _read_body(receive)
                    try:
                        data = json.loads(body)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        data = None

                    if isinstance(data, dict):
                        boundary = uuid.uuid4().hex
                        new_body = _build_multipart_body(data, boundary)
                        new_ct = f"multipart/form-data; boundary={boundary}"
                        scope["headers"] = [
                            (k, v) if k.lower() != b"content-type" else (k, new_ct.encode())
                            for k, v in scope["headers"]
                        ]
                        receive = _make_receive(new_body)
                        logger.info("Converted JSON to form-data for %s", route)

        await self.app(scope, receive, send)


async def _read_body(receive: Receive) -> bytes:
    """Read the full request body from ASGI receive."""
    chunks = []
    while True:
        message = await receive()
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            break
    return b"".join(chunks)


def _make_receive(body: bytes) -> Receive:
    """Create a new ASGI receive callable that yields the given body."""
    sent = False

    async def receive() -> dict:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    return receive
