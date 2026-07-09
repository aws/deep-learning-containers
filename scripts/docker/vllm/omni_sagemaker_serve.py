"""SageMaker routing middleware for vLLM-Omni.

Routes /invocations requests based on the X-Amzn-SageMaker-Custom-Attributes
header. Clients specify the target endpoint via route=<path>, e.g.:

  CustomAttributes="route=/v1/audio/speech"

If no route is specified, falls through to vLLM's built-in /invocations
handler (chat/completion/embed).

Some routes (e.g. /v1/videos, /v1/videos/sync) are implemented upstream as
multipart/form-data handlers. Clients may target them either with a
pre-built multipart/form-data body, or — for convenience over SageMaker
InvokeEndpoint — with a plain application/json object. When the resolved
route is one of FORM_DATA_ROUTES and the request arrives as application/json,
this middleware transparently converts the JSON object into a
multipart/form-data body (each top-level key becomes a form field) so the
upstream handler receives the encoding it expects. multipart/form-data bodies
are passed through unchanged.

Usage: vllm serve --omni --middleware omni_sagemaker_serve.SageMakerRouteMiddleware

The middleware is loaded via vLLM's `--middleware` flag, which v0.20.0 wires
through `args.middleware` -> `app.add_middleware()` in
`vllm/entrypoints/openai/api_server.py:build_app`. vllm-omni v0.20.0's
"delegate to upstream entrypoint" rebase (vllm-omni#3082, #3232) preserves
this contract — if a future upstream change replaces the `--middleware` arg or
removes the `args.middleware` loop in `build_app`, this loader will silently
no-op and SageMaker /invocations will return 404 for non-default routes.
"""

import json
import logging
import re
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("omni_sagemaker")

# Routes whose upstream handlers expect multipart/form-data. A JSON body
# targeting one of these is converted to multipart before it reaches vLLM.
FORM_DATA_ROUTES = frozenset({"/v1/videos", "/v1/videos/sync"})


def _parse_route(headers: list[tuple[bytes, bytes]]) -> str | None:
    """Extract route=<path> from SageMaker custom attributes header."""
    for key, value in headers:
        if key.lower() == b"x-amzn-sagemaker-custom-attributes":
            m = re.search(r"route=(/[^\s,]+)", value.decode())
            return m.group(1) if m else None
    return None


def _content_type(headers: list[tuple[bytes, bytes]]) -> str:
    """Return the bare content-type (no parameters), lowercased."""
    for key, value in headers:
        if key.lower() == b"content-type":
            return value.decode().split(";", 1)[0].strip().lower()
    return ""


def _field_value(value: object) -> str:
    """Render a non-null JSON value as a multipart form field string.

    Scalars map to their natural string form; bool maps to "true"/"false";
    nested objects/arrays are re-encoded as JSON. None is handled by the
    caller (_build_multipart drops null keys) and never reaches here.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value)


def _escape_field_name(name: str) -> str:
    """Escape a form-field name for the Content-Disposition header.

    RFC 7578 requires CR/LF and double-quote to be escaped inside name="...".
    Without this a key containing " or a newline would corrupt the header.
    """
    return name.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "%0D").replace("\n", "%0A")


def _build_multipart(fields: dict, boundary: str) -> bytes:
    # JSON null has no faithful form-data representation: a form field is
    # always a present string. Emitting the literal "null" would make the
    # upstream handler see a present-but-bogus value instead of an absent one,
    # bypassing its default-on-missing logic. Dropping the key reproduces what
    # a caller omitting the field would send, which is the closest match.
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{_escape_field_name(k)}"'
        f"\r\n\r\n{_field_value(v)}\r\n"
        for k, v in fields.items()
        if v is not None
    ]
    parts.append(f"--{boundary}--\r\n")
    return "".join(parts).encode()


def _replay(body: bytes) -> Receive:
    """Build a receive callable that yields `body` as a single http.request."""
    sent = False

    async def receive() -> Message:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    return receive


class SageMakerRouteMiddleware:
    """ASGI middleware that reroutes /invocations based on CustomAttributes.

    Explicit route via header -> rewrites path to that endpoint.
    No route specified -> falls through to vLLM's built-in /invocations handler.

    When the resolved route expects multipart/form-data (FORM_DATA_ROUTES) but
    the body arrives as application/json, the JSON object is converted to a
    multipart/form-data body so the upstream handler receives its expected
    encoding.
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

                if (
                    route in FORM_DATA_ROUTES
                    and _content_type(scope.get("headers", [])) == "application/json"
                ):
                    scope, receive = await self._json_to_multipart(scope, receive)

        await self.app(scope, receive, send)

    async def _json_to_multipart(self, scope: Scope, receive: Receive) -> tuple[Scope, Receive]:
        """Drain the JSON body, convert it to multipart, and return a new
        (scope, receive) carrying the rewritten content-type and body.

        On any failure (non-object JSON, parse error) the original body is
        replayed untouched so the upstream handler produces its normal error
        response rather than this middleware masking it.
        """
        body = b""
        more_body = True
        while more_body:
            message = await receive()
            body += message.get("body", b"")
            more_body = message.get("more_body", False)

        try:
            parsed = json.loads(body)
            if not isinstance(parsed, dict):
                raise ValueError("JSON body for a form route must be an object")
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Could not convert JSON body to multipart: %s", exc)
            return scope, _replay(body)

        boundary = uuid.uuid4().hex
        new_body = _build_multipart(parsed, boundary)
        content_type = f"multipart/form-data; boundary={boundary}".encode()

        headers = [
            (k, v)
            for k, v in scope["headers"]
            if k.lower() not in (b"content-type", b"content-length")
        ]
        headers.append((b"content-type", content_type))
        headers.append((b"content-length", str(len(new_body)).encode()))
        scope = dict(scope)
        scope["headers"] = headers

        logger.info("Converted application/json body to multipart/form-data")
        return scope, _replay(new_body)
