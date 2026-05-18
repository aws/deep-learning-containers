"""SageMaker routing middleware for vLLM.

Routes /invocations requests based on the X-Amzn-SageMaker-Custom-Attributes
header. Clients specify the target endpoint via route=<path>, e.g.:

  CustomAttributes="route=/v1/audio/transcriptions"

If no route is specified, falls through to vLLM's built-in /invocations
handler (chat/completion/embed).

SageMaker passes ContentType and Body through to the container verbatim,
so clients can send multipart/form-data directly for audio endpoints.

Usage: vllm serve --middleware sagemaker_serve.SageMakerRouteMiddleware
"""

import logging
import re

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger("sagemaker_route")


def _parse_route(headers: list[tuple[bytes, bytes]]) -> str | None:
    for key, value in headers:
        if key.lower() == b"x-amzn-sagemaker-custom-attributes":
            m = re.search(r"route=(/[^\s,]+)", value.decode())
            return m.group(1) if m else None
    return None


class SageMakerRouteMiddleware:
    """ASGI middleware that reroutes /invocations based on CustomAttributes.

    Explicit route via header -> rewrites path to that endpoint.
    No route specified -> falls through to vLLM's built-in /invocations handler.
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

        await self.app(scope, receive, send)
