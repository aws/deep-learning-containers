"""SageMaker launch wrapper for the SGLang diffusion (multimodal_gen) server.

SGLang's LLM server (sglang.launch_server) natively exposes the SageMaker
serving contract — GET /ping and POST /invocations — but the separate
diffusion server (sglang.multimodal_gen) does not: its FastAPI app only serves
a Vertex AI route and the OpenAI-images routes. As a result a FLUX.2 (or other
diffusion) SageMaker endpoint fails the /ping health check.

This wrapper adds the two SageMaker routes to the diffusion server without
forking it: it monkeypatches multimodal_gen's create_app() to register
GET /ping (200) and POST /invocations (delegates to the same handler as
POST /v1/images/generations), then hands off to the real launch_server() so
the GPU workers/scheduler are bootstrapped exactly as usual.

TODO(remove-when-upstream): delete this wrapper and point the entrypoint back
at `python3 -m sglang.multimodal_gen.runtime.launch_server` once upstream
SGLang adds /ping + /invocations to multimodal_gen's create_app() (mirroring
srt/entrypoints/http_server.py) and the DLC image is bumped to that version.
"""

import sys

from fastapi import Request, Response

from sglang.multimodal_gen.runtime import launch_server as _launch
from sglang.multimodal_gen.runtime.entrypoints.http_server import create_app as _create_app
from sglang.multimodal_gen.runtime.entrypoints.openai.image_api import generations
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
    ImageResponse,
)


async def _sagemaker_ping() -> Response:
    """SageMaker startup health probe."""
    return Response(status_code=200)


async def _sagemaker_invocations(
    request: ImageGenerationsRequest, raw_request: Request
) -> ImageResponse:
    """SageMaker inference route — same code path as POST /v1/images/generations."""
    return await generations(request, raw_request)


def _create_app_with_sagemaker_routes(server_args):
    app = _create_app(server_args)
    app.add_api_route("/ping", _sagemaker_ping, methods=["GET"])
    app.add_api_route(
        "/invocations",
        _sagemaker_invocations,
        methods=["POST"],
        response_model=ImageResponse,
    )
    return app


def main():
    # Patch the name launch_server resolves at call time so the real launch
    # flow (worker/scheduler bootstrap + uvicorn.run) builds our augmented app.
    _launch.create_app = _create_app_with_sagemaker_routes

    server_args = _launch.prepare_server_args(sys.argv[1:])
    _launch.launch_server(server_args)


if __name__ == "__main__":
    main()
