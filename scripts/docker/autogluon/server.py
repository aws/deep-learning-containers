"""SageMaker HTTP adapter for AutoGluon inference handlers."""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from types import ModuleType

from flask import Flask, Response, jsonify, request

MODEL_DIR = Path(os.getenv("SAGEMAKER_BASE_DIR", "/opt/ml")) / "model"
CODE_DIR = MODEL_DIR / "code"
PROGRAM = os.getenv("SAGEMAKER_PROGRAM", "inference.py").strip() or "inference.py"
DEFAULT_ACCEPT = (
    os.getenv("SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT", "application/json").strip()
    or "application/json"
)
LOG_LEVEL = os.getenv("SAGEMAKER_CONTAINER_LOG_LEVEL", "INFO").strip() or "INFO"
logging.basicConfig(level=int(LOG_LEVEL) if LOG_LEVEL.isdigit() else LOG_LEVEL.upper())
LOGGER = logging.getLogger("autogluon-serving")


def _load_handler() -> ModuleType:
    """Load the user-provided inference module from the model artifact."""
    program = Path(PROGRAM)
    path = program if program.is_absolute() else CODE_DIR / program
    if not path.is_file():
        raise RuntimeError(f"AutoGluon inference handler not found: {path}")

    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("autogluon_user_handler", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import AutoGluon inference handler: {path}")

    handler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(handler)
    return handler


def _media_type(value: str | None, default: str) -> str:
    value = (value or default).split(",", 1)[0].split(";", 1)[0].strip().lower()
    return default if not value or value == "*/*" else value


def _request_body(body: bytes, content_type: str) -> bytes | str:
    if content_type.startswith("text/") or content_type == "application/json":
        return body.decode("utf-8")
    return body


HANDLER = _load_handler()
if not callable(getattr(HANDLER, "model_fn", None)):
    raise TypeError("AutoGluon inference handler must define model_fn")
if not callable(getattr(HANDLER, "transform_fn", None)):
    raise TypeError("AutoGluon inference handler must define transform_fn")
MODEL = HANDLER.model_fn(str(MODEL_DIR))
app = Flask(__name__)


@app.get("/ping")
def ping() -> Response:
    return Response(status=200)


@app.post("/invocations")
def invocations() -> Response:
    content_type = _media_type(request.headers.get("content-type"), "application/json")
    accept = _media_type(request.headers.get("accept"), DEFAULT_ACCEPT)

    try:
        body = _request_body(request.get_data(cache=False), content_type)
        result = HANDLER.transform_fn(MODEL, body, content_type, accept)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("transform_fn must return (body, content_type)")
        output, output_content_type = result
    except ValueError as error:
        return jsonify(detail=str(error)), 400
    except Exception:
        LOGGER.exception("AutoGluon inference failed")
        return jsonify(detail="inference failed"), 500

    output_content_type = _media_type(output_content_type, accept)
    if isinstance(output, (bytes, str)):
        return Response(response=output, content_type=output_content_type)
    return (
        jsonify(detail=f"handler returned {type(output).__name__} for {output_content_type}"),
        500,
    )
