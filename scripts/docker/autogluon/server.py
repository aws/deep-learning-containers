"""SageMaker HTTP adapter for AutoGluon inference handlers."""

from __future__ import annotations

import importlib.util
import logging
import sys
from types import ModuleType
from typing import Any

from flask import Flask, Response, jsonify, request
from settings import Settings

SETTINGS = Settings.from_environment()
logging.basicConfig(level=SETTINGS.log_level)
LOGGER = logging.getLogger("autogluon-serving")


def _load_handler() -> ModuleType:
    """Load the user-provided inference module from the model artifact."""
    path = SETTINGS.handler_path
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
    if content_type.startswith("text/") or content_type in {
        "application/json",
        "application/jsonl",
        "application/x-autogluon",
    }:
        return body.decode("utf-8")
    return body


class Handler:
    """Validated model and invocation functions for one Gunicorn worker."""

    def __init__(self, module: ModuleType):
        model_fn = getattr(module, "model_fn", None)
        if not callable(model_fn):
            raise TypeError("AutoGluon inference handler must define model_fn")

        transform_fn = getattr(module, "transform_fn", None)
        pipeline = {
            "input_fn": getattr(module, "input_fn", None),
            "predict_fn": getattr(module, "predict_fn", None),
            "output_fn": getattr(module, "output_fn", None),
        }
        if transform_fn is not None and any(function is not None for function in pipeline.values()):
            raise RuntimeError(
                "transform_fn cannot be combined with input_fn, predict_fn, or output_fn"
            )
        if transform_fn is None:
            missing = [name for name, function in pipeline.items() if not callable(function)]
            if missing:
                raise RuntimeError(
                    "AutoGluon inference handler must define transform_fn or all of "
                    f"input_fn, predict_fn, and output_fn; missing {', '.join(missing)}"
                )
            self._transform_fn = None
            self._input_fn = pipeline["input_fn"]
            self._predict_fn = pipeline["predict_fn"]
            self._output_fn = pipeline["output_fn"]
        elif not callable(transform_fn):
            raise RuntimeError("transform_fn must be callable")
        else:
            self._transform_fn = transform_fn
            self._input_fn = None
            self._predict_fn = None
            self._output_fn = None

        self._model = model_fn(str(SETTINGS.model_dir))

    def transform(
        self,
        body: bytes | str,
        content_type: str,
        accept: str,
    ) -> tuple[Any, str]:
        if self._transform_fn is not None:
            result = self._transform_fn(self._model, body, content_type, accept)
        else:
            input_data = self._input_fn(body, content_type)
            prediction = self._predict_fn(input_data, self._model)
            result = self._output_fn(prediction, accept)

        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, accept


HANDLER = Handler(_load_handler())
app = Flask(__name__)


@app.get("/ping")
def ping() -> Response:
    return Response(status=200)


@app.post("/invocations")
def invocations() -> Response:
    content_type = _media_type(request.headers.get("content-type"), "application/json")
    accept = _media_type(request.headers.get("accept"), SETTINGS.default_accept)

    try:
        body = _request_body(request.get_data(cache=False), content_type)
        output, output_content_type = HANDLER.transform(body, content_type, accept)
    except ValueError as error:
        return jsonify(detail=str(error)), 400
    except Exception:
        LOGGER.exception("AutoGluon inference failed")
        return jsonify(detail="inference failed"), 500

    output_content_type = _media_type(output_content_type, accept)
    if isinstance(output, bytearray):
        output = bytes(output)
    if isinstance(output, memoryview):
        output = output.tobytes()
    if isinstance(output, (bytes, str)):
        return Response(response=output, content_type=output_content_type)
    return (
        jsonify(detail=f"handler returned {type(output).__name__} for {output_content_type}"),
        500,
    )
