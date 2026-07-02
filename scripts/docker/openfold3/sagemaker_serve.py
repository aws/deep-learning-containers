"""Flask serving layer for SageMaker inference container.

Loaded by gunicorn from the `serve` script. The model is loaded ONCE at
worker startup (module import time) so /ping is a real liveness check, not
a model loader. SageMaker won't route traffic until /ping returns 200, so
this also means the first /invocations doesn't pay startup cost.
"""

import json
import logging
import os
import traceback

from flask import Flask, Response, request
from sagemaker_handler import input_fn, model_fn, output_fn, predict_fn

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s", level=logging.INFO)
logger = logging.getLogger("openfold3-serve")

app = Flask(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/ml/model")
logger.info("Loading model at worker startup from %s", MODEL_DIR)
model = model_fn(MODEL_DIR)
logger.info("Model load complete; ready to serve")


@app.route("/ping", methods=["GET"])
def ping():
    return Response(status=200)


@app.route("/invocations", methods=["POST"])
def invocations():
    try:
        content_type = request.content_type or "application/json"
        data = input_fn(request.data, content_type)
        result = predict_fn(data, model)
        output, accept = output_fn(result, "application/json")
        return Response(response=output, status=200, mimetype=accept)
    except Exception as e:
        logger.exception("Invocation failed")
        body = json.dumps(
            {
                "status": "error",
                "error": str(e),
                "trace": traceback.format_exc(limit=10),
            }
        )
        return Response(response=body, status=500, mimetype="application/json")


if __name__ == "__main__":
    # Dev only. In the container, gunicorn imports this module.
    app.run(host="0.0.0.0", port=8080)
