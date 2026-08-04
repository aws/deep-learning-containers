"""nginx / gunicorn env-var configuration test for TF 2.20 inference DLC.

Deploys with non-default SAGEMAKER_TFS_NGINX_LOGLEVEL /
SAGEMAKER_GUNICORN_{WORKERS,THREADS,LOGLEVEL} and asserts endpoint still
serves predictions — proves template substitution + gunicorn spawn accept
these values without crashing.
"""

from __future__ import annotations

import json
import tempfile

import pytest

from .resources.build_sample_model import build_sample_model
from .resources.helpers import CUSTOM_INFERENCE_PY, upload_tarball


def test_nginx_and_gunicorn_env_tuning(
    sagemaker_session,
    deploy_endpoint,
    unique_name,
    cleanup_endpoint,
):
    """Deploy with non-default nginx/gunicorn env vars — must serve correctly.

    Ships a customer inference.py so `_use_gunicorn=True` in serve.py, else
    requests bypass gunicorn and the tuning vars would be inert.
    """
    with tempfile.TemporaryDirectory(prefix="tf220-nginx-env-") as workdir:
        tar_path = build_sample_model(
            output_dir=workdir,
            multiplier=2.0,
            code_files={"inference.py": CUSTOM_INFERENCE_PY},
        )
        model_data = upload_tarball(
            sagemaker_session,
            tar_path,
            key_prefix=f"tf220-inference-tests/nginx-env/{unique_name('run')}",
        )
        endpoint, endpoint_name, model_name = deploy_endpoint(
            model_data_url=model_data,
            name_prefix="tf220-nginx-env",
            container_env={
                # Small non-default values, fits on ml.c5.xlarge.
                "SAGEMAKER_TFS_NGINX_LOGLEVEL": "warn",
                "SAGEMAKER_GUNICORN_WORKERS": "2",
                "SAGEMAKER_GUNICORN_THREADS": "2",
                "SAGEMAKER_GUNICORN_LOGLEVEL": "warning",
            },
        )
        cleanup_endpoint(endpoint_name, model_name=model_name)

        payload = json.dumps({"instances": [[1.0, 2.0, 3.0]]})
        result = endpoint.invoke(
            body=payload,
            content_type="application/json",
            accept="application/json",
        )
        body = json.loads(result.body.read().decode("utf-8"))

        # Presence of the marker proves the gunicorn worker loaded the
        # customer handler under the tuned env vars.
        assert body.get("_handler_marker") == "input_output_ok", (
            f"gunicorn-served output_handler marker missing — env-tuned "
            f"gunicorn worker did not load customer inference.py. body: {body!r}"
        )

        # Handler prepends 1 marker row => 2 total rows; customer row 2x.
        predictions = body["predictions"]
        assert len(predictions) == 2, (
            f"expected 2 predictions (1 marker + 1 customer), got {len(predictions)}: "
            f"{predictions!r}"
        )
        customer_values = (
            predictions[1]["output"]
            if isinstance(predictions[1], dict) and "output" in predictions[1]
            else predictions[1]
        )
        assert customer_values == pytest.approx([2.0, 4.0, 6.0]), (
            "endpoint under tuned nginx/gunicorn config returned wrong "
            f"predictions — template substitution or worker spawn regressed. "
            f"got {customer_values!r}"
        )
