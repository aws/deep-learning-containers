"""nginx / gunicorn env-var configuration test for TF 2.20 inference DLC.

Customers tune the serving stack via a documented set of env vars that
``serve.py`` and ``nginx.conf.template`` consume:

  * ``SAGEMAKER_TFS_NGINX_LOGLEVEL``    (nginx error_log severity)
  * ``SAGEMAKER_GUNICORN_WORKERS``       (gunicorn worker count)
  * ``SAGEMAKER_GUNICORN_THREADS``       (gunicorn threads per worker)
  * ``SAGEMAKER_GUNICORN_LOGLEVEL``      (gunicorn log severity)

Master TF 2.19 had 3 dedicated nginx-config tests (``test_nginx_config*``)
that spawned the container locally and grepped the generated conf. Since
we can't ``docker exec`` into a SageMaker managed endpoint, the strongest
end-to-end assertion is that the endpoint still deploys and serves
predictions correctly under the tuned config — i.e. the template
substitution and gunicorn spawn logic accepts these values without
crashing.

Covers audit finding G8.
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
    """Deploy with non-default nginx/gunicorn tuning env vars — endpoint
    must reach InService and return correct predictions. A template
    substitution bug or a bad worker/thread combo would surface as a 5xx
    at endpoint deploy time (nginx -t fails, gunicorn refuses to start,
    etc.).

    Ships a customer ``inference.py`` under ``code/`` so the serving stack
    actually runs gunicorn (``_use_gunicorn = True`` in ``serve.py``).
    Without a customer handler the SAGEMAKER_GUNICORN_* env vars would be
    inert — the request would go straight through nginx to TFS and the
    three gunicorn tuning vars in this test would not be exercised.
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
                # Non-default values in a range that the DLC handler must
                # accept. Values chosen small enough to fit on ml.c5.xlarge.
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

        # Gunicorn-served marker — CUSTOM_INFERENCE_PY's output_handler wraps
        # the TFS response with a ``_handler_marker`` key. Its presence
        # proves the gunicorn worker actually loaded the customer handler
        # under the tuned SAGEMAKER_GUNICORN_* env vars (a template
        # substitution or worker-spawn regression would fail before this
        # key ever appears).
        assert body.get("_handler_marker") == "input_output_ok", (
            f"gunicorn-served output_handler marker missing — env-tuned "
            f"gunicorn worker did not load customer inference.py. body: {body!r}"
        )

        # Customer handler prepends 1 marker row → 2 rows returned (marker + 1
        # customer). Multiplier 2 on customer row [1, 2, 3] → [2, 4, 6].
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
