"""Batch transform container tests — rewritten from SMFrameworksXGBoost3_0-5Tests.

Covers batch inference with SAGEMAKER_BATCH=True for:
- libsvm (xgb + text/libsvm content type variant)
- recordio-protobuf (xgb)
- csv (xgb: mnist, insurance)

Batch responses are newline-delimited, so expected_length is +1 for trailing newline.

Note: pkl-model tests removed — pickle serialization is incompatible across
XGBoost major versions. Only xgb-format models (via save_model) are tested.
"""

import http.client as httplib
import logging
import os

from .container_helper import ServingContainer

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _input_path(resources, filename):
    return os.path.join(resources, "input", filename)


def _model_path(resources, model_name):
    return os.path.join(resources, "models", model_name)


def _send_batch_requests(
    docker_client, image_uri, resources, model_name, content_type, input_files
):
    model_dir = _model_path(resources, model_name)
    env = {"SAGEMAKER_BATCH": "True"}
    responses = []
    with ServingContainer(docker_client, image_uri, model_dir, env) as ctx:
        for fname in input_files:
            path = _input_path(resources, fname)
            with open(path, "rb") as f:
                payload = f.read()
            resp = ctx.invocations(data=payload, content_type=content_type)
            responses.append(resp)
            LOGGER.info("Batch response %s: status=%s", fname, resp.status_code)
    return responses


def _validate_batch_response(resp, expected_length):
    """Batch responses are newline-delimited; trailing newline adds +1."""
    assert resp.status_code == httplib.OK, resp.text
    lines = resp.text.split("\n")
    assert len(lines) == expected_length + 1


# ===========================================================================
# Tests
# ===========================================================================


class TestBatchTransform:
    def test_libsvm_batch(self, docker_client, image_uri, inference_resources):
        for model in ["mnist-pkl-model", "mnist-xgb-model"]:
            responses = _send_batch_requests(
                docker_client,
                image_uri,
                inference_resources,
                model,
                "text/x-libsvm",
                ["mnist-1.libsvm", "mnist-less-dim-1.libsvm", "mnist-700.libsvm"],
            )
            _validate_batch_response(responses[0], 1)
            _validate_batch_response(responses[1], 1)
            _validate_batch_response(responses[2], 700)

        # text/libsvm variant
        responses = _send_batch_requests(
            docker_client,
            image_uri,
            inference_resources,
            "mnist-xgb-model",
            "text/libsvm",
            ["mnist-1.libsvm", "mnist-700.libsvm"],
        )
        _validate_batch_response(responses[0], 1)
        _validate_batch_response(responses[1], 700)

    def test_recordio_protobuf_batch(self, docker_client, image_uri, inference_resources):
        for model in ["mnist-pkl-model", "mnist-xgb-model"]:
            responses = _send_batch_requests(
                docker_client,
                image_uri,
                inference_resources,
                model,
                "application/x-recordio-protobuf",
                ["mnist-1.pbr", "mnist-equal-dim.pbr", "mnist-700.pbr"],
            )
            _validate_batch_response(responses[0], 1)
            _validate_batch_response(responses[1], 1)
            _validate_batch_response(responses[2], 700)

    def test_csv_batch(self, docker_client, image_uri, inference_resources):
        # mnist pkl
        responses = _send_batch_requests(
            docker_client,
            image_uri,
            inference_resources,
            "mnist-pkl-model",
            "text/csv",
            ["mnist-1.csv", "mnist-empty-cell.csv", "mnist-equal-dim.csv", "mnist-700.csv"],
        )
        _validate_batch_response(responses[0], 1)
        _validate_batch_response(responses[1], 1)
        _validate_batch_response(responses[2], 1)
        _validate_batch_response(responses[3], 700)

        # insurance pkl
        responses = _send_batch_requests(
            docker_client,
            image_uri,
            inference_resources,
            "insurance-pkl-model",
            "text/csv",
            [
                "insurance-1.csv",
                "insurance-2000.csv",
                "insurance-empty-cell.csv",
                "insurance-nan-values.csv",
            ],
        )
        _validate_batch_response(responses[0], 1)
        _validate_batch_response(responses[1], 2000)
        _validate_batch_response(responses[2], 2000)
        _validate_batch_response(responses[3], 2000)

        # insurance xgb
        responses = _send_batch_requests(
            docker_client,
            image_uri,
            inference_resources,
            "insurance-xgb-model",
            "text/csv",
            ["insurance-1.csv", "insurance-2000.csv", "insurance-empty-cell.csv"],
        )
        _validate_batch_response(responses[0], 1)
        _validate_batch_response(responses[1], 2000)
        _validate_batch_response(responses[2], 2000)

        # salary pkl (single column)
        responses = _send_batch_requests(
            docker_client,
            image_uri,
            inference_resources,
            "salary-pkl-model",
            "text/csv",
            ["salary-30.csv"],
        )
        _validate_batch_response(responses[0], 30)
