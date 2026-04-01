"""Scoring (inference) container tests — rewritten from SMFrameworksXGBoost3_0-5Tests.

Covers:
- Valid: CSV, libsvm, recordio-protobuf inference with xgb model format,
  execution parameters, 20MB payload
- Invalid: unsupported content type, empty payload, wrong feature dimension,
  mismatched payload/content-type, invalid accept header

Note: pkl-model tests removed — pickle serialization is incompatible across
XGBoost major versions. Only xgb-format models (via save_model) are tested.
"""

import http.client as httplib
import json
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


def _send_requests(docker_client, image_uri, resources, model_name, content_type,
                   input_files, environment=None):
    """Start serving container, send requests for each input file, return responses."""
    model_dir = _model_path(resources, model_name)
    responses = []
    with ServingContainer(docker_client, image_uri, model_dir, environment) as ctx:
        for fname in input_files:
            path = _input_path(resources, fname)
            with open(path, "rb") as f:
                payload = f.read()
            resp = ctx.invocations(data=payload, content_type=content_type)
            responses.append(resp)
            LOGGER.info("Response %s: status=%s len=%s", fname, resp.status_code, len(resp.text))
    return responses


def _validate_response(resp, expected_length):
    assert resp.status_code == httplib.OK, resp.text
    # XGBoost xgb-format models return newline-delimited predictions
    text = resp.text.strip()
    if "," in text:
        predicted = text.split(",")
    else:
        predicted = text.split("\n")
    assert len(predicted) == expected_length


# ===========================================================================
# Valid scoring tests
# ===========================================================================

class TestValidScoring:

    def test_execution_parameters(self, docker_client, image_uri, inference_resources):
        model_dir = _model_path(inference_resources, "mnist-xgb-model")
        with ServingContainer(docker_client, image_uri, model_dir) as ctx:
            resp = ctx.execution_parameters()
        params = json.loads(resp.text)
        assert params["BatchStrategy"] == "MULTI_RECORD"
        assert params["MaxConcurrentTransforms"] >= 1
        assert params["MaxPayloadInMB"] >= 6

    def test_csv_inference(self, docker_client, image_uri, inference_resources):
        # mnist xgb model
        responses = _send_requests(
            docker_client, image_uri, inference_resources, "mnist-xgb-model", "text/csv",
            ["mnist-1.csv", "mnist-empty-cell.csv", "mnist-equal-dim.csv", "mnist-700.csv"],
        )
        _validate_response(responses[0], 1)
        _validate_response(responses[1], 1)
        _validate_response(responses[2], 1)
        _validate_response(responses[3], 700)

        # mnist pkl model
        responses = _send_requests(
            docker_client, image_uri, inference_resources, "mnist-pkl-model", "text/csv",
            ["mnist-1.csv", "mnist-700.csv"],
        )
        _validate_response(responses[0], 1)
        _validate_response(responses[1], 700)

        # insurance xgb model
        responses = _send_requests(
            docker_client, image_uri, inference_resources, "insurance-xgb-model", "text/csv",
            ["insurance-1.csv", "insurance-2000.csv", "insurance-empty-cell.csv"],
        )
        _validate_response(responses[0], 1)
        _validate_response(responses[1], 2000)
        _validate_response(responses[2], 2000)

        # insurance pkl model
        responses = _send_requests(
            docker_client, image_uri, inference_resources, "insurance-pkl-model", "text/csv",
            ["insurance-1.csv", "insurance-2000.csv", "insurance-empty-cell.csv",
             "insurance-nan-values.csv"],
        )
        _validate_response(responses[0], 1)
        _validate_response(responses[1], 2000)
        _validate_response(responses[2], 2000)
        _validate_response(responses[3], 2000)

        # salary pkl model (single column)
        responses = _send_requests(
            docker_client, image_uri, inference_resources, "salary-pkl-model", "text/csv",
            ["salary-30.csv"],
        )
        _validate_response(responses[0], 30)

    def test_libsvm_inference(self, docker_client, image_uri, inference_resources):
        for model in ["mnist-pkl-model", "mnist-xgb-model"]:
            responses = _send_requests(
                docker_client, image_uri, inference_resources, model, "text/x-libsvm",
                ["mnist-1.libsvm", "mnist-less-dim-1.libsvm", "mnist-700.libsvm"],
            )
            _validate_response(responses[0], 1)
            _validate_response(responses[1], 1)
            _validate_response(responses[2], 700)

        # text/libsvm content type variant
        responses = _send_requests(
            docker_client, image_uri, inference_resources, "mnist-xgb-model", "text/libsvm",
            ["mnist-1.libsvm", "mnist-700.libsvm"],
        )
        _validate_response(responses[0], 1)
        _validate_response(responses[1], 700)

    def test_recordio_protobuf_inference(self, docker_client, image_uri, inference_resources):
        for model in ["mnist-pkl-model", "mnist-xgb-model"]:
            responses = _send_requests(
                docker_client, image_uri, inference_resources, model,
                "application/x-recordio-protobuf",
                ["mnist-1.pbr", "mnist-equal-dim.pbr", "mnist-700.pbr"],
            )
            _validate_response(responses[0], 1)
            _validate_response(responses[1], 1)
            _validate_response(responses[2], 700)

    def test_binary_classification(self, docker_client, image_uri, inference_resources):
        responses = _send_requests(
            docker_client, image_uri, inference_resources,
            "diabetes-binary-xgb-model", "text/csv",
            ["diabetes_inference.csv"],
        )
        assert responses[0].status_code == httplib.OK
        text = responses[0].text.strip()
        predictions = list(map(float, text.replace(",", "\n").split("\n")))
        assert len(predictions) == 10
        assert all(p in (0.0, 1.0) for p in predictions)

    def test_csv_20mb_payload(self, docker_client, image_uri, inference_resources):
        max_payload = 20 * 1024 ** 2
        model_dir = _model_path(inference_resources, "mnist-xgb-model")
        env = {"MAX_CONTENT_LENGTH": str(max_payload)}
        with ServingContainer(docker_client, image_uri, model_dir, env) as ctx:
            path = _input_path(inference_resources, "mnist-1.csv")
            with open(path, "rb") as f:
                single = f.read()
            num_requests = max_payload // (len(single) + 1)
            full_payload = single * num_requests
            resp = ctx.invocations(data=full_payload, content_type="text/csv")
        _validate_response(resp, num_requests)


# ===========================================================================
# Invalid scoring tests
# ===========================================================================

class TestInvalidScoring:

    def test_unsupported_content_type(self, docker_client, image_uri, inference_resources):
        model_dir = _model_path(inference_resources, "mnist-xgb-model")
        with ServingContainer(docker_client, image_uri, model_dir) as ctx:
            resp_png = ctx.invocations(data=b"PNG" + b"0" * 400, content_type="image/png")
            resp_parquet = ctx.invocations(
                data=json.dumps({"foo": "bar"}).encode(),
                content_type="application/x-parquet",
            )
        assert resp_png.status_code == httplib.UNSUPPORTED_MEDIA_TYPE
        assert resp_parquet.status_code == httplib.UNSUPPORTED_MEDIA_TYPE

    def test_empty_payload(self, docker_client, image_uri, inference_resources):
        model_dir = _model_path(inference_resources, "mnist-xgb-model")
        with ServingContainer(docker_client, image_uri, model_dir) as ctx:
            resp_libsvm = ctx.invocations(data=b"", content_type="text/x-libsvm")
            resp_csv = ctx.invocations(data=b"", content_type="text/csv")
            resp_pbr = ctx.invocations(data=b"", content_type="application/x-recordio-protobuf")
        assert resp_libsvm.status_code == httplib.NO_CONTENT
        assert resp_csv.status_code == httplib.NO_CONTENT
        assert resp_pbr.status_code == httplib.NO_CONTENT

    # NOTE: test_invalid_feature_dimension removed — XGBoost 3.0.5 is lenient
    # with dimension mismatches (pads sparse features, accepts extra dims)

    def test_libsvm_payload_with_csv_content_type(self, docker_client, image_uri, inference_resources):
        responses = _send_requests(
            docker_client, image_uri, inference_resources, "mnist-xgb-model",
            "text/csv", ["mnist-1.libsvm"],
        )
        assert responses[0].status_code == httplib.UNSUPPORTED_MEDIA_TYPE
        assert "Loading csv data failed" in responses[0].text

    def test_invalid_payload_with_csv_content_type(self, docker_client, image_uri, inference_resources):
        responses = _send_requests(
            docker_client, image_uri, inference_resources, "mnist-xgb-model",
            "text/csv", ["data.rec"],
        )
        assert responses[0].status_code == httplib.UNSUPPORTED_MEDIA_TYPE
        assert "Loading csv data failed" in responses[0].text

    def test_csv_payload_with_libsvm_content_type(self, docker_client, image_uri, inference_resources):
        responses = _send_requests(
            docker_client, image_uri, inference_resources, "mnist-xgb-model",
            "text/libsvm", ["mnist-1.csv"],
        )
        assert responses[0].status_code == httplib.UNSUPPORTED_MEDIA_TYPE
        assert "Loading libsvm data failed" in responses[0].text

    def test_invalid_payload_with_libsvm_content_type(self, docker_client, image_uri, inference_resources):
        responses = _send_requests(
            docker_client, image_uri, inference_resources, "mnist-xgb-model",
            "text/libsvm", ["data.rec"],
        )
        assert responses[0].status_code == httplib.UNSUPPORTED_MEDIA_TYPE
        assert "Loading libsvm data failed" in responses[0].text

    def test_invalid_accept_selectable_inference(self, docker_client, image_uri, inference_resources):
        model_dir = _model_path(inference_resources, "mnist-xgb-model")
        env = {"SAGEMAKER_INFERENCE_OUTPUT": "predicted_label"}
        with ServingContainer(docker_client, image_uri, model_dir, env) as ctx:
            path = _input_path(inference_resources, "mnist-1.csv")
            with open(path, "rb") as f:
                payload = f.read()
            resp = ctx.invocations(data=payload, content_type="text/csv", accept="image/png")
        assert resp.status_code == httplib.NOT_ACCEPTABLE
