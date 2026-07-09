"""Online inference test — deploy an endpoint with a no-op echo model,
send a JSON payload, verify a response comes back.
"""

import json

import pytest

from .conftest import data_uri, delete_endpoint, deploy_endpoint, predict_and_log

ECHO_MODEL_KEY = "model/empty.tar.gz"
ECHO_CODE_KEY = "code/echo-2.4.10.tar.gz"


def _payload(feature_dim=100):
    return json.dumps({"instances": [{"data": {"features": {"values": list(range(feature_dim))}}}]})


@pytest.fixture(scope="module")
def echo_env():
    return {
        "SAGEMAKER_PROGRAM": "echo.py",
        "SAGEMAKER_SUBMIT_DIRECTORY": data_uri(ECHO_CODE_KEY),
    }


class TestInference:
    def test_json_inference(self, image_uri, role, echo_env):
        endpoint_name = None
        try:
            predictor, endpoint_name = deploy_endpoint(
                image_uri=image_uri,
                role=role,
                model_data=data_uri(ECHO_MODEL_KEY),
                env=echo_env,
                test_name="inference-solo",
            )
            predictor.content_type = "application/json"
            predictor.accept = "application/json"
            response = predict_and_log(predictor, _payload())
            assert response is not None
        finally:
            if endpoint_name:
                delete_endpoint(endpoint_name)
