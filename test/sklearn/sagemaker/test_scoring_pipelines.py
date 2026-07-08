"""Inference-pipeline endpoint tests — deploy sklearn + linear-learner in an
inference pipeline (both directions), verify a JSON payload round-trips.
"""

import json

import pytest

from .conftest import (
    E2E_TEST_BUCKET,
    data_uri,
    delete_endpoint,
    deploy_inference_pipeline,
    predict_and_log,
    s3_uri,
)

# us-west-2 SageMaker 1P algo ECR account for linear-learner
LINEAR_LEARNER_IMAGE = "174872318107.dkr.ecr.us-west-2.amazonaws.com/linear-learner:latest"

SKLEARN_MODEL_KEY = "model/empty.tar.gz"
SKLEARN_CODE_KEY = "code/echo-2.4.10.tar.gz"
LINEAR_LEARNER_MODEL = s3_uri(E2E_TEST_BUCKET, "input/linearlearner/a9a/model.tar.gz")


def _linear_learner_payload():
    """JSON payload matching linear-learner's expected 123-feature input dim."""
    return json.dumps({"instances": [{"data": {"features": {"values": list(range(123))}}}]})


@pytest.fixture(scope="module")
def sklearn_container(image_uri):
    return (
        image_uri,
        data_uri(SKLEARN_MODEL_KEY),
        {
            "SAGEMAKER_PROGRAM": "echo.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": data_uri(SKLEARN_CODE_KEY),
        },
    )


@pytest.fixture(scope="module")
def algo_container():
    return (LINEAR_LEARNER_IMAGE, LINEAR_LEARNER_MODEL, {})


class TestScoringPipelines:
    def test_sklearn_then_algo(self, role, sklearn_container, algo_container):
        endpoint_name = None
        try:
            predictor, endpoint_name = deploy_inference_pipeline(
                models=[sklearn_container, algo_container],
                role=role,
                test_name="pipe-skl-algo",
            )
            predictor.content_type = "application/json"
            predictor.accept = "application/json"
            response = predict_and_log(predictor, _linear_learner_payload())
            assert response is not None
        finally:
            if endpoint_name:
                delete_endpoint(endpoint_name)

    def test_algo_then_sklearn(self, role, sklearn_container, algo_container):
        endpoint_name = None
        try:
            predictor, endpoint_name = deploy_inference_pipeline(
                models=[algo_container, sklearn_container],
                role=role,
                test_name="pipe-algo-skl",
            )
            predictor.content_type = "application/json"
            predictor.accept = "application/json"
            response = predict_and_log(predictor, _linear_learner_payload())
            assert response is not None
        finally:
            if endpoint_name:
                delete_endpoint(endpoint_name)
