"""Selectable content E2E tests — multiclass training + inference with CSV/JSON/JSONLINES.

Migrated from SMFrameworksXGBoost3_0-5Tests/src/integration_tests/test_e2e_selectable.py
"""

import json
import pytest

from .conftest import delete_endpoint, deploy_endpoint, run_training_job

SELECTABLE_HP = {
    "max_depth": "3",
    "num_round": "50",
    "subsample": "1",
    "gamma": "0",
    "min_child_weight": "1",
    "verbosity": "3",
    "objective": "multi:softprob",
    "num_class": "3",
}

INFERENCE_PAYLOAD = "1,5.1,3.5,1.4,0.2\n60,5.2,2.7,3.9,1.4\n113,6.8,3,5.5,2.1"


@pytest.fixture(scope="module")
def selectable_model(image_uri, role):
    """Train a multiclass model on iris dataset."""
    _, _, desc = run_training_job(
        image_uri=image_uri, role=role, hyperparameters=SELECTABLE_HP,
        train_s3_key="iris/train", validation_s3_key="iris/test",
        content_type="text/csv", test_name="select-train",
    )
    assert desc["TrainingJobStatus"] == "Completed"
    return desc["ModelArtifacts"]["S3ModelArtifacts"]


class TestSelectableInference:
    def test_csv_accept(self, image_uri, role, selectable_model):
        endpoint_name = None
        try:
            predictor, endpoint_name = deploy_endpoint(
                image_uri=image_uri, role=role,
                model_data=selectable_model, test_name="select-csv",
                env={"SAGEMAKER_INFERENCE_OUTPUT": "predicted_label,labels"},
            )
            predictor.content_type = "text/csv"
            predictor.accept = "text/csv"
            response = predictor.predict(INFERENCE_PAYLOAD)
            assert response is not None
        finally:
            if endpoint_name:
                delete_endpoint(endpoint_name)

    def test_json_accept(self, image_uri, role, selectable_model):
        endpoint_name = None
        try:
            predictor, endpoint_name = deploy_endpoint(
                image_uri=image_uri, role=role,
                model_data=selectable_model, test_name="select-json",
                env={"SAGEMAKER_INFERENCE_OUTPUT": "labels,probabilities"},
            )
            predictor.content_type = "text/csv"
            predictor.accept = "application/json"
            response = predictor.predict(INFERENCE_PAYLOAD)
            result = json.loads(response)
            assert "predictions" in result
            assert len(result["predictions"]) == 3
        finally:
            if endpoint_name:
                delete_endpoint(endpoint_name)

    def test_jsonlines_accept(self, image_uri, role, selectable_model):
        endpoint_name = None
        try:
            predictor, endpoint_name = deploy_endpoint(
                image_uri=image_uri, role=role,
                model_data=selectable_model, test_name="select-jl",
                env={"SAGEMAKER_INFERENCE_OUTPUT": "predicted_label,probability"},
            )
            predictor.content_type = "text/csv"
            predictor.accept = "application/jsonlines"
            response = predictor.predict(INFERENCE_PAYLOAD)
            lines = response.decode().strip().splitlines() if isinstance(response, bytes) else response.strip().splitlines()
            assert len(lines) == 3
            for line in lines:
                parsed = json.loads(line)
                assert "predicted_label" in parsed
        finally:
            if endpoint_name:
                delete_endpoint(endpoint_name)

    def test_csv_nans_misconfigured_keys(self, image_uri, role, selectable_model):
        endpoint_name = None
        try:
            predictor, endpoint_name = deploy_endpoint(
                image_uri=image_uri, role=role,
                model_data=selectable_model, test_name="select-nan",
                env={"SAGEMAKER_INFERENCE_OUTPUT": "foo,predicted_label,predicted_score,porbabilitise"},
            )
            predictor.content_type = "text/csv"
            predictor.accept = "text/csv"
            response = predictor.predict(INFERENCE_PAYLOAD)
            assert response is not None
        finally:
            if endpoint_name:
                delete_endpoint(endpoint_name)
