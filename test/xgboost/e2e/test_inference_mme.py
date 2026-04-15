"""Multi-model endpoint inference tests.

Migrated from SMFrameworksXGBoost3_0-5Tests/src/integration_tests/test_inference_mme.py
"""

import pytest

from .conftest import delete_endpoint, deploy_endpoint, run_training_job

TRAIN_HP = {
    "max_depth": "3",
    "num_round": "50",
    "subsample": "1",
    "gamma": "0",
    "min_child_weight": "1",
    "verbosity": "3",
    "objective": "multi:softprob",
    "num_class": "3",
}


@pytest.fixture(scope="module")
def mme_model(image_uri, role):
    """Train an iris model for MME tests."""
    _, _, desc = run_training_job(
        image_uri=image_uri,
        role=role,
        hyperparameters=TRAIN_HP,
        train_s3_key="iris/train",
        validation_s3_key="iris/test",
        content_type="text/csv",
        test_name="mme-train",
    )
    assert desc["TrainingJobStatus"] == "Completed"
    return desc["ModelArtifacts"]["S3ModelArtifacts"]


class TestInferenceMME:
    def test_csv_multimodel(self, image_uri, role, mme_model):
        endpoint_name = None
        try:
            endpoint, endpoint_name = deploy_endpoint(
                image_uri=image_uri,
                role=role,
                model_data=mme_model,
                test_name="mme-csv",
            )

            payload = "5.1,3.5,1.4,0.2\n6.8,3.0,5.5,2.1"
            response = endpoint.invoke_endpoint(
                body=payload,
                content_type="text/csv",
                accept="text/csv",
            )
            assert response is not None
        finally:
            if endpoint_name:
                delete_endpoint(endpoint_name)
