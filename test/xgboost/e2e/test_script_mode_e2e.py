"""Script mode end-to-end tests.

Migrated from SMFrameworksXGBoost3_0-5Tests/src/integration_tests/test_script_mode_e2e.py
"""

import pytest

from .conftest import data_uri, delete_endpoint, deploy_endpoint, run_training_job

SCRIPT_HP = {
    "sagemaker_program": "abalone.py",
    "max_depth": "5",
    "eta": "0.2",
    "gamma": "4",
    "min_child_weight": "6",
    "subsample": "0.7",
    "verbosity": "3",
    "objective": "reg:squarederror",
    "num_round": "50",
}


@pytest.fixture(scope="module")
def script_mode_model(image_uri, role):
    """Train a script-mode model once for all tests in this module."""
    hp = {
        **SCRIPT_HP,
        "sagemaker_submit_directory": data_uri("script_mode/code/abalone.1.2-1.tar.gz"),
    }
    _, _, desc = run_training_job(
        image_uri=image_uri, role=role, hyperparameters=hp,
        train_s3_key="script_mode/data/train",
        validation_s3_key="script_mode/data/validation",
        content_type="text/libsvm", test_name="script-train",
        instance_count=2, volume_size=20, max_run=3600,
    )
    assert desc["TrainingJobStatus"] == "Completed"
    return desc["ModelArtifacts"]["S3ModelArtifacts"]


class TestScriptModeE2E:
    def test_inference_single_model(self, image_uri, role, script_mode_model):
        endpoint_name = None
        try:
            predictor, endpoint_name = deploy_endpoint(
                image_uri=image_uri, role=role,
                model_data=script_mode_model, test_name="script-infer",
                env={
                    "SAGEMAKER_PROGRAM": "abalone.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": data_uri(
                        "script_mode/code/abalone.1.2-1.tar.gz"
                    ),
                },
            )
            predictor.content_type = "text/csv"
            predictor.accept = "text/csv"

            payload = "0.455,0.365,0.095,0.514,0.2245,0.101,0.15,15"
            response = predictor.predict(payload)
            assert response is not None
        finally:
            if endpoint_name:
                delete_endpoint(endpoint_name)
