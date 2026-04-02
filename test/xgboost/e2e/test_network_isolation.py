"""Network isolation training tests.

Migrated from SMFrameworksXGBoost3_0-5Tests/src/integration_tests/test_network_isolation.py
"""

import pytest

from .conftest import run_training_job

BASE_HP = {
    "max_depth": "5",
    "eta": "0.2",
    "gamma": "4",
    "min_child_weight": "6",
    "subsample": "0.7",
    "verbosity": "3",
    "objective": "reg:squarederror",
    "num_round": "1",
}


class TestNetworkIsolation:
    def test_algo_mode(self, image_uri, role):
        _, duration, desc = run_training_job(
            image_uri=image_uri, role=role, hyperparameters=BASE_HP,
            train_s3_key="train", validation_s3_key="test",
            content_type="text/libsvm", test_name="netiso-algo",
            enable_network_isolation=True,
        )
        assert desc["TrainingJobStatus"] == "Completed"

    def test_script_mode(self, image_uri, role):
        """Script mode with network isolation.

        Skipped: script mode requires S3 access at runtime to download
        and extract sagemaker_submit_directory, which network isolation blocks.
        """
        pytest.skip("Script mode requires S3 access incompatible with network isolation")
