"""Network isolation training tests.

Migrated from SMFrameworksXGBoost3_0-5Tests/src/integration_tests/test_network_isolation.py
"""

from .conftest import data_uri, run_training_job

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

        SageMaker pre-stages sagemaker_submit_directory into the container
        before enabling network isolation. Note: the original Hydra test
        overrides training_parameters entirely for script mode, passing
        only sagemaker_program and sagemaker_submit_directory.
        """
        hp = {
            "sagemaker_program": "abalone.py",
            "sagemaker_submit_directory": data_uri("script_mode/code/abalone.1.2-1.tar.gz"),
        }
        _, duration, desc = run_training_job(
            image_uri=image_uri, role=role, hyperparameters=hp,
            train_s3_key="script_mode/data/train",
            validation_s3_key="script_mode/data/validation",
            content_type="text/libsvm", test_name="netiso-script",
            instance_count=2, enable_network_isolation=True,
        )
        assert desc["TrainingJobStatus"] == "Completed"
