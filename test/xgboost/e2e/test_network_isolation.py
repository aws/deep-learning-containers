"""Network isolation training tests.

Migrated from SMFrameworksXGBoost3_0-5Tests/src/integration_tests/test_network_isolation.py
"""

import boto3
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from test_utils import random_suffix_name

from .conftest import data_uri, E2E_TEST_BUCKET, E2E_DATA_PREFIX, s3_uri

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
        from .conftest import run_training_job
        _, duration, desc = run_training_job(
            image_uri=image_uri, role=role, hyperparameters=BASE_HP,
            train_s3_key="train", validation_s3_key="test",
            content_type="text/libsvm", test_name="netiso-algo",
            enable_network_isolation=True,
        )
        assert desc["TrainingJobStatus"] == "Completed"

    def test_script_mode(self, image_uri, role):
        """Script mode with network isolation.

        Uses entry_point + source_dir so SageMaker delivers the code
        via the platform before the container starts (not via S3 at runtime).
        """
        job_name = random_suffix_name("xgb-netiso-script", 32)
        output_path = s3_uri(E2E_TEST_BUCKET, f"e2e-output/{job_name}")

        estimator = Estimator(
            image_uri=image_uri,
            role=role,
            instance_count=2,
            instance_type="ml.m5.xlarge",
            output_path=output_path,
            hyperparameters=BASE_HP,
            volume_size=10,
            max_run=1800,
            entry_point="abalone.py",
            source_dir=data_uri("script_mode/code/abalone.1.2-1.tar.gz"),
            enable_network_isolation=True,
        )

        channels = {
            "train": TrainingInput(
                s3_data=data_uri("script_mode/data/train"),
                content_type="text/libsvm",
            ),
            "validation": TrainingInput(
                s3_data=data_uri("script_mode/data/validation"),
                content_type="text/libsvm",
                distribution="FullyReplicated",
            ),
        }

        sm = boto3.client("sagemaker")
        start = time.time()
        try:
            estimator.fit(channels, job_name=job_name)
        except Exception:
            try:
                sm.stop_training_job(TrainingJobName=job_name)
            except Exception:
                pass
            raise

        desc = sm.describe_training_job(TrainingJobName=job_name)
        assert desc["TrainingJobStatus"] == "Completed"
