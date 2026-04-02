"""HPO (Hyperparameter Optimization) tests.

Migrated from SMFrameworksXGBoost3_0-5Tests/src/integration_tests/test_hpo.py
"""

import pytest
import boto3
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from test_utils import random_suffix_name

from .conftest import data_uri, E2E_TEST_BUCKET, s3_uri

BASE_HP = {
    "max_depth": "5",
    "gamma": "4",
    "min_child_weight": "6",
    "subsample": "0.7",
    "verbosity": "3",
    "objective": "reg:squarederror",
}


class TestHPO:
    def test_tuning_rmse(self, image_uri, role):
        job_name = random_suffix_name("xgb-hpo", 32)
        output_path = s3_uri(E2E_TEST_BUCKET, f"e2e-output/{job_name}")

        estimator = Estimator(
            image_uri=image_uri,
            role=role,
            instance_count=1,
            instance_type="ml.m5.xlarge",
            output_path=output_path,
            hyperparameters=BASE_HP,
            volume_size=10,
            max_run=2700,
        )

        tuner = HyperparameterTuner(
            estimator=estimator,
            objective_metric_name="validation:rmse",
            objective_type="Minimize",
            hyperparameter_ranges={
                "num_round": IntegerParameter(5, 20),
                "eta": ContinuousParameter(0.1, 0.5),
            },
            max_jobs=4,
            max_parallel_jobs=2,
        )

        channels = {
            "train": TrainingInput(
                s3_data=data_uri("train"), content_type="text/libsvm",
            ),
            "validation": TrainingInput(
                s3_data=data_uri("test"), content_type="text/libsvm",
                distribution="FullyReplicated",
            ),
        }

        tuner.fit(channels, job_name=job_name)
        tuner.wait()

        sm = boto3.client("sagemaker")
        desc = sm.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=job_name
        )
        assert desc["HyperParameterTuningJobStatus"] == "Completed"
