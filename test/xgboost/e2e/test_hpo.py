"""HPO (Hyperparameter Optimization) tests.

Migrated from SMFrameworksXGBoost3_0-5Tests/src/integration_tests/test_hpo.py
"""

import boto3
from sagemaker.core.parameter import ContinuousParameter, IntegerParameter
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, InputData, OutputDataConfig, StoppingCondition
from sagemaker.train.tuner import HyperparameterTuner
from test_utils import random_suffix_name

from .conftest import E2E_TEST_BUCKET, data_uri, s3_uri

RMSE_METRIC = [{"Name": "validation:rmse", "Regex": r"\[.*\].*#011validation-rmse:([\d.]+)"}]
AUCPR_METRIC = [{"Name": "validation:aucpr", "Regex": r"\[.*\].*#011validation-aucpr:([\d.]+)"}]


def _run_hpo(
    image_uri,
    role,
    hp,
    train_key,
    val_key,
    content_type,
    objective_name,
    objective_type,
    metric_defs,
    test_name,
    instance_type="ml.m5.xlarge",
):
    job_name = random_suffix_name(f"xgb-{test_name}", 32)
    output_path = s3_uri(E2E_TEST_BUCKET, f"e2e-output/{job_name}")

    compute = Compute(
        instance_type=instance_type,
        instance_count=1,
        volume_size_in_gb=10,
    )

    trainer = ModelTrainer(
        training_image=image_uri,
        role=role,
        compute=compute,
        hyperparameters=hp,
        output_data_config=OutputDataConfig(s3_output_path=output_path),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=2700),
    )

    tuner = HyperparameterTuner(
        model_trainer=trainer,
        objective_metric_name=objective_name,
        objective_type=objective_type,
        hyperparameter_ranges={
            "num_round": IntegerParameter(5, 20),
            "eta": ContinuousParameter(0.1, 0.5),
        },
        max_jobs=4,
        max_parallel_jobs=2,
        metric_definitions=metric_defs,
    )

    input_data_config = [
        InputData(channel_name="train", data_source=data_uri(train_key), content_type=content_type),
        InputData(
            channel_name="validation",
            data_source=data_uri(val_key),
            content_type=content_type,
        ),
    ]

    tuner.tune(input_data_config=input_data_config, hyper_parameter_tuning_job_name=job_name)
    tuner.wait()

    sm = boto3.client("sagemaker")
    desc = sm.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=job_name)
    assert desc["HyperParameterTuningJobStatus"] == "Completed"


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
        _run_hpo(
            image_uri,
            role,
            BASE_HP,
            "train",
            "test",
            "text/libsvm",
            "validation:rmse",
            "Minimize",
            RMSE_METRIC,
            "hpo-rmse",
        )

    def test_tuning_aucpr(self, image_uri, role):
        hp = {**BASE_HP, "objective": "binary:hinge"}
        _run_hpo(
            image_uri,
            role,
            hp,
            "csv/binary_train",
            "csv/binary_train",
            "text/csv",
            "validation:aucpr",
            "Maximize",
            AUCPR_METRIC,
            "hpo-aucpr",
        )

    def test_gpu_tuning_rmse(self, image_uri, role):
        hp = {**BASE_HP, "tree_method": "gpu_hist"}
        _run_hpo(
            image_uri,
            role,
            hp,
            "train",
            "test",
            "text/libsvm",
            "validation:rmse",
            "Minimize",
            RMSE_METRIC,
            "hpo-gpu",
            instance_type="ml.g4dn.2xlarge",
        )

    def test_gpu_tuning_aucpr(self, image_uri, role):
        hp = {**BASE_HP, "objective": "binary:hinge", "tree_method": "gpu_hist"}
        _run_hpo(
            image_uri,
            role,
            hp,
            "csv/binary_train",
            "csv/binary_train",
            "text/csv",
            "validation:aucpr",
            "Maximize",
            AUCPR_METRIC,
            "hpo-gpu-auc",
            instance_type="ml.g4dn.2xlarge",
        )
