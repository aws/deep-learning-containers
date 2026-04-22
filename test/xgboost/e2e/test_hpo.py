"""HPO (Hyperparameter Optimization) tests.

Migrated to SageMaker SDK v3 using sagemaker-core HyperParameterTuningJob.
"""

import boto3
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

    # Static hyperparameters (not tuned)
    static_hp = {k: str(v) for k, v in hp.items()}

    sm = boto3.client("sagemaker")
    sm.create_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=job_name,
        HyperParameterTuningJobConfig={
            "Strategy": "Bayesian",
            "HyperParameterTuningJobObjective": {
                "Type": objective_type,
                "MetricName": objective_name,
            },
            "ResourceLimits": {"MaxNumberOfTrainingJobs": 4, "MaxParallelTrainingJobs": 2},
            "ParameterRanges": {
                "IntegerParameterRanges": [
                    {"Name": "num_round", "MinValue": "5", "MaxValue": "20", "ScalingType": "Auto"}
                ],
                "ContinuousParameterRanges": [
                    {"Name": "eta", "MinValue": "0.1", "MaxValue": "0.5", "ScalingType": "Auto"}
                ],
                "CategoricalParameterRanges": [],
            },
        },
        TrainingJobDefinition={
            "StaticHyperParameters": static_hp,
            "AlgorithmSpecification": {
                "TrainingImage": image_uri,
                "TrainingInputMode": "File",
                "MetricDefinitions": [
                    {"Name": m["Name"], "Regex": m["Regex"]} for m in metric_defs
                ],
            },
            "RoleArn": role,
            "InputDataConfig": [
                {
                    "ChannelName": "train",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": data_uri(train_key),
                        }
                    },
                    "ContentType": content_type,
                },
                {
                    "ChannelName": "validation",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": data_uri(val_key),
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                    "ContentType": content_type,
                },
            ],
            "OutputDataConfig": {"S3OutputPath": output_path},
            "ResourceConfig": {
                "InstanceType": instance_type,
                "InstanceCount": 1,
                "VolumeSizeInGB": 10,
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": 2700},
        },
    )

    # Wait for completion
    import time

    for _ in range(90):  # up to 45 minutes
        desc = sm.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=job_name)
        status = desc["HyperParameterTuningJobStatus"]
        if status in ("Completed", "Failed", "Stopped"):
            break
        time.sleep(30)

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
