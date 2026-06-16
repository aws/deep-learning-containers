"""SageMaker hyperparameter-tuning integration test for the TF DLC.

A `HyperparameterTuner` is constructed around a `ModelTrainer` and
`.tune(...)` is invoked. SDK v3 API.

CPU-only: the feature under test is the SageMaker HPO control plane,
not anything CUDA-specific."""

import os

import boto3
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.parameter import IntegerParameter
from sagemaker.core.training.configs import Compute, InputData, SourceCode
from sagemaker.train import ModelTrainer
from sagemaker.train.tuner import HyperparameterTuner
from test_utils import random_suffix_name

RESOURCE_DIR = os.path.join(os.path.dirname(__file__), "resources")
SOURCE_DIR = os.path.join(RESOURCE_DIR, "scripts")
MNIST_DATA_DIR = os.path.join(RESOURCE_DIR, "mnist", "data")
INSTANCE_TYPE = "ml.c5.xlarge"
IMAGE_URI = os.environ["TEST_IMAGE_URI"]
DEFAULT_REGION = "us-west-2"


def test_tuning_model_dir_cpu():
    """Smoke-test SageMaker HPO with the TF DLC.

    Tunes the `epochs` hyperparameter over [1, 2] with max_jobs=2 and
    max_parallel_jobs=2. The objective metric is parsed from training logs
    via the `accuracy` regex; the standard `mnist.py` entry script already
    prints `accuracy: <val>` per epoch via Keras."""
    sagemaker_session = Session(boto3.session.Session(region_name=DEFAULT_REGION))
    inputs_s3 = sagemaker_session.upload_data(path=MNIST_DATA_DIR, key_prefix="scriptmode/mnist")

    source_code = SourceCode(source_dir=SOURCE_DIR, entry_script="mnist.py")
    compute = Compute(instance_type=INSTANCE_TYPE, instance_count=1)

    model_trainer = ModelTrainer(
        training_image=IMAGE_URI,
        source_code=source_code,
        compute=compute,
        role=os.environ.get("SM_ROLE_ARN"),
        base_job_name=random_suffix_name("tf-tuning-cpu", 32),
        hyperparameters={"strategy": "none"},
        distributed=None,
    )

    objective_metric_name = "accuracy"
    tuner = HyperparameterTuner(
        model_trainer=model_trainer,
        objective_metric_name=objective_metric_name,
        hyperparameter_ranges={"epochs": IntegerParameter(1, 2)},
        metric_definitions=[{"Name": objective_metric_name, "Regex": "accuracy: ([0-9\\.]+)"}],
        max_jobs=2,
        max_parallel_jobs=2,
    )

    tuner.tune(
        inputs=[InputData(channel_name="training", data_source=inputs_s3)],
        job_name=random_suffix_name("tf-tune", 32),
        wait=True,
    )
