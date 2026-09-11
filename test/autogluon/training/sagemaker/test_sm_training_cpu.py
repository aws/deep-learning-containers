"""SageMaker CPU training integration test for the AutoGluon DLC (SageMaker SDK v3)."""

import os

from sagemaker.core.training.configs import Compute, InputData, SourceCode
from sagemaker.train import ModelTrainer
from test_utils import random_suffix_name

RESOURCE_DIR = os.path.join(os.path.dirname(__file__), "resources")
DATA_DIR = os.path.join(RESOURCE_DIR, "data")
IMAGE_URI = os.environ["TEST_IMAGE_URI"]


def run_tabular_training(sagemaker_session, instance_type, job_name_prefix):
    """Fit and evaluate a TabularPredictor in a SageMaker training job; waits for completion."""
    key_prefix = random_suffix_name("autogluon-dlc-test", 32)
    channels = [
        InputData(
            channel_name=channel,
            data_source=sagemaker_session.upload_data(
                path=os.path.join(DATA_DIR, filename), key_prefix=f"{key_prefix}/{channel}"
            ),
        )
        for channel, filename in (
            ("config", "config.yaml"),
            ("train", "train.csv"),
            ("test", "eval.csv"),
        )
    ]
    ModelTrainer(
        training_image=IMAGE_URI,
        source_code=SourceCode(source_dir=RESOURCE_DIR, entry_script="train_tab.py"),
        compute=Compute(instance_type=instance_type, instance_count=1),
        role=os.environ.get("SM_ROLE_ARN"),
        base_job_name=random_suffix_name(job_name_prefix, 32),
    ).train(input_data_config=channels, wait=True)


def test_tabular_training_cpu(sagemaker_session):
    """Bagged GBM + NN_TORCH fit completes and beats chance on the held-out split."""
    run_tabular_training(sagemaker_session, "ml.m5.2xlarge", "ag-tab-cpu")
