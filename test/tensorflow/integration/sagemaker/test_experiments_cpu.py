"""SageMaker Experiments tracking integration test for the TF DLC.

Mirrors master's `test_experiments.py` pattern: create an Experiment +
Trial, launch a training job, look up the auto-created TrialComponent
from the training job's ARN, associate it with the Trial, then clean up.

Translated to SDK v3 (`sagemaker.core.experiments.Experiment` /
`_Trial` / `_TrialComponent` and `ModelTrainer` instead of the v2
`smexperiments` package + `TensorFlow` estimator).

CPU-only by design: the feature under test is the SageMaker Experiments
control plane, not anything CUDA-specific.
"""

import datetime
import os
import random
import time

import boto3
from sagemaker.core.experiments.experiment import Experiment
from sagemaker.core.experiments.trial import _Trial
from sagemaker.core.experiments.trial_component import _TrialComponent
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.training.configs import Compute, InputData, SourceCode
from sagemaker.train import ModelTrainer
from test_utils import random_suffix_name

RESOURCE_DIR = os.path.join(os.path.dirname(__file__), "resources")
SOURCE_DIR = os.path.join(RESOURCE_DIR, "scripts")
MNIST_DATA_DIR = os.path.join(RESOURCE_DIR, "mnist", "data")
INSTANCE_TYPE = "ml.c5.xlarge"
IMAGE_URI = os.environ["TEST_IMAGE_URI"]
DEFAULT_REGION = "us-west-2"


def test_experiments_cpu():
    """Create an Experiment + Trial, run a training job, associate the
    auto-created TrialComponent, then clean everything up.

    Equivalent to master's `test_experiments.test_training`. The training
    job ARN is looked up via the SageMaker SDK after `train(wait=True)`
    returns; we then list TrialComponents whose `source_arn` matches the
    job ARN and verify at least one was auto-created."""
    boto_session = boto3.session.Session(region_name=DEFAULT_REGION)
    sagemaker_session = Session(boto_session)
    sm_client = boto_session.client("sagemaker")

    # Match master's unique-id format so concurrent CI runs don't collide.
    random.seed(f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}")
    unique_id = random.randint(1, 6000)
    experiment_name = f"tf-dlc-integ-test-{unique_id}-{int(time.time())}"
    trial_name = f"tf-dlc-integ-trial-{unique_id}-{int(time.time())}"

    inputs_s3 = sagemaker_session.upload_data(path=MNIST_DATA_DIR, key_prefix="scriptmode/mnist")

    experiment = Experiment.create(
        experiment_name=experiment_name,
        description="Integration test experiment from TF DLC",
        sagemaker_session=sagemaker_session,
    )
    trial = _Trial.create(
        experiment_name=experiment_name,
        trial_name=trial_name,
        sagemaker_session=sagemaker_session,
    )

    job_name = random_suffix_name("tf-experiments-cpu", 32)
    try:
        source_code = SourceCode(source_dir=SOURCE_DIR, entry_script="mnist.py")
        compute = Compute(instance_type=INSTANCE_TYPE, instance_count=1)
        model_trainer = ModelTrainer(
            training_image=IMAGE_URI,
            source_code=source_code,
            compute=compute,
            role=os.environ.get("SM_ROLE_ARN"),
            base_job_name=job_name,
            hyperparameters={"epochs": "1", "strategy": "none"},
            distributed=None,
        )
        model_trainer.train(
            input_data_config=[InputData(channel_name="training", data_source=inputs_s3)],
            wait=True,
        )

        # Resolve the actual training job name (SDK appends a timestamp to
        # base_job_name) and look up its ARN. Iterate recent jobs that match
        # the base prefix and pick the most recent succeeded one.
        jobs = sm_client.list_training_jobs(
            NameContains=job_name, StatusEquals="Completed", MaxResults=1, SortOrder="Descending"
        )["TrainingJobSummaries"]
        assert jobs, f"no completed training job found with prefix {job_name}"
        training_job_arn = jobs[0]["TrainingJobArn"]

        # Verify SageMaker auto-created at least one TrialComponent for this
        # training job — this is the actual feature under test.
        trial_components = list(
            _TrialComponent.list(source_arn=training_job_arn, sagemaker_session=sagemaker_session)
        )
        assert trial_components, (
            f"no TrialComponent was auto-created for training job {training_job_arn}"
        )

        # Associate the TrialComponent with our Trial — mirrors master's
        # pattern. Then immediately remove + delete to keep cleanup simple.
        trial_component_summary = trial_components[0]
        trial_component = _TrialComponent.load(
            trial_component_name=trial_component_summary.trial_component_name,
            sagemaker_session=sagemaker_session,
        )
        trial.add_trial_component(trial_component)
        trial.remove_trial_component(trial_component_summary.trial_component_name)
        trial_component.delete(force_disassociate=True)
    finally:
        # Best-effort cleanup — keep the test from leaking state on failure.
        try:
            trial.delete()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.2)  # avoid Experiments control-plane throttling
        try:
            experiment.delete()
        except Exception:  # noqa: BLE001
            pass
