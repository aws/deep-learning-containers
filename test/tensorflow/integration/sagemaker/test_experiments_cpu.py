"""SageMaker Experiments tracking integration test for the TF DLC.

Creates an Experiment + Trial, launches a training job, looks up the
auto-created TrialComponent from the training job's ARN, associates it with
the Trial, then cleans up.

Uses SDK v3 (`sagemaker.core.experiments.Experiment` / `_Trial` /
`_TrialComponent` and `ModelTrainer`).

CPU-only by design: the feature under test is the SageMaker Experiments
control plane, not anything CUDA-specific.
"""

import logging
import os
import time

import boto3
from botocore.exceptions import ClientError
from sagemaker.core.experiments.experiment import Experiment
from sagemaker.core.experiments.trial import _Trial
from sagemaker.core.experiments.trial_component import _TrialComponent
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.training.configs import Compute, InputData, SourceCode
from sagemaker.train import ModelTrainer
from test_utils import random_suffix_name

LOG = logging.getLogger(__name__)

INSTANCE_TYPE = "ml.c5.xlarge"


def test_experiments_cpu(image_uri, source_dir, default_region, mnist_s3_uri):
    """Create an Experiment + Trial, run a training job, associate the
    auto-created TrialComponent, then clean everything up.

    After `train(wait=True)` returns, we read the training job ARN from
    `model_trainer._latest_training_job` (avoiding ListTrainingJobs
    eventual-consistency races) and list TrialComponents whose `source_arn`
    matches the job ARN to verify auto-creation."""
    boto_session = boto3.session.Session(region_name=default_region)
    sagemaker_session = Session(boto_session)

    experiment_name = random_suffix_name("tf-dlc-integ-test", 63)
    trial_name = random_suffix_name("tf-dlc-integ-trial", 63)

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
        source_code = SourceCode(source_dir=source_dir, entry_script="mnist.py")
        compute = Compute(instance_type=INSTANCE_TYPE, instance_count=1)
        model_trainer = ModelTrainer(
            training_image=image_uri,
            source_code=source_code,
            compute=compute,
            role=os.environ.get("SM_ROLE_ARN"),
            base_job_name=job_name,
            hyperparameters={"strategy": "none"},
            distributed=None,
        )
        model_trainer.train(
            input_data_config=[InputData(channel_name="training", data_source=mnist_s3_uri)],
            wait=True,
        )

        # Read the training job ARN directly off the trainer rather than
        # ListTrainingJobs — the latter is eventually consistent and racy
        # for jobs that just finished.
        training_job_arn = model_trainer._latest_training_job.training_job_arn
        assert training_job_arn, (
            "ModelTrainer._latest_training_job did not expose a training_job_arn"
        )

        # Verify SageMaker auto-created at least one TrialComponent for this
        # training job — this is the actual feature under test.
        trial_components = list(
            _TrialComponent.list(source_arn=training_job_arn, sagemaker_session=sagemaker_session)
        )
        assert trial_components, (
            f"no TrialComponent was auto-created for training job {training_job_arn}"
        )

        # Associate the TrialComponent with our Trial, then immediately
        # remove + delete to keep cleanup simple.
        trial_component_summary = trial_components[0]
        trial_component = _TrialComponent.load(
            trial_component_name=trial_component_summary.trial_component_name,
            sagemaker_session=sagemaker_session,
        )
        trial.add_trial_component(trial_component)
        trial.remove_trial_component(trial_component_summary.trial_component_name)
        trial_component.delete(force_disassociate=True)
    finally:
        # Best-effort cleanup — narrow to AWS SDK errors, log non-not-found
        # failures so silent leaks don't accumulate in the CI account.
        _safe_delete(trial.delete, f"trial {trial_name}")
        time.sleep(1.2)  # avoid Experiments control-plane throttling
        _safe_delete(experiment.delete, f"experiment {experiment_name}")


def _safe_delete(delete_fn, label: str) -> None:
    try:
        delete_fn()
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("ResourceNotFound", "ValidationException"):
            return
        LOG.warning("cleanup failed for %s: %s", label, e)
