"""SageMaker CPU training integration tests for TensorFlow DLC.

Uses SDK v3 (ModelTrainer + InputData + SourceCode):

  - test_mnist_cpu                      parametrized: single-host and 2-host
                                        plain Keras (each host trains
                                        independently; smoke-tests the
                                        multi-host launcher path).
  - test_mnist_distributed_mwms_cpu     2-host, MultiWorkerMirroredStrategy
                                        (RING). On TF 2.21 / Keras 3,
                                        model.fit() under MWMS hits a
                                        PerReplica distribution gap, so the
                                        entry script uses a custom training
                                        loop (strategy.run + reduce).
"""

import pytest
from sagemaker.core.training.configs import InputData

INSTANCE_TYPE = "ml.c5.xlarge"


@pytest.mark.parametrize(
    "instance_count,job_name_prefix",
    [
        (1, "tf-mnist-cpu"),
        (2, "tf-mnist-2h-cpu"),
    ],
    ids=["single-node", "multi-host"],
)
def test_mnist_cpu(
    instance_count,
    job_name_prefix,
    sm_trainer,
    assert_s3_file_exists,
    mnist_s3_uri,
):
    """CPU training smoke tests across single-node and multi-host, plain Keras.

    - single-node: SageMaker mounts the MNIST S3 prefix at
      SM_CHANNEL_TRAINING. After the job completes we assert the model
      artifact was uploaded to S3 — a deployability smoke check.
    - multi-host: Runs the same plain Keras script on each host
      independently — no collective op or parameter coordination.
      Smoke-tests the multi-host SageMaker launcher path. Each host writes
      the same artifact, but only the chief saves to SM_MODEL_DIR (the
      script enforces the chief gate)."""
    model_trainer = sm_trainer(
        entry_script="mnist.py",
        instance_type=INSTANCE_TYPE,
        instance_count=instance_count,
        hyperparameters={"strategy": "none"},
        input_data=[InputData(channel_name="training", data_source=mnist_s3_uri)],
        job_name_prefix=job_name_prefix,
    )
    assert_s3_file_exists(
        model_trainer._latest_training_job.model_artifacts.s3_model_artifacts
    )


def test_mnist_distributed_mwms_cpu(sm_trainer, mnist_s3_uri):
    """2-node distributed CPU training with MultiWorkerMirroredStrategy (RING).

    The training script wires TF_CONFIG from SM_HOSTS so MWMS handles
    inter-host gRPC. On TF 2.21 / Keras 3, model.fit() under MWMS hits a
    PerReplica distribution gap; the script uses a custom strategy.run
    training loop instead."""
    sm_trainer(
        entry_script="mnist.py",
        instance_type=INSTANCE_TYPE,
        instance_count=2,
        hyperparameters={"epochs": "2", "strategy": "mwms"},
        input_data=[InputData(channel_name="training", data_source=mnist_s3_uri)],
        job_name_prefix="tf-mnist-mwms-cpu",
    )
