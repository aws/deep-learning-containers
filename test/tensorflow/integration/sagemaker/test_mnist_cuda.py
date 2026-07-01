"""SageMaker GPU training integration tests for TensorFlow DLC.

Uses SDK v3 (ModelTrainer + InputData + SourceCode):

  - test_mnist_gpu                      parametrized: single-GPU, multi-host,
                                        and MirroredStrategy variants on the
                                        GPU image.
  - test_mnist_distributed_mwms_gpu     2-host, MultiWorkerMirroredStrategy
                                        (NCCL). On TF 2.21 / Keras 3,
                                        model.fit() under MWMS hits a
                                        PerReplica distribution gap, so the
                                        entry script uses a custom training
                                        loop (strategy.run + reduce).

Tests launch real SageMaker training jobs — no GPU needed on the runner.
"""

import pytest
from sagemaker.core.training.configs import InputData


@pytest.mark.parametrize(
    "instance_type,instance_count,strategy",
    [
        ("ml.g4dn.xlarge", 1, "none"),
        # DLC test account has ml.g4dn.xlarge quota=1; multi-host needs 2
        # instances. ml.g4dn.2xlarge has quota=8, same T4 GPU.
        ("ml.g4dn.2xlarge", 2, "none"),
        ("ml.g4dn.12xlarge", 1, "mirrored"),
    ],
    ids=["single-gpu", "multi-host", "mirrored"],
)
def test_mnist_gpu(
    instance_type,
    instance_count,
    strategy,
    sm_trainer,
    assert_s3_file_exists,
    mnist_s3_uri,
):
    """GPU training smoke tests across single-GPU, multi-host, and MirroredStrategy.

    - single-gpu: TF picks up the one visible GPU automatically without any
      explicit distribute scope.
    - multi-host: Each host trains independently — no NCCL, no TF_CONFIG.
      Smoke-tests the GPU multi-host launcher path.
    - mirrored: Single-host multi-GPU with tf.distribute.MirroredStrategy.
      Covers the common DLC scenario of one box with all GPUs; NCCL all-reduce
      across the local GPUs only."""
    model_trainer = sm_trainer(
        entry_script="mnist.py",
        instance_type=instance_type,
        instance_count=instance_count,
        hyperparameters={"strategy": strategy},
        input_data=[InputData(channel_name="training", data_source=mnist_s3_uri)],
        job_name_prefix=f"tf-mnist-gpu-{strategy}",
    )
    assert_s3_file_exists(model_trainer._latest_training_job.model_artifacts.s3_model_artifacts)


def test_mnist_distributed_mwms_gpu(sm_trainer, mnist_s3_uri):
    """2-node distributed GPU training with MultiWorkerMirroredStrategy (NCCL).

    The training script uses a custom strategy.run loop on TF 2.21 / Keras 3
    (model.fit() hits a PerReplica distribution gap under MWMS). If NCCL
    fails to initialise or all_reduce hangs, the SageMaker job fails and
    pytest surfaces that here."""
    sm_trainer(
        entry_script="mnist.py",
        instance_type="ml.g4dn.12xlarge",
        instance_count=2,
        hyperparameters={"epochs": "2", "strategy": "mwms"},
        environment={"FI_EFA_FORK_SAFE": "1"},
        input_data=[InputData(channel_name="training", data_source=mnist_s3_uri)],
        job_name_prefix="tf-mnist-mwms-gpu",
    )
