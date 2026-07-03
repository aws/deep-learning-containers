# Amazon SageMaker AI Deployment

Use the SageMaker variants for training jobs launched via the SageMaker Python SDK or `boto3`. The `*-sagemaker` images bundle the
`sagemaker-pytorch-training` toolkit, hostname-fixup wrappers for multi-node MPI, and SageMaker-specific Python libraries (`mlflow`, `smclarify`,
`pandas`, `seaborn`, `s3fs`, etc.).

## SageMaker Python SDK v2

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    image_uri="public.ecr.aws/deep-learning-containers/pytorch:2.12-cu130-amzn2023-sagemaker",
    role="arn:aws:iam::<account_id>:role/<role_name>",
    entry_point="train.py",
    source_dir="src",
    instance_type="ml.p5.48xlarge",
    instance_count=2,
    distribution={"torch_distributed": {"enabled": True}},
)

estimator.fit({"training": "s3://<bucket>/datasets/train/"})
```

The toolkit invokes your `entry_point` script under `torchrun` when `torch_distributed` is enabled, and over MPI when `mpi.enabled` is set.

## SageMaker Python SDK v3

```python
from sagemaker.core.resources import TrainingJob
from sagemaker.core.shapes import (
    AlgorithmSpecification,
    Channel,
    DataSource,
    InputDataConfig,
    OutputDataConfig,
    ResourceConfig,
    S3DataSource,
    StoppingCondition,
)

job = TrainingJob.create(
    training_job_name="pytorch-train",
    role_arn="arn:aws:iam::<account_id>:role/<role_name>",
    algorithm_specification=AlgorithmSpecification(
        training_image="public.ecr.aws/deep-learning-containers/pytorch:2.12-cu130-amzn2023-sagemaker",
        training_input_mode="File",
    ),
    resource_config=ResourceConfig(
        instance_type="ml.p5.48xlarge",
        instance_count=2,
        volume_size_in_gb=200,
    ),
    input_data_config=[
        InputDataConfig(
            channel_name="training",
            data_source=DataSource(
                s3_data_source=S3DataSource(
                    s3_data_type="S3Prefix",
                    s3_uri="s3://<bucket>/datasets/train/",
                ),
            ),
        ),
    ],
    output_data_config=OutputDataConfig(s3_output_path="s3://<bucket>/output/"),
    stopping_condition=StoppingCondition(max_runtime_in_seconds=3600),
)
job.wait_for_status("Completed")
```

## Multi-Node Training over EFA

SageMaker provisions EFA on the instance types that support it (e.g., `ml.p5.48xlarge`, `ml.p4d.24xlarge`). The image's NCCL OFI plugin picks up the
EFA fabric automatically — no extra plumbing in your training script.

The image ships a SageMaker-specific entrypoint (`start_with_right_hostname.sh`) that rewrites the container hostname to the value SageMaker assigns.
This is required for NCCL/MPI to identify peers correctly across nodes, and runs automatically — your script does not need to know about it.

To verify EFA before a long job, you can inject a quick `all_reduce_perf` warm-up at the top of `train.py`:

```python
import subprocess
subprocess.run(
    ["/usr/local/bin/all_reduce_perf", "-b", "8", "-e", "1G", "-f", "2", "-g", "1"],
    check=False,
)
```

## Container Layout

| Path | Purpose |
| --- | --- |
| `/opt/ml/input/data/<channel_name>/` | Training data SageMaker mounts from S3 (one subdir per channel) |
| `/opt/ml/model/` | Write your final model artifacts here — SageMaker uploads to `OutputDataConfig.s3_output_path` |
| `/opt/ml/output/` | Auxiliary outputs (logs, checkpoints) |
| `/opt/ml/code/` | Your training source dir (populated from the SDK's `source_dir`) |
| `/opt/venv/` | Python venv with PyTorch + DLC libraries |

## Notes

- The SageMaker variants set `SAGEMAKER_TRAINING_MODULE=sagemaker_pytorch_container.training:main` so the toolkit handles entry-point invocation.
- For a baseline driver/AMI compatible with these CUDA 13.0 images, request the latest SageMaker training AMI when launching jobs.
- For inference workloads, use the [vLLM](../../vllm/index.md), [vLLM-Omni](../../vllm-omni/index.md), or [Ray](../../ray/index.md) DLCs — this image
  is not configured for serving.
