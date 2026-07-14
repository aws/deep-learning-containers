# Amazon SageMaker AI Deployment

Use the TensorFlow DLC for training jobs launched via the SageMaker Python SDK or `boto3`. The images bundle the `sagemaker-tensorflow-training`
toolkit, OpenMPI for multi-node coordination, and SageMaker-specific Python libraries (`mlflow`, `smclarify`, `pandas`, `seaborn`, `s3fs`,
etc.).

The TensorFlow 2.21 DLC is a **SageMaker-only** release — there is no EC2 or EKS variant of this image. Point your estimators, processors, and
training jobs at the SageMaker image URIs shown below.

## SageMaker Python SDK v2

Pass the DLC image URI via `image_uri=` rather than `framework_version=`:

```python
from sagemaker.tensorflow import TensorFlow

estimator = TensorFlow(
    image_uri="public.ecr.aws/deep-learning-containers/tensorflow-training:2.21-gpu-py312-cu129-amzn2023-sagemaker",
    role="arn:aws:iam::<account_id>:role/<role_name>",
    entry_point="train.py",
    source_dir="src",
    instance_type="ml.p5.48xlarge",
    instance_count=2,
    distribution={"mpi": {"enabled": True, "processes_per_host": 8}},
)

estimator.fit({"training": "s3://<bucket>/datasets/train/"})
```

The toolkit invokes your `entry_point` script over MPI when `mpi.enabled` is set and as a single-process launcher otherwise.

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
    training_job_name="tensorflow-train",
    role_arn="arn:aws:iam::<account_id>:role/<role_name>",
    algorithm_specification=AlgorithmSpecification(
        training_image="public.ecr.aws/deep-learning-containers/tensorflow-training:2.21-gpu-py312-cu129-amzn2023-sagemaker",
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

## Processing Jobs

### `TensorFlowProcessor` (SDK v2)

```python
from sagemaker.tensorflow.processing import TensorFlowProcessor

processor = TensorFlowProcessor(
    image_uri="public.ecr.aws/deep-learning-containers/tensorflow-training:2.21-cpu-py312-amzn2023-sagemaker",
    framework_version="2.21",  # workaround: value ignored when image_uri is set
    role="arn:aws:iam::<account_id>:role/<role_name>",
    instance_type="ml.m5.xlarge",
    instance_count=1,
)
```

### `FrameworkProcessor` (SDK v3)

```python
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import FrameworkProcessor

processor = FrameworkProcessor(
    image_uri="public.ecr.aws/deep-learning-containers/tensorflow-training:2.21-cpu-py312-amzn2023-sagemaker",
    role="arn:aws:iam::<account_id>:role/<role_name>",
    instance_type="ml.m5.xlarge",
    instance_count=1,
)
```

## Multi-Node Training over EFA

SageMaker provisions EFA on the instance types that support it (e.g., `ml.p5.48xlarge`, `ml.p4d.24xlarge`). The GPU image ships with EFA 1.49.0 and
OpenMPI 4.1.8 pre-installed — no extra plumbing in your training script.

Multi-node peer discovery is handled by the `sagemaker_tensorflow_container` training toolkit, which parses
`/opt/ml/input/config/resourceconfig.json` at container start to enumerate hosts, current host rank, and network interface. Peers are addressed
by the SageMaker-assigned hostnames (e.g., `algo-1`, `algo-2`) that resolve inside the training cluster — your script does not need to know about
this.

## Container Layout

| Path                                 | Purpose                                                                                        |
| ------------------------------------ | ---------------------------------------------------------------------------------------------- |
| `/opt/ml/input/data/<channel_name>/` | Training data SageMaker mounts from S3 (one subdir per channel)                                |
| `/opt/ml/model/`                     | Write your final model artifacts here — SageMaker uploads to `OutputDataConfig.s3_output_path` |
| `/opt/ml/output/`                    | Auxiliary outputs (logs, checkpoints)                                                          |
| `/opt/ml/code/`                      | Your training source dir (populated from the SDK's `source_dir`)                               |
| `/opt/venv/`                         | Python venv with TensorFlow + DLC libraries                                                    |

## Notes

- The image sets `SAGEMAKER_TRAINING_MODULE=sagemaker_tensorflow_container.training:main` — this is the entry-point the SageMaker
  training toolkit invokes at container start, which in turn launches your `entry_point` script.
- For a baseline driver/AMI compatible with these CUDA 12.9 images, request the latest SageMaker training AMI when launching jobs.
- For inference workloads, use the [vLLM](../../vllm/index.md), [vLLM-Omni](../../vllm-omni/index.md), or [Ray](../../ray/index.md) DLCs — this image
  is not configured for serving.
