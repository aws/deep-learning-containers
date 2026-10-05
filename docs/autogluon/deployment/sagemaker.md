# Amazon SageMaker AI Deployment

The AutoGluon image runs both {{ sm_short }} training jobs and {{ sm_short }} endpoints. The typical workflow is:

1. **Train** — a training job runs your `train.py`, fits an AutoGluon predictor, and saves it to `SM_MODEL_DIR` together with an inference handler
   under `code/`.
2. **Deploy** — a {{ sm_short }} model points at the same image and the `model.tar.gz` produced by the training job. The server loads your handler and
   serves predictions on `POST /invocations`.

The examples below train a `TabularPredictor`. The same pattern works for `TimeSeriesPredictor` and `MultiModalPredictor`.

## Project Layout

```
code/
├── train.py    # training entry script
└── serve.py    # inference handler, copied into the model artifact by train.py
```

### `train.py`

```python
import os
import shutil
from pathlib import Path

from autogluon.tabular import TabularDataset, TabularPredictor

if __name__ == "__main__":
    model_dir = Path(os.environ["SM_MODEL_DIR"])
    train_dir = Path(os.environ["SM_CHANNEL_TRAIN"])
    num_gpus = int(os.environ.get("SM_NUM_GPUS", "0"))

    train_data = TabularDataset(str(train_dir / "train.csv"))
    TabularPredictor(label="class", path=str(model_dir)).fit(
        train_data,
        presets="medium",
        num_gpus=num_gpus,
    )

    # Ship the inference handler inside the model artifact
    code_dir = model_dir / "code"
    code_dir.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).parent / "serve.py", code_dir / "serve.py")
```

Anything written to `SM_MODEL_DIR` is packaged into `model.tar.gz` at the end of the job. Add a `code/requirements.txt` to the artifact the same way
if your handler needs extra packages at serving time.

### `serve.py`

```python
from io import StringIO

import pandas as pd
from autogluon.tabular import TabularPredictor


def model_fn(model_dir):
    predictor = TabularPredictor.load(model_dir)
    predictor.persist()  # keep models in memory between requests
    return predictor


def transform_fn(model, request_body, input_content_type, output_content_type):
    if input_content_type != "application/json":
        raise ValueError(f"unsupported content type: {input_content_type}")  # returns HTTP 400

    data = pd.read_json(StringIO(request_body))
    predictions = model.predict(data)
    return predictions.to_frame(name=model.label).to_json(orient="records"), "application/json"
```

See [Configuration](configuration.md#inference-handler) for the full handler contract.

## Training Job

Use the {{ sm_short }} Python SDK to launch the training job. The `train` channel is mounted at `SM_CHANNEL_TRAIN`.

```python
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.training.configs import Compute, InputData, SourceCode
from sagemaker.train import ModelTrainer

session = Session()
REGION = session.boto_region_name
ROLE_ARN = "arn:aws:iam::<account_id>:role/<SageMakerRole>"
IMAGE_URI = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/autogluon:1.6-cpu-amzn2023"

trainer = ModelTrainer(
    training_image=IMAGE_URI,
    source_code=SourceCode(source_dir="code", entry_script="train.py"),
    compute=Compute(instance_type="ml.m5.2xlarge", instance_count=1),
    role=ROLE_ARN,
    base_job_name="autogluon-tabular",
)
trainer.train(
    input_data_config=[
        InputData(channel_name="train", data_source=session.upload_data("train.csv", key_prefix="autogluon/train")),
    ],
    wait=True,
)
model_data_url = trainer._latest_training_job.model_artifacts.s3_model_artifacts
print(model_data_url)
```

For GPU training, switch to the `1.6-cu133-amzn2023` tag and a GPU instance type such as `ml.g5.2xlarge`.

## Real-Time Endpoint

Deploy the trained artifact with the same image. `SAGEMAKER_PROGRAM` names the handler file inside `code/` (default `inference.py`).

```python
import json

import boto3

sm = boto3.client("sagemaker")
smrt = boto3.client("sagemaker-runtime")
NAME = "autogluon-tabular"

# 1. Model — same image as training, plus the artifact from the training job
sm.create_model(
    ModelName=NAME,
    PrimaryContainer={
        "Image": IMAGE_URI,
        "ModelDataUrl": model_data_url,
        "Environment": {"SAGEMAKER_PROGRAM": "serve.py"},
    },
    ExecutionRoleArn=ROLE_ARN,
)

# 2. Endpoint config
sm.create_endpoint_config(
    EndpointConfigName=NAME,
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": NAME,
        "InitialInstanceCount": 1,
        "InstanceType": "ml.c5.xlarge",
        # model_fn runs before /ping succeeds, so allow time for large predictors to load
        "ContainerStartupHealthCheckTimeoutInSeconds": 600,
    }],
)

# 3. Endpoint
sm.create_endpoint(EndpointName=NAME, EndpointConfigName=NAME)
sm.get_waiter("endpoint_in_service").wait(EndpointName=NAME)

# 4. Invoke
# Every feature column used in training, without the label
rows = [{
    "age": 25, "workclass": "Private", "fnlwgt": 178478, "education": "Bachelors", "education-num": 13,
    "marital-status": "Never-married", "occupation": "Tech-support", "relationship": "Own-child", "race": "White",
    "sex": "Female", "capital-gain": 0, "capital-loss": 0, "hours-per-week": 40, "native-country": "United-States",
}]
resp = smrt.invoke_endpoint(
    EndpointName=NAME,
    ContentType="application/json",
    Accept="application/json",
    Body=json.dumps(rows),
)
print(json.loads(resp["Body"].read()))

# 5. Cleanup
sm.delete_endpoint(EndpointName=NAME)
sm.delete_endpoint_config(EndpointConfigName=NAME)
sm.delete_model(ModelName=NAME)
```

## Notes

- **GPU endpoints need a CUDA 13 inference AMI.** The GPU image is built on CUDA 13.3. Set
  `"InferenceAmiVersion": "al2023-ami-sagemaker-inference-gpu-4-1"` in the production variant.
- **Multi-model endpoints are not supported.** Each container serves one model artifact.
