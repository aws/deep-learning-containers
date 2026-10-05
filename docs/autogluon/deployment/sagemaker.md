# Amazon SageMaker AI Deployment

Use the AutoGluon image to run your own scripts in {{ sm_short }} training and processing jobs, and to serve the resulting models from endpoints.

> **Prefer not to write scripts?** [AutoGluon-Cloud](https://auto.gluon.ai/cloud/stable/index.html) ships the training and inference scripts for you
> and provides a simpler API: pass in a DataFrame, get back a DataFrame of predictions.

## Training

The training job runs any script you provide, using AutoGluon or any of the bundled packages. See
[How {{ sm_short }} runs your training image](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html)
for how data, hyperparameters, and model outputs reach your script.

## Inference

The endpoint expects a `model.tar.gz` that contains your model files and an inference handler under `code/`:

```text
model.tar.gz
├── <your model files>
└── code/
    ├── inference.py        # inference handler
    └── requirements.txt    # optional, installed at container start
```

The simplest way to produce it is to have your training script write the model and copy `code/` into `SM_MODEL_DIR`. To use a different handler file
name, set the `SAGEMAKER_PROGRAM` environment variable on the model.

The handler defines two functions:

```python
from typing import Any


def model_fn(model_dir: str) -> Any:
    """Load the model from /opt/ml/model. Called once at startup."""


def transform_fn(
    model: Any,
    request_body: str | bytes,
    input_content_type: str,
    output_content_type: str,
) -> tuple[str | bytes, str]:
    """Make predictions for one request. Return (response_body, response_content_type)."""
```

See [Configuration](../configuration.md) for details.

## Example: AutoGluon-Tabular

This example uses the {{ sm_short }} Python SDK v3 (`sagemaker>=3.0,<4`).

### Train

`train.py` fits the predictor and copies the `code/` directory, including `inference.py`, into the model artifact.

```python
# code/train.py
import argparse
import os
import shutil

from autogluon.tabular import TabularPredictor

parser = argparse.ArgumentParser()
parser.add_argument("--time_limit", type=int, default=600)
args = parser.parse_args()

model_dir = os.environ["SM_MODEL_DIR"]
predictor = TabularPredictor(label="class", path=model_dir)
predictor.fit(
    train_data=os.environ["SM_CHANNEL_TRAIN"] + "/train.csv",
    time_limit=args.time_limit,
)

# Package this directory (including inference.py) with the model
shutil.copytree(os.path.dirname(os.path.abspath(__file__)), model_dir + "/code", dirs_exist_ok=True)
```

Launch the training job. Write [`inference.py`](#deploy) first, since it is packaged at training time.

```python
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.training.configs import Compute, InputData, SourceCode
from sagemaker.train import ModelTrainer

session = Session()
ROLE_ARN = "arn:aws:iam::<account_id>:role/<SageMakerRole>"
IMAGE_URI = f"763104351884.dkr.ecr.{session.boto_region_name}.amazonaws.com/autogluon:1.6-cpu-amzn2023"

trainer = ModelTrainer(
    training_image=IMAGE_URI,
    source_code=SourceCode(source_dir="code", entry_script="train.py"),
    compute=Compute(instance_type="ml.m5.2xlarge", instance_count=1),
    hyperparameters={"time_limit": 600},
    role=ROLE_ARN,
)
trainer.train(
    input_data_config=[
        InputData(channel_name="train", data_source=session.upload_data("train.csv")),
    ],
)
print(trainer._latest_training_job.model_artifacts.s3_model_artifacts)  # s3://.../model.tar.gz
```

### Deploy

`inference.py` loads the predictor and returns predictions as JSON.

```python
# code/inference.py
from io import StringIO

import pandas as pd
from autogluon.tabular import TabularPredictor


def model_fn(model_dir):
    return TabularPredictor.load(model_dir)


def transform_fn(model, request_body, input_content_type, output_content_type):
    data = pd.read_json(StringIO(request_body))
    predictions = model.predict(data)
    return predictions.to_json(orient="records"), "application/json"
```

Deploy the model artifact from the training job, then invoke the endpoint:

```python
import pandas as pd
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import ContainerDefinition, ProductionVariant

MODEL_DATA_URL = "s3://<bucket>/<training-job>/output/model.tar.gz"
NAME = "autogluon-tabular"

model = Model.create(
    model_name=NAME,
    primary_container=ContainerDefinition(
        image=IMAGE_URI,
        model_data_url=MODEL_DATA_URL,
    ),
    execution_role_arn=ROLE_ARN,
)
config = EndpointConfig.create(
    endpoint_config_name=NAME,
    production_variants=[
        ProductionVariant(
            variant_name="AllTraffic",
            model_name=NAME,
            instance_type="ml.m5.xlarge",
            initial_instance_count=1,
        ),
    ],
)
endpoint = Endpoint.create(endpoint_name=NAME, endpoint_config_name=NAME)
endpoint.wait_for_status("InService")

rows = pd.read_csv("train.csv").drop(columns="class").head(3)
response = endpoint.invoke(body=rows.to_json(orient="records"), content_type="application/json")
print(response.body.read().decode())

# Clean up
for resource in (endpoint, config, model):
    resource.delete()
```

## Notes

- **GPU:** use the `1.6-cu133-amzn2023` tag with GPU instances. GPU endpoints also need
  `inference_ami_version="al2023-ami-sagemaker-inference-gpu-4-1"` in the `ProductionVariant`.
- **Local testing:** pass `training_mode=Mode.LOCAL_CONTAINER` (from `sagemaker.train.model_trainer`) and `instance_type="local_cpu"` to run the
  training job in Docker on your machine. Requires Docker Compose.
- **More examples:** the AutoGluon tutorials for [tabular](https://auto.gluon.ai/stable/tutorials/tabular/index.html) and
  [time series](https://auto.gluon.ai/stable/tutorials/timeseries/index.html) data. Any of them can go into `train.py` with a matching `inference.py`.
