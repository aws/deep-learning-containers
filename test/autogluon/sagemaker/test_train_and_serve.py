"""Train with the unified AutoGluon image, then serve the resulting artifact."""

import csv
import json
import logging
import os
from pathlib import Path

import pytest
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import ContainerDefinition, ProductionVariant
from sagemaker.core.training.configs import Compute, InputData, SourceCode
from sagemaker.train import ModelTrainer
from test_utils import random_suffix_name
from test_utils.constants import INFERENCE_AMI_VERSION

LOGGER = logging.getLogger(__name__)
RESOURCE_DIR = Path(__file__).parent / "resources"
CODE_DIR = RESOURCE_DIR / "code"
DATA_DIR = RESOURCE_DIR / "data"
PREDICTION_LENGTH = 5

_WORKLOADS = {
    "tabular": {
        "source_dir": CODE_DIR / "tabular",
        "data": DATA_DIR / "tabular" / "train.csv",
    },
    "timeseries": {
        "source_dir": CODE_DIR / "timeseries",
        "data": DATA_DIR / "timeseries" / "train.csv",
    },
}

_INSTANCE_BY_DEVICE = {
    "cpu": {
        "training": "ml.m5.2xlarge",
        "inference": "ml.c5.xlarge",
    },
    "gpu": {
        "training": "ml.g4dn.2xlarge",
        "inference": "ml.g6.xlarge",
    },
}
_NUMERIC_COLUMNS = {
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
}


def _cleanup(resources):
    for resource in reversed(resources):
        try:
            resource.delete()
        except Exception as error:  # noqa: BLE001 - cleanup must not mask test failures
            LOGGER.warning("Cleanup of %s failed: %s", type(resource).__name__, error)


@pytest.fixture(scope="module")
def device_config():
    device = os.environ.get("SM_DEVICE_TYPE", "").lower()
    assert device in _INSTANCE_BY_DEVICE, f"SM_DEVICE_TYPE must be cpu or gpu, got {device!r}"
    return device, _INSTANCE_BY_DEVICE[device]


@pytest.fixture(scope="module", params=_WORKLOADS)
def workload(request):
    return request.param, _WORKLOADS[request.param]


@pytest.fixture(scope="module")
def model_data_url(image_uri, sagemaker_session, device_config, workload):
    """Train once and return the model artifact produced by the tested image."""
    device, instances = device_config
    workload_name, config = workload
    key_prefix = random_suffix_name(f"ag-e2e-{workload_name}", 32)
    trainer = ModelTrainer(
        training_image=image_uri,
        source_code=SourceCode(source_dir=str(config["source_dir"]), entry_script="train.py"),
        compute=Compute(instance_type=instances["training"], instance_count=1),
        role=os.environ["SM_ROLE_ARN"],
        base_job_name=random_suffix_name(f"ag-{workload_name}-{device}", 32),
    )
    trainer.train(
        input_data_config=[
            InputData(
                channel_name="train",
                data_source=sagemaker_session.upload_data(
                    path=str(config["data"]),
                    key_prefix=f"{key_prefix}/train",
                ),
            )
        ],
        wait=True,
    )
    return trainer._latest_training_job.model_artifacts.s3_model_artifacts


@pytest.fixture(scope="module")
def endpoint(aws_session, image_uri, model_data_url, device_config, workload):
    """Deploy the training artifact with the same unified image."""
    device, instances = device_config
    workload_name, _ = workload
    endpoint_name = random_suffix_name(f"ag-{workload_name}", 63)
    model_name = random_suffix_name(f"ag-{workload_name}-model", 63)
    resources = []
    try:
        model = Model.create(
            model_name=model_name,
            primary_container=ContainerDefinition(
                image=image_uri,
                model_data_url=model_data_url,
                environment={"SAGEMAKER_PROGRAM": "serve.py"},
            ),
            execution_role_arn=os.environ["SM_ROLE_ARN"],
            session=aws_session.session,
        )
        resources.append(model)

        variant = {
            "variant_name": "AllTraffic",
            "model_name": model_name,
            "initial_instance_count": 1,
            "instance_type": instances["inference"],
            "container_startup_health_check_timeout_in_seconds": 600,
        }
        if device == "gpu":
            variant["inference_ami_version"] = INFERENCE_AMI_VERSION
        endpoint_config = EndpointConfig.create(
            endpoint_config_name=endpoint_name,
            production_variants=[ProductionVariant(**variant)],
            session=aws_session.session,
        )
        resources.append(endpoint_config)

        deployed = Endpoint.create(
            endpoint_name=endpoint_name,
            endpoint_config_name=endpoint_name,
            session=aws_session.session,
        )
        resources.append(deployed)
        deployed.wait_for_status("InService", timeout=1800)
        yield deployed
    finally:
        _cleanup(resources)


def _assert_tabular_prediction(endpoint, data_path):
    with data_path.open(newline="") as data_file:
        row = next(csv.DictReader(data_file))
    expected_labels = {"<=50K", ">50K"}
    row.pop("class")
    for column in _NUMERIC_COLUMNS:
        row[column] = int(row[column])

    response = endpoint.invoke(
        body=json.dumps([row]),
        content_type="application/json",
        accept="application/json",
    )
    prediction = json.loads(response.body.read())

    assert len(prediction) == 1
    assert prediction[0]["class"].strip() in expected_labels


def _assert_timeseries_prediction(endpoint, data_path):
    with data_path.open(newline="") as data_file:
        rows = list(csv.DictReader(data_file))
    for row in rows:
        row["target"] = float(row["target"])

    response = endpoint.invoke(
        body=json.dumps(rows),
        content_type="application/json",
        accept="application/json",
    )
    prediction = json.loads(response.body.read())

    assert len(prediction) == 2 * PREDICTION_LENGTH
    assert {row["item_id"] for row in prediction} == {"series_1", "series_2"}
    assert all(isinstance(row["mean"], (int, float)) for row in prediction)


def test_train_then_serve(endpoint, workload):
    workload_name, config = workload
    if workload_name == "tabular":
        _assert_tabular_prediction(endpoint, config["data"])
    else:
        _assert_timeseries_prediction(endpoint, config["data"])
