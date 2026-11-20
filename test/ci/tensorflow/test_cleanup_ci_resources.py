"""Tests for test/tensorflow/integration/inference/resources/cleanup_ci_resources.py.

The cleanup step lists every tf220- SageMaker resource in the account and
deletes only those tagged with this job's ID. Uses a fake SageMaker client, so
no AWS access is needed.
"""

import importlib.util
from pathlib import Path

import pytest

# Needs boto3/botocore (installed by the inference PR workflow's hook-test job);
# skipped where only pytest and pyyaml are installed.
botocore_exceptions = pytest.importorskip("botocore.exceptions")
pytest.importorskip("boto3")
ClientError = botocore_exceptions.ClientError

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "test/tensorflow/integration/inference/resources/cleanup_ci_resources.py"
JOB = "123-1-sagemaker-test-gpu"


def _load():
    spec = importlib.util.spec_from_file_location("cleanup_ci_resources", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Paginator:
    def __init__(self, pages):
        self._pages = pages

    def paginate(self, **_kwargs):
        return iter(self._pages)


class FakeSageMaker:
    """Endpoints, configs and models keyed by name -> tag value (None = vanished)."""

    def __init__(self, resources):
        self.resources = resources
        self.deleted = []

    def get_paginator(self, call):
        kind = {
            "list_endpoints": "Endpoint",
            "list_endpoint_configs": "EndpointConfig",
            "list_models": "Model",
        }[call]
        items = [
            {f"{kind}Name": name, f"{kind}Arn": f"arn:{kind}/{name}"}
            for (k, name) in self.resources
            if k == kind
        ]
        return _Paginator([{f"{kind}s": items}])

    def list_tags(self, ResourceArn):
        kind, name = ResourceArn[4:].split("/", 1)
        tag = self.resources[(kind, name)]
        if tag is None:
            raise ClientError(
                {"Error": {"Code": "ValidationException", "Message": "gone"}}, "ListTags"
            )
        return {"Tags": [{"Key": "dlc-ci-job", "Value": tag}]}

    def _delete(self, kind, name):
        self.deleted.append((kind, name))

    def delete_endpoint(self, EndpointName):
        self._delete("Endpoint", EndpointName)

    def delete_endpoint_config(self, EndpointConfigName):
        self._delete("EndpointConfig", EndpointConfigName)

    def delete_model(self, ModelName):
        self._delete("Model", ModelName)


@pytest.fixture
def run(monkeypatch):
    module = _load()

    def _run(resources):
        fake = FakeSageMaker(resources)
        monkeypatch.setattr(module.boto3, "client", lambda *_a, **_k: fake)
        monkeypatch.setenv("DLC_CI_JOB_ID", JOB)
        monkeypatch.setattr("sys.argv", ["cleanup", "--region", "us-west-2"])
        assert module.main() == 0
        return fake.deleted

    return _run


def test_deletes_only_this_jobs_resources(run):
    deleted = run(
        {
            ("Endpoint", "tf220-mine"): JOB,
            ("Endpoint", "tf220-other"): "999-1-sagemaker-test-gpu",
            ("EndpointConfig", "tf220-mine"): JOB,
            ("Model", "tf220-mine-model"): JOB,
        }
    )
    assert sorted(deleted) == [
        ("Endpoint", "tf220-mine"),
        ("EndpointConfig", "tf220-mine"),
        ("Model", "tf220-mine-model"),
    ]


def test_resource_deleted_by_another_job_mid_run_is_skipped(run):
    deleted = run(
        {
            ("EndpointConfig", "tf220-vanished"): None,
            ("EndpointConfig", "tf220-mine"): JOB,
            ("Model", "tf220-mine-model"): JOB,
        }
    )
    assert sorted(deleted) == [("EndpointConfig", "tf220-mine"), ("Model", "tf220-mine-model")]
