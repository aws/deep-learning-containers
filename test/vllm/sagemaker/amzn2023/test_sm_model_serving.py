"""Config-driven SageMaker endpoint test for vLLM DLC models.

Supports text, audio, and video models. Reads model configuration from
.github/config/model-tests/vllm-sagemaker-endpoint-tests.yml (sagemaker section).

Each entry deploys a model from S3 with configured env vars, sends
requests to the specified route, and validates the response.

Usage:
    pytest test_sm_model_serving.py --image-uri <ecr_uri> --model-name voxtral-mini-4b
    pytest test_sm_model_serving.py --image-uri <ecr_uri>  # runs all sagemaker models
"""

import io
import json
import logging
import os
import time
from pathlib import Path

import boto3
import pytest
import yaml
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import (
    ContainerDefinition,
    ModelDataSource,
    ProductionVariant,
    S3ModelDataSource,
)
from test_utils import random_suffix_name
from test_utils.constants import INFERENCE_AMI_VERSION, SAGEMAKER_ROLE

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

CONFIG_PATH = (
    Path(__file__).parents[4] / ".github/config/model-tests/vllm-sagemaker-endpoint-tests.yml"
)


def pytest_addoption(parser):
    parser.addoption("--model-name", default=None, help="Run only this model (default: all)")


def _load_sagemaker_config(config_path, model_name=None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    models = cfg.get("sagemaker", [])
    s3_prefix = cfg.get("s3_prefix", "")
    if model_name:
        models = [m for m in models if m["name"] == model_name]
    for m in models:
        m["s3_path"] = f"{s3_prefix}/{m['s3_model']}"
    return models


def _download_s3(s3_client, s3_uri):
    bucket = s3_uri.split("/")[2]
    key = "/".join(s3_uri.split("/")[3:])
    buf = io.BytesIO()
    s3_client.download_fileobj(bucket, key, buf)
    buf.seek(0)
    return buf.read()


def _build_multipart_body(request_cfg, s3_client):
    boundary = "----FormBoundary7MA4YWxkTrZu0gW"
    parts = []

    for field_name, field_value in request_cfg.get("fields", {}).items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field_name}"\r\n\r\n'
            f"{field_value}\r\n"
        )

    if "file_s3" in request_cfg:
        file_data = _download_s3(s3_client, request_cfg["file_s3"])
        file_field = request_cfg.get("file_field", "file")
        file_name = request_cfg.get("file_name", "input.bin")
        file_ct = request_cfg.get("file_content_type", "application/octet-stream")

        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{file_field}"; filename="{file_name}"\r\n'
            f"Content-Type: {file_ct}\r\n\r\n"
        )
        body = "".join(parts).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()
    else:
        parts.append(f"--{boundary}--\r\n")
        body = "".join(parts).encode()

    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def _build_json_body(request_cfg):
    payload = request_cfg.get("body", {})
    return json.dumps(payload).encode(), "application/json"


def _validate(response_body, raw_bytes, rule):
    if rule.startswith("contains:"):
        expected = rule[len("contains:") :]
        if isinstance(response_body, dict):
            text = response_body.get("text", "")
            if not text:
                text = json.dumps(response_body)
        else:
            text = raw_bytes.decode("utf-8", errors="replace")
        assert expected.lower() in text.lower(), (
            f"Expected '{expected}' in response, got: {text[:500]}"
        )

    elif rule.startswith("json_field:"):
        field_path = rule[len("json_field:") :]
        obj = response_body
        for part in field_path.replace("]", "").replace("[", ".").split("."):
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = obj[part]
        assert obj, f"Field '{field_path}' is empty in response"

    elif rule.startswith("binary_size_gt:"):
        min_size = int(rule[len("binary_size_gt:") :])
        assert len(raw_bytes) > min_size, f"Response size {len(raw_bytes)} <= {min_size}"

    else:
        raise ValueError(f"Unknown validation rule: {rule}")


def _cleanup(resources):
    for resource in resources:
        if resource is None:
            continue
        try:
            resource.delete()
        except Exception as e:
            LOGGER.warning(f"Cleanup {type(resource).__name__} failed: {e}")


def _get_role_arn(region):
    try:
        iam = boto3.client("iam", region_name=region)
        role_info = iam.get_role(RoleName=SAGEMAKER_ROLE)
        return role_info["Role"]["Arn"]
    except Exception:
        sts = boto3.client("sts", region_name=region)
        account = sts.get_caller_identity()["Account"]
        return f"arn:aws:iam::{account}:role/{SAGEMAKER_ROLE}"


def _flatten_jinja(template_str):
    """Flatten newlines into ``{{ "\\n" }}`` so the template survives transport.

    The SM entrypoint reads env vars via ``IFS='=' read`` from a line-oriented
    ``env`` listing, so multi-line values would break across lines. Replacing
    physical newlines with ``{{ "\\n" }}`` keeps the value single-line; vLLM's
    ``--chat-template`` falls back to inline Jinja when the value isn't a valid
    file path and contains ``{``, ``}``, or newline, and each ``{{ "\\n" }}``
    expression evaluates to a real newline at request time.

    No literal-quote wrapping is applied. ``standard-supervisor`` (>=0.1.15,
    pinned in docker/vllm/Dockerfile.amzn2023) ``shlex.quote()``s each argv
    element before the supervisord ``shlex.split`` round-trip, so the inner
    double quotes survive intact. Adding outer single quotes here would pass
    them through verbatim into the rendered template.
    """
    return template_str.replace("\n", '{{ "\\n" }}')


def _deploy_endpoint(image_uri, model_cfg, region):
    endpoint_name = random_suffix_name(f"vllm-{model_cfg['name']}", 50)
    role_arn = _get_role_arn(region)
    env_vars = dict(model_cfg.get("env", {}))

    chat_template_file = model_cfg.get("chat_template_file")
    if chat_template_file:
        repo_root = Path(__file__).parents[4]
        template_path = repo_root / chat_template_file
        env_vars["SM_VLLM_CHAT_TEMPLATE"] = _flatten_jinja(template_path.read_text())

    LOGGER.info(f"Creating model: {endpoint_name}")
    create_kwargs = dict(
        model_name=endpoint_name,
        primary_container=ContainerDefinition(
            image=image_uri,
            model_data_source=ModelDataSource(
                s3_data_source=S3ModelDataSource(
                    s3_uri=model_cfg["s3_path"],
                    s3_data_type="S3Prefix",
                    compression_type="Gzip",
                ),
            ),
            environment=env_vars,
        ),
        execution_role_arn=role_arn,
    )
    if model_cfg.get("network_isolation"):
        create_kwargs["enable_network_isolation"] = True
    model = Model.create(**create_kwargs)

    LOGGER.info(f"Creating endpoint config: {endpoint_name}")
    endpoint_config = EndpointConfig.create(
        endpoint_config_name=endpoint_name,
        production_variants=[
            ProductionVariant(
                variant_name="AllTraffic",
                model_name=endpoint_name,
                initial_instance_count=1,
                instance_type=model_cfg["instance_type"],
                inference_ami_version=INFERENCE_AMI_VERSION,
            ),
        ],
    )

    LOGGER.info(f"Deploying endpoint: {endpoint_name}")
    endpoint = Endpoint.create(
        endpoint_name=endpoint_name,
        endpoint_config_name=endpoint_name,
    )
    endpoint.wait_for_status("InService", timeout=2700)
    LOGGER.info(f"Endpoint InService: {endpoint_name}")

    return endpoint_name, model, endpoint_config, endpoint


def _generate_test_params():
    model_name_filter = os.environ.get("SM_MODEL_NAME")
    models = _load_sagemaker_config(CONFIG_PATH, model_name_filter)
    return [(m["name"], m) for m in models]


@pytest.fixture(scope="module", params=_generate_test_params(), ids=lambda x: x[0])
def deployed_model(request, image_uri):
    _, model_cfg = request.param

    pattern = model_cfg.get("required_image_pattern")
    if pattern and pattern not in image_uri:
        pytest.skip(f"Model requires image matching '{pattern}', got: {image_uri}")

    region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")

    endpoint_name, model, endpoint_config, endpoint = _deploy_endpoint(image_uri, model_cfg, region)

    yield {
        "endpoint_name": endpoint_name,
        "model_cfg": model_cfg,
        "region": region,
    }

    _cleanup([endpoint, endpoint_config, model])


def test_model_serving(deployed_model):
    endpoint_name = deployed_model["endpoint_name"]
    model_cfg = deployed_model["model_cfg"]
    region = deployed_model["region"]

    sm_runtime = boto3.client("sagemaker-runtime", region_name=region)
    s3_client = boto3.client("s3", region_name=region)

    test_cases = model_cfg["test_cases"]
    failed = []

    for tc in test_cases:
        tc_name = tc["name"]
        LOGGER.info(f"--- Running: {tc_name} ---")

        content_type = tc.get("content_type", "application/json")
        request_cfg = tc["request"]

        if content_type.startswith("multipart/form-data"):
            body, ct = _build_multipart_body(request_cfg, s3_client)
        else:
            body, ct = _build_json_body(request_cfg)

        invoke_kwargs = {
            "EndpointName": endpoint_name,
            "Body": body,
            "ContentType": ct,
        }
        if "route" in tc:
            invoke_kwargs["CustomAttributes"] = f"route={tc['route']}"

        t0 = time.time()
        try:
            response = sm_runtime.invoke_endpoint(**invoke_kwargs)
            elapsed = time.time() - t0
            raw_bytes = response["Body"].read()

            try:
                response_body = json.loads(raw_bytes)
            except (json.JSONDecodeError, UnicodeDecodeError):
                response_body = None

            LOGGER.info(f"  Response ({elapsed:.1f}s, {len(raw_bytes)} bytes)")
            if response_body:
                LOGGER.info(f"  Body: {json.dumps(response_body)[:300]}")

            _validate(response_body, raw_bytes, tc["validate"])
            LOGGER.info(f"  PASS: {tc_name}")

        except Exception as e:
            elapsed = time.time() - t0
            LOGGER.error(f"  FAIL: {tc_name} ({elapsed:.1f}s) — {e}")
            failed.append((tc_name, str(e)))

    if failed:
        msg = "\n".join(f"  - {name}: {err}" for name, err in failed)
        pytest.fail(f"{len(failed)}/{len(test_cases)} test cases failed:\n{msg}")
