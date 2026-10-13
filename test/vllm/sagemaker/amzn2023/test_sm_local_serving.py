"""Local serving test for the vLLM SageMaker image: the ``fleet:`` entries of
.github/config/model-tests/vllm-sagemaker-endpoint-tests.yml.

The config header describes the serving contract and how CI starts the container
(vllm.tests-sagemaker.yml, local-test-models). This test only sends requests and
validates the responses.

Environment:
    SM_MODEL_NAME             required; must name exactly one ``fleet:`` entry
    SM_LOCAL_CONTAINER        container name; required for network_isolation entries
                              (probed with docker exec); when set, a stopped
                              container fails the startup wait at once
    SM_LOCAL_STARTUP_TIMEOUT  seconds to wait for /ping to return 200 (default 1200)

Usage, with the container started as the workflow does:
    cd test/
    SM_MODEL_NAME=<name> python3 -m pytest -vs vllm/sagemaker/amzn2023/test_sm_local_serving.py
"""

import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request

import boto3
import pytest
from sm_serving_common import (
    CONFIG_PATH,
    build_json_body,
    build_multipart_body,
    load_sagemaker_config,
    validate_response,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

PORT = 8080
# SageMaker real-time endpoints fail an invocation that takes longer than this.
INVOKE_TIMEOUT = 60
STARTUP_TIMEOUT = int(os.environ.get("SM_LOCAL_STARTUP_TIMEOUT", "1200"))
PING_INTERVAL = 5


def _send(model_cfg, method, path, body=b"", headers=None, timeout=INVOKE_TIMEOUT):
    """Send one request to the serving container and return ``(status, body bytes)``.

    Raises OSError when no HTTP response arrives (nothing listening yet, timeout).
    """
    headers = headers or {}
    if model_cfg.get("network_isolation"):
        # --network none: the host cannot reach the port, so curl runs in the container.
        container = os.environ.get("SM_LOCAL_CONTAINER")
        if not container:
            raise RuntimeError(
                "SM_LOCAL_CONTAINER must name the serving container: network_isolation "
                "entries run with --network none and are probed with docker exec"
            )
        cmd = ["docker", "exec", "-i", container, "curl", "-sS"]
        cmd += ["-X", method, "--max-time", str(timeout), "-w", "\n%{http_code}"]
        for name, value in headers.items():
            cmd += ["-H", f"{name}: {value}"]
        if body:
            cmd += ["--data-binary", "@-"]
        cmd.append(f"http://localhost:{PORT}{path}")
        try:
            result = subprocess.run(cmd, input=body, capture_output=True, timeout=timeout + 30)
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(f"docker exec curl did not return within {timeout + 30}s") from e
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            raise ConnectionError(f"docker exec curl exited {result.returncode}: {stderr}")
        response_body, _, status = result.stdout.rpartition(b"\n")
        return int(status), response_body

    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}", data=body or None, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _generate_test_params():
    """The ``fleet:`` entry named by SM_MODEL_NAME.

    A missing or unknown name fails collection: an empty parameter set would make
    pytest skip, and a typo would pass CI without testing anything.
    """
    model_name = os.environ.get("SM_MODEL_NAME")
    local_models = [m for m in load_sagemaker_config() if "fleet" in m]
    selected = [m for m in local_models if m["name"] == model_name]
    if len(selected) != 1:
        names = ", ".join(m["name"] for m in local_models)
        pytest.fail(
            f"SM_MODEL_NAME={model_name!r} must name exactly one fleet entry in "
            f"{CONFIG_PATH.name} (fleet entries: {names})",
            pytrace=False,
        )
    return selected


def _fail_unless_running(container):
    """Fail at once when the container has stopped, instead of waiting out the deadline."""
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}} {{.State.ExitCode}}", container],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            f"docker inspect {container} failed: {result.stderr.strip()}; "
            "see the Dump container logs step",
            pytrace=False,
        )
    status, _, exit_code = result.stdout.strip().partition(" ")
    if status != "running":
        pytest.fail(
            f"container {container} is {status} (exit code {exit_code}) before /ping "
            "returned 200; see the Dump container logs step",
            pytrace=False,
        )


@pytest.fixture(scope="module", params=_generate_test_params(), ids=lambda m: m["name"])
def serving_model(request):
    """Wait until the container answers ``/ping`` with 200.

    A stopped container fails at once. standard-supervisor restarts a crashed vLLM, so
    the container stays up through crash loops and hangs; the deadline catches those.
    """
    model_cfg = request.param
    container = os.environ.get("SM_LOCAL_CONTAINER")
    t0 = time.time()
    deadline = time.monotonic() + STARTUP_TIMEOUT
    last = "no response"
    while time.monotonic() < deadline:
        try:
            status, _ = _send(model_cfg, "GET", "/ping", timeout=PING_INTERVAL)
            if status == 200:
                LOGGER.info(f"{model_cfg['name']} ready after {time.time() - t0:.0f}s")
                return model_cfg
            last = f"HTTP {status}"
        except OSError as e:
            last = str(e)
        if container:
            _fail_unless_running(container)
        time.sleep(PING_INTERVAL)
    pytest.fail(
        f"{model_cfg['name']}: /ping did not return 200 within {STARTUP_TIMEOUT}s "
        f"(last: {last}); see the Dump container logs step",
        pytrace=False,
    )


def test_local_serving(serving_model):
    model_cfg = serving_model
    region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
    s3_client = boto3.client("s3", region_name=region)

    test_cases = model_cfg["test_cases"]
    failed = []

    for tc in test_cases:
        tc_name = tc["name"]
        LOGGER.info(f"--- Running: {tc_name} ---")

        content_type = tc.get("content_type", "application/json")
        request_cfg = tc["request"]

        if content_type.startswith("multipart/form-data"):
            body, ct = build_multipart_body(request_cfg, s3_client)
        else:
            body, ct = build_json_body(request_cfg)

        headers = {"Content-Type": ct}
        if "route" in tc:
            headers["X-Amzn-SageMaker-Custom-Attributes"] = f"route={tc['route']}"

        t0 = time.time()
        try:
            status, raw_bytes = _send(model_cfg, "POST", "/invocations", body, headers)
            elapsed = time.time() - t0
            LOGGER.info(f"  Response ({elapsed:.1f}s, HTTP {status}, {len(raw_bytes)} bytes)")
            # SageMaker turns a non-2xx container response into a ModelError.
            assert 200 <= status < 300, (
                f"HTTP {status}: {raw_bytes[:500].decode('utf-8', errors='replace')}"
            )

            try:
                response_body = json.loads(raw_bytes)
            except (json.JSONDecodeError, UnicodeDecodeError):
                response_body = None
                LOGGER.info(f"  Non-JSON body: {raw_bytes[:300].decode('utf-8', errors='replace')}")

            if response_body:
                LOGGER.info(f"  Body: {json.dumps(response_body)[:300]}")

            validate_response(response_body, raw_bytes, tc["validate"])
            LOGGER.info(f"  PASS: {tc_name}")

        except Exception as e:
            elapsed = time.time() - t0
            LOGGER.error(f"  FAIL: {tc_name} ({elapsed:.1f}s) — {e}")
            failed.append((tc_name, str(e)))

    if failed:
        msg = "\n".join(f"  - {name}: {err}" for name, err in failed)
        pytest.fail(f"{len(failed)}/{len(test_cases)} test cases failed:\n{msg}")
