"""CPU-only tests for the AutoGluon SageMaker server."""

from __future__ import annotations

import importlib.util
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

SERVER_DIR = Path(__file__).resolve().parents[3] / "scripts" / "docker" / "autogluon"
SERVER_PATH = SERVER_DIR / "server.py"
SETTINGS_PATH = SERVER_DIR / "settings.py"
INSTALL_REQUIREMENTS_PATH = SERVER_DIR / "install_requirements.py"
GUNICORN_CONFIG_PATH = SERVER_DIR / "gunicorn.conf.py"

SERVING_ENVIRONMENT_VARIABLES = {
    "SAGEMAKER_PROGRAM",
    "SAGEMAKER_BIND_TO_PORT",
    "SAGEMAKER_MODEL_SERVER_WORKERS",
    "SAGEMAKER_MODEL_SERVER_TIMEOUT",
    "SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT",
    "SAGEMAKER_CONTAINER_LOG_LEVEL",
    "SAGEMAKER_BASE_DIR",
    "CA_REPOSITORY_ARN",
}


def _load(path: Path, prefix: str):
    module_name = f"{prefix}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _clear_serving_environment(monkeypatch):
    for name in SERVING_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def _write_handler(base_dir: Path, source: str, program: str = "inference.py") -> Path:
    model_dir = base_dir / "model"
    code_dir = model_dir / "code"
    code_dir.mkdir(parents=True)
    (code_dir / program).write_text(source)
    return model_dir


def _load_server(monkeypatch, base_dir: Path, program: str | None = None):
    _clear_serving_environment(monkeypatch)
    monkeypatch.setenv("SAGEMAKER_BASE_DIR", str(base_dir))
    if program is not None:
        monkeypatch.setenv("SAGEMAKER_PROGRAM", program)
    monkeypatch.syspath_prepend(str(SERVER_DIR))
    return _load(SERVER_PATH, "autogluon_server")


def _request(app, method: str, path: str, **kwargs):
    with app.test_client() as client:
        return client.open(path, method=method, **kwargs)


def test_model_artifact_handler_receives_normalized_json_headers(monkeypatch, tmp_path):
    model_dir = _write_handler(
        tmp_path,
        """
import json

def model_fn(model_dir):
    return {"model_dir": model_dir}

def transform_fn(model, body, content_type, accept):
    return json.dumps({
        "body": body,
        "content_type": content_type,
        "accept": accept,
        "model_dir": model["model_dir"],
    }), "application/json"
""",
    )
    server = _load_server(monkeypatch, tmp_path)

    assert _request(server.app, "GET", "/ping").data == b""
    response = _request(
        server.app,
        "POST",
        "/invocations",
        data='[{"feature": 1}]',
        headers={
            "content-type": "application/json; charset=utf-8",
            "accept": "application/json, */*",
        },
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "body": '[{"feature": 1}]',
        "content_type": "application/json",
        "accept": "application/json",
        "model_dir": str(model_dir),
    }


def test_binary_request_and_response_are_not_decoded(monkeypatch, tmp_path):
    _write_handler(
        tmp_path,
        """
def model_fn(model_dir):
    return None

def transform_fn(model, body, content_type, accept):
    assert isinstance(body, bytes)
    assert content_type == "application/x-parquet"
    return body[::-1], "application/x-parquet"
""",
    )
    server = _load_server(monkeypatch, tmp_path)

    response = _request(
        server.app,
        "POST",
        "/invocations",
        data=b"parquet-bytes",
        headers={"content-type": "application/x-parquet", "accept": "application/x-parquet"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-parquet"
    assert response.data == b"setyb-teuqrap"


def test_bytearray_response_is_supported(monkeypatch, tmp_path):
    _write_handler(
        tmp_path,
        """
def model_fn(model_dir):
    return None

def transform_fn(model, body, content_type, accept):
    return bytearray(b"result"), "application/octet-stream"
""",
    )
    server = _load_server(monkeypatch, tmp_path)

    response = _request(server.app, "POST", "/invocations", data=b"input")

    assert response.status_code == 200
    assert response.data == b"result"


def test_body_only_transform_response_uses_accept_header(monkeypatch, tmp_path):
    _write_handler(
        tmp_path,
        """
def model_fn(model_dir):
    return None

def transform_fn(model, body, content_type, accept):
    return b"result"
""",
    )
    server = _load_server(monkeypatch, tmp_path)

    response = _request(
        server.app,
        "POST",
        "/invocations",
        data=b"input",
        headers={"accept": "application/octet-stream"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.data == b"result"


def test_default_accept_environment_variable(monkeypatch, tmp_path):
    _write_handler(
        tmp_path,
        """
def model_fn(model_dir):
    return None

def transform_fn(model, body, content_type, accept):
    return accept, "text/plain"
""",
    )
    _clear_serving_environment(monkeypatch)
    monkeypatch.setenv("SAGEMAKER_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT", "text/csv")
    monkeypatch.syspath_prepend(str(SERVER_DIR))
    server = _load(SERVER_PATH, "autogluon_server")

    response = _request(server.app, "POST", "/invocations", data=b"input")

    assert response.status_code == 200
    assert response.text == "text/csv"


def test_sagemaker_program_selects_custom_handler(monkeypatch, tmp_path):
    _write_handler(
        tmp_path,
        """
def model_fn(model_dir):
    return model_dir

def transform_fn(model, body, content_type, accept):
    return model, "text/plain"
""",
        program="tabular_serve.py",
    )
    server = _load_server(monkeypatch, tmp_path, program="tabular_serve.py")

    response = _request(server.app, "POST", "/invocations", data=b"input")

    assert response.status_code == 200
    assert response.text == str(tmp_path / "model")


def test_input_predict_output_handler_pipeline(monkeypatch, tmp_path):
    _write_handler(
        tmp_path,
        """
def model_fn(model_dir):
    return {"suffix": "!"}

def input_fn(body, content_type):
    assert content_type == "text/plain"
    return body.upper()

def predict_fn(data, model):
    return data + model["suffix"]

def output_fn(prediction, accept):
    assert accept == "text/plain"
    return prediction, accept
""",
    )
    server = _load_server(monkeypatch, tmp_path)

    response = _request(
        server.app,
        "POST",
        "/invocations",
        data="hello",
        headers={"content-type": "text/plain", "accept": "text/plain"},
    )

    assert response.status_code == 200
    assert response.text == "HELLO!"


def test_transform_handler_cannot_mix_pipeline_functions(monkeypatch, tmp_path):
    _write_handler(
        tmp_path,
        """
def model_fn(model_dir):
    return None

def transform_fn(model, body, content_type, accept):
    return body

def input_fn(body, content_type):
    return body
""",
    )

    with pytest.raises(RuntimeError, match="transform_fn cannot be combined"):
        _load_server(monkeypatch, tmp_path)


def test_handler_value_error_is_a_client_error(monkeypatch, tmp_path):
    _write_handler(
        tmp_path,
        """
def model_fn(model_dir):
    return None

def transform_fn(model, body, content_type, accept):
    raise ValueError("unsupported payload")
""",
    )
    server = _load_server(monkeypatch, tmp_path)

    response = _request(server.app, "POST", "/invocations", data=b"bad")

    assert response.status_code == 400
    assert response.get_json()["detail"] == "unsupported payload"


def test_settings_preserve_only_agreed_environment_variables(monkeypatch, tmp_path):
    _clear_serving_environment(monkeypatch)
    monkeypatch.setenv("SAGEMAKER_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("SAGEMAKER_PROGRAM", "custom.py")
    monkeypatch.setenv("SAGEMAKER_BIND_TO_PORT", "9000")
    monkeypatch.setenv("SAGEMAKER_MODEL_SERVER_WORKERS", "3")
    monkeypatch.setenv("SAGEMAKER_MODEL_SERVER_TIMEOUT", "120")
    monkeypatch.setenv("SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT", "text/csv")
    monkeypatch.setenv("SAGEMAKER_CONTAINER_LOG_LEVEL", "10")
    monkeypatch.setenv(
        "CA_REPOSITORY_ARN",
        "arn:aws:codeartifact:us-west-2:123456789012:repository/domain/repository",
    )
    # Deliberately unsupported aliases must not affect the configuration.
    monkeypatch.setenv("SAGEMAKER_MODEL_DIR", "/ignored")
    monkeypatch.setenv("SAGEMAKER_NUM_MODEL_WORKERS", "99")
    monkeypatch.syspath_prepend(str(SERVER_DIR))

    settings_module = _load(SETTINGS_PATH, "autogluon_settings")
    settings = settings_module.Settings.from_environment()
    gunicorn_config = _load(GUNICORN_CONFIG_PATH, "autogluon_gunicorn_config")

    assert settings.model_dir == tmp_path / "model"
    assert settings.handler_path == tmp_path / "model" / "code" / "custom.py"
    assert settings.default_accept == "text/csv"
    assert settings.codeartifact_repository_arn.endswith("repository/domain/repository")
    assert gunicorn_config.bind == "0.0.0.0:9000"
    assert gunicorn_config.workers == 3
    assert gunicorn_config.timeout == 120
    assert gunicorn_config.graceful_timeout == 120
    assert gunicorn_config.loglevel == "debug"
    assert gunicorn_config.worker_class == "sync"
    assert gunicorn_config.threads == 1
    assert gunicorn_config.preload_app is False


def test_requirements_installed_once_with_codeartifact_index(monkeypatch, tmp_path):
    model_dir = _write_handler(
        tmp_path,
        """
def model_fn(model_dir):
    return None

def transform_fn(model, body, content_type, accept):
    return body
""",
    )
    requirements = model_dir / "code" / "requirements.txt"
    requirements.write_text("customer-package==1.2.3\n")
    monkeypatch.syspath_prepend(str(SERVER_DIR))
    installer = _load(INSTALL_REQUIREMENTS_PATH, "autogluon_install_requirements")
    settings_module = _load(SETTINGS_PATH, "autogluon_settings")
    settings = settings_module.Settings.from_environment(
        {
            "SAGEMAKER_BASE_DIR": str(tmp_path),
            "CA_REPOSITORY_ARN": (
                "arn:aws:codeartifact:us-west-2:123456789012:repository/domain/repository"
            ),
        }
    )
    monkeypatch.setattr(
        installer,
        "_codeartifact_index",
        lambda arn: "https://aws:secret@example.com/pypi/repository/simple/",
    )
    calls = []
    monkeypatch.setattr(
        installer.subprocess,
        "check_call",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    installer.install_requirements(settings)

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--no-cache",
        "-r",
        str(requirements),
    ]
    assert "secret" not in " ".join(command)
    assert kwargs["env"]["UV_INDEX_URL"] == (
        "https://aws:secret@example.com/pypi/repository/simple/"
    )


def test_codeartifact_arn_is_resolved_without_logging_credentials(monkeypatch):
    monkeypatch.syspath_prepend(str(SERVER_DIR))
    installer = _load(INSTALL_REQUIREMENTS_PATH, "autogluon_install_requirements")
    calls = []

    class CodeArtifactClient:
        def get_authorization_token(self, **kwargs):
            calls.append(("token", kwargs))
            return {"authorizationToken": "token+/="}

        def get_repository_endpoint(self, **kwargs):
            calls.append(("endpoint", kwargs))
            return {"repositoryEndpoint": "https://example.com/pypi/repository/"}

    boto3 = types.SimpleNamespace(
        client=lambda service, region_name: (
            calls.append(("client", service, region_name)) or CodeArtifactClient()
        )
    )
    monkeypatch.setitem(sys.modules, "boto3", boto3)

    index = installer._codeartifact_index(
        "arn:aws:codeartifact:us-west-2:123456789012:repository/domain/repository"
    )

    assert index == "https://aws:token%2B%2F%3D@example.com/pypi/repository/simple/"
    assert calls == [
        ("client", "codeartifact", "us-west-2"),
        (
            "token",
            {"domain": "domain", "domainOwner": "123456789012"},
        ),
        (
            "endpoint",
            {
                "domain": "domain",
                "domainOwner": "123456789012",
                "repository": "repository",
                "format": "pypi",
            },
        ),
    ]


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _wait_for_server(process: subprocess.Popen, port: int) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"Gunicorn exited during startup:\n{stdout}\n{stderr}")
        try:
            if httpx.get(f"http://127.0.0.1:{port}/ping", timeout=0.2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    pytest.fail("Gunicorn did not become ready")


def test_one_sync_worker_serializes_inference_requests(tmp_path):
    gunicorn = shutil.which("gunicorn")
    assert gunicorn is not None
    events_path = tmp_path / "events.txt"
    _write_handler(
        tmp_path,
        """
import os
import time

EVENTS_PATH = os.environ["AUTOGLOON_TEST_EVENTS_PATH"]

def model_fn(model_dir):
    return None

def transform_fn(model, body, content_type, accept):
    with open(EVENTS_PATH, "a") as events:
        events.write(f"start {body}\\n")
    time.sleep(0.4)
    with open(EVENTS_PATH, "a") as events:
        events.write(f"end {body}\\n")
    return body, "text/plain"
""",
    )
    port = _free_port()
    environment = {
        key: value for key, value in os.environ.items() if key not in SERVING_ENVIRONMENT_VARIABLES
    }
    environment.update(
        {
            "SAGEMAKER_BASE_DIR": str(tmp_path),
            "SAGEMAKER_BIND_TO_PORT": str(port),
            "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
            "SAGEMAKER_MODEL_SERVER_TIMEOUT": "10",
            "AUTOGLOON_TEST_EVENTS_PATH": str(events_path),
        }
    )
    process = subprocess.Popen(
        [
            gunicorn,
            "--config",
            str(GUNICORN_CONFIG_PATH),
            "--chdir",
            str(SERVER_DIR),
            "server:app",
        ],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_for_server(process, port)
        barrier = threading.Barrier(3)

        def invoke(label: str):
            barrier.wait()
            return httpx.post(
                f"http://127.0.0.1:{port}/invocations",
                content=label,
                headers={"content-type": "text/plain", "accept": "text/plain"},
                timeout=5,
            )

        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(invoke, "first")
            second = executor.submit(invoke, "second")
            barrier.wait()
            responses = [first.result(), second.result()]
        elapsed = time.monotonic() - started

        assert [response.status_code for response in responses] == [200, 200]
        assert {response.text for response in responses} == {"first", "second"}
        assert elapsed >= 0.7

        events = events_path.read_text().splitlines()
        assert len(events) == 4
        assert events[0].startswith("start ")
        first_label = events[0].removeprefix("start ")
        assert events[1] == f"end {first_label}"
        assert events[2].startswith("start ")
        second_label = events[2].removeprefix("start ")
        assert second_label != first_label
        assert events[3] == f"end {second_label}"
    finally:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
