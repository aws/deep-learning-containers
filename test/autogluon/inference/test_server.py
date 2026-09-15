"""CPU-only tests for the AutoGluon SageMaker server."""

from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[3] / "scripts" / "docker" / "autogluon"
SERVER_PATH = SERVER_DIR / "server.py"
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


def test_gunicorn_environment_configuration(monkeypatch):
    _clear_serving_environment(monkeypatch)
    monkeypatch.setenv("SAGEMAKER_BIND_TO_PORT", "9000")
    monkeypatch.setenv("SAGEMAKER_MODEL_SERVER_WORKERS", "3")
    monkeypatch.setenv("SAGEMAKER_MODEL_SERVER_TIMEOUT", "120")
    monkeypatch.setenv("SAGEMAKER_CONTAINER_LOG_LEVEL", "10")

    gunicorn_config = _load(GUNICORN_CONFIG_PATH, "autogluon_gunicorn_config")

    assert gunicorn_config.bind == "0.0.0.0:9000"
    assert gunicorn_config.workers == 3
    assert gunicorn_config.timeout == 120
    assert gunicorn_config.graceful_timeout == 120
    assert gunicorn_config.loglevel == "debug"
    assert gunicorn_config.worker_class == "sync"


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
    _clear_serving_environment(monkeypatch)
    monkeypatch.setenv("SAGEMAKER_BASE_DIR", str(tmp_path))
    monkeypatch.setenv(
        "CA_REPOSITORY_ARN",
        "arn:aws:codeartifact:us-west-2:123456789012:repository/domain/repository",
    )
    monkeypatch.syspath_prepend(str(SERVER_DIR))
    installer = _load(INSTALL_REQUIREMENTS_PATH, "autogluon_install_requirements")
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

    installer.install_requirements()

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
