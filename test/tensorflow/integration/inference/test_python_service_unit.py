"""Unit tests for MME lifecycle endpoints in ``python_service.py``.

Why unit tests rather than SageMaker integration for DELETE/GET:

    The SageMaker Runtime data-plane in front of the container does NOT
    expose the MME lifecycle APIs (``DELETE /models/{name}``,
    ``GET /models[/{name}]``) to external callers. Those routes are only
    reachable from the on-host SM Model Manager sidecar. Driving eviction
    externally via LRU alone would require piling on models until memory
    pressure trips it — both slow (~5 min/model on ``ml.c5.xlarge``) and
    non-deterministic (eviction order depends on runtime memory + timing).

    Falcon-level unit tests hit the handler methods directly with a
    bare-bones ``PythonServiceResource`` — bypassing ``__init__`` (which
    starts gRPC channels and parses env-driven port ranges) via
    ``object.__new__``. Only the immediate boundary is mocked:
    ``os.kill`` / ``shutil.rmtree`` / ``_upload_mme_instance_status``
    (pickle status file) for DELETE, ``requests.get`` for GET, and
    ``_load_model`` (which shells out to TFS) for the reload path.
    Handler control flow itself runs unaltered.

Test coverage:

    * ``test_mme_delete_model_unloads_and_reloads`` — DELETE clears state,
      subsequent load is treated as fresh (200, not 409).
    * ``test_mme_get_returns_loaded_model`` — GET /models/{name} on a
      loaded model returns 200 with upstream TFS payload passed through.
    * ``test_mme_get_missing_model_returns_404`` — GET on an unknown model
      returns 404 with a JSON error body.
    * ``test_mme_traversal_rejected_by_handler_guard`` — traversal-style
      model_name values are rejected 400 by the guard at lines ~275-292 of
      ``python_service.py``. Complements the SM-level test in
      ``test_mme_dynamic.py::test_mme_traversal_rejected``.
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Handler modules live in the DLC scripts tree, not under the test root.
# Path layout: <workspace>/test/tensorflow/integration/inference/<this file>
# and <workspace>/scripts/docker/tensorflow/inference/sagemaker/.
_HANDLER_DIR = (
    Path(__file__).resolve().parents[4]
    / "scripts"
    / "docker"
    / "tensorflow"
    / "inference"
    / "sagemaker"
)
if str(_HANDLER_DIR) not in sys.path:
    sys.path.insert(0, str(_HANDLER_DIR))

# ``python_service`` instantiates ``ServiceResources()`` at import time,
# which pulls port/mode from env vars. Set MME-mode + a port range so the
# import succeeds outside the DLC container. Individual tests bypass
# ``__init__`` via ``object.__new__`` so these values are only used by the
# module-level singleton — not by the objects under test.
os.environ.setdefault("SAGEMAKER_MULTI_MODEL", "true")
os.environ.setdefault("SAGEMAKER_SAFE_PORT_RANGE", "8500-9000")

# ``python_service`` calls ``gevent.monkey.patch_all()`` at import time.
# That mutates the interpreter's stdlib, but only inside the pytest worker;
# other test files are unaffected. Skip cleanly if deps are missing on
# developer laptops (e.g. no gevent / grpc installed locally).
try:
    import python_service  # type: ignore  # noqa: E402
except Exception as exc:  # pragma: no cover - env-dependent
    pytest.skip(f"handler module not importable ({exc})", allow_module_level=True)

import falcon  # noqa: E402  (safe: python_service already imports falcon)


def _make_resource(loaded_models: dict | None = None):
    """Instantiate ``PythonServiceResource`` without going through ``__init__``.

    ``__init__`` requires ``SAGEMAKER_MULTI_MODEL`` /
    ``SAGEMAKER_SAFE_PORT_RANGE`` env vars, gRPC channel creation, and a
    handful of paths that only exist on a real DLC. ``object.__new__`` +
    explicit attribute setup gives us just the state the endpoints under
    test actually read.
    """
    r = object.__new__(python_service.PythonServiceResource)
    r._mme_tfs_instances_status = dict(loaded_models or {})
    r._tfs_ports = {"rest_port": [8501], "grpc_port": [8500]}
    r._tfs_available_ports = {"rest_port": [8501], "grpc_port": [8500]}
    r.model_handlers = {}
    r._tfs_instance_count = 1
    r._tfs_default_model_name = "None"
    r._gunicorn_workers = 1
    r._tfs_enable_batching = False
    r._handlers = lambda data, ctx: (b"", "application/json")
    r._default_handlers_enabled = True
    return r


def _fake_response() -> SimpleNamespace:
    """Minimal stand-in for a Falcon ``Response`` — status + body + content_type."""
    return SimpleNamespace(status=None, body=None, content_type=None)


def _body_text(res: SimpleNamespace) -> str:
    """Return ``res.body`` as text regardless of str/bytes/bytearray."""
    body = res.body
    if isinstance(body, (bytes, bytearray)):
        return bytes(body).decode("utf-8", errors="replace")
    return "" if body is None else str(body)


# ---------------------------------------------------------------------------
# Test A: DELETE /models/{name} unloads and the model can be reloaded
# ---------------------------------------------------------------------------


def test_mme_delete_model_unloads_and_reloads():
    """DELETE /models/{name} must remove the model from
    ``_mme_tfs_instances_status`` and leave state consistent enough for a
    subsequent ``_handle_load_model_post`` to succeed with 200 (not 409).

    Mechanism choice: (c) Falcon-level unit test.
    Rationale:
      (a) No SageMaker Runtime API exists to unload a specific MME model —
          the Model Manager sidecar owns lifecycle.
      (b) LRU eviction is possible but requires loading enough models to
          trip memory pressure — slow (~5 min/model) and non-deterministic.
      (c) Direct handler test is fast, deterministic, and exercises the
          exact code path (on_delete → _delete_model → _remove_model_config
          → state clear → upload status).
    """
    tfs = python_service.TfsInstanceStatus(rest_port=8501, grpc_port=8500, pid=1234)
    resource = _make_resource(loaded_models={"modelA": [tfs]})

    with (
        patch("python_service.os.kill") as kill_mock,
        patch("python_service.shutil.rmtree") as rmtree_mock,
        patch.object(resource, "_upload_mme_instance_status") as upload_mock,
        patch.object(resource, "_sync_local_mme_instance_status"),
    ):
        res = _fake_response()
        resource.on_delete(req=MagicMock(), res=res, model_name="modelA")

    assert res.status == falcon.HTTP_200, (
        f"DELETE modelA expected 200, got status={res.status!r} body={_body_text(res)!r}"
    )
    assert "modelA" not in resource._mme_tfs_instances_status, (
        "on_delete failed to remove modelA from the loaded-model registry"
    )
    kill_mock.assert_called_once_with(1234, signal.SIGKILL)
    rmtree_mock.assert_any_call("/sagemaker/tfs-config/modelA", ignore_errors=True)
    rmtree_mock.assert_any_call("/sagemaker/batching/modelA", ignore_errors=True)
    upload_mock.assert_called_once()

    # Re-load modelA — state was cleared, so this is a fresh load path
    # (200), not "already loaded" (409). Mock the subprocess boundary
    # (_load_model spawns TFS via Popen).
    load_response = {
        "status": falcon.HTTP_200,
        "body": '{"success": "loaded modelA"}',
        "pid": 5678,
    }
    with (
        patch.object(resource, "_load_model", return_value=load_response) as load_mock,
        patch.object(resource, "_upload_mme_instance_status"),
        patch.object(resource, "_sync_local_mme_instance_status"),
        patch.object(resource, "_sync_model_handlers"),
    ):
        res2 = _fake_response()
        resource._handle_load_model_post(
            res=res2,
            data={"model_name": "modelA", "url": "/opt/ml/models/modelA"},
        )

    assert res2.status == falcon.HTTP_200, (
        f"Re-loading modelA after DELETE should return 200, got status={res2.status!r} "
        f"body={_body_text(res2)!r}"
    )
    assert "modelA" in resource._mme_tfs_instances_status, (
        "reload succeeded but state was not repopulated with modelA"
    )
    load_mock.assert_called_once()


# ---------------------------------------------------------------------------
# Test B: GET /models/{name} returns 200 with TFS payload passed through
# ---------------------------------------------------------------------------


def test_mme_get_returns_loaded_model():
    """GET /models/{name} on a loaded model must return 200 with the
    upstream TFS status payload embedded in the response body.

    Same mechanism choice / rationale as ``test_mme_delete_model_unloads_and_reloads``:
    the SM Runtime does not expose GET /models to external callers, so a
    Falcon-level unit test is the direct way to prove the container-side
    handler is correct.
    """
    tfs = python_service.TfsInstanceStatus(rest_port=8501, grpc_port=8500, pid=1234)
    resource = _make_resource(loaded_models={"modelA": [tfs]})

    # TFS's /v1/models/{name} returns a JSON status blob. Mock at the
    # requests.get boundary — the handler's own logic runs unaltered.
    fake_tfs_body = {"modelName": "modelA", "state": "AVAILABLE"}
    with (
        patch("python_service.requests.get", return_value=fake_tfs_body) as get_mock,
        patch.object(resource, "_sync_local_mme_instance_status"),
    ):
        res = _fake_response()
        resource.on_get(req=MagicMock(), res=res, model_name="modelA")

    assert res.status == falcon.HTTP_200, (
        f"GET /models/modelA expected 200, got status={res.status!r} body={_body_text(res)!r}"
    )
    body_str = _body_text(res)
    assert "modelName" in body_str, (
        f"expected TFS payload passed through to include 'modelName', got: {body_str!r}"
    )
    assert "modelA" in body_str
    get_mock.assert_called_once()
    # And the URL must target the rest_port bound to modelA.
    called_url = get_mock.call_args.args[0]
    assert "8501" in called_url and "modelA" in called_url, (
        f"GET forwarded to unexpected TFS URL: {called_url!r}"
    )


def test_mme_get_missing_model_returns_404():
    """GET /models/{name} for a model that isn't loaded returns 404 with
    a JSON error body — not a hang or 500."""
    resource = _make_resource(loaded_models={})
    with patch.object(resource, "_sync_local_mme_instance_status"):
        res = _fake_response()
        resource.on_get(req=MagicMock(), res=res, model_name="does_not_exist")

    assert res.status == falcon.HTTP_404, (
        f"GET on missing model expected 404, got status={res.status!r}"
    )
    body_str = _body_text(res)
    assert "not loaded" in body_str.lower(), f"expected error body, got: {body_str!r}"


# ---------------------------------------------------------------------------
# Test C: Traversal guard on model_name (defence-in-depth)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_model_name",
    [
        "../../evil.tar.gz",
        "/etc/passwd",
        "..",
        ".",
        "./secret",
        ".hidden",
        "with\x00null",
        "",
    ],
    ids=[
        "parent-traversal",
        "absolute-path",
        "dot-dot",
        "dot",
        "dot-slash-prefix",
        "hidden-dotfile",
        "embedded-null",
        "empty",
    ],
)
def test_mme_traversal_rejected_by_handler_guard(bad_model_name):
    """Direct unit test on the traversal guard in
    ``_handle_load_model_post`` (python_service.py lines ~275-292).

    Even though the SM MME data-plane may reject some patterns client-side
    before they reach the container, this test proves the container-side
    guard exists as defence-in-depth. If the guard is deleted or its
    predicate weakens, this test fails immediately — a container-side
    regression would otherwise be invisible outside a full end-to-end
    penetration test.
    """
    resource = _make_resource()
    res = _fake_response()
    resource._handle_load_model_post(
        res=res,
        data={"model_name": bad_model_name, "url": "/opt/ml/models/whatever"},
    )
    assert res.status == falcon.HTTP_400, (
        f"traversal input {bad_model_name!r} was accepted (status={res.status!r})"
    )
    body_str = _body_text(res)
    assert "invalid model_name" in body_str, (
        f"expected guard error message for {bad_model_name!r}, got body: {body_str!r}"
    )
