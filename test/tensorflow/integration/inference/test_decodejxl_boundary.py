"""TFS 2.20 x TF 2.21 SavedModel forward-compat boundary test — DecodeJxl.

The v2 inference image ships ``tensorflow_model_server`` 2.20 (linked
against TF 2.20's kernel registry) while customers on TF >= 2.21 can
author SavedModels that include ops added in TF 2.21. ``DecodeJxl``
(JPEG XL image decoder, ``tf.raw_ops.DecodeJxl``) is the one such
op that has no TF 2.20 equivalent in the base op set — TFS 2.20's
kernel registry does not know about it.

Failure signature is uniquely nasty:

  * ``Model.create`` succeeds (SavedModel parses fine — proto-level
    validation only checks the op name / attr shape, not that a kernel
    is registered for it).
  * ``Endpoint.create`` reports ``InService`` — TFS boots, loads the
    graph, exposes the ``/models/model`` status endpoint reporting
    ``AVAILABLE``.
  * First customer ``predict`` returns HTTP 400 with body
    ``"Op type not registered 'DecodeJxl'"``.

A health-check-based rollout gate (``/ping`` returns 200, model status
is ``AVAILABLE``) would go green and the customer would take the
traffic error. This test pins that boundary in code so a future TFS
upgrade that gains DecodeJxl support flips it green — and a regression
where TFS silently gains kernel-not-registered *acceptance* (e.g.
returning 2xx with an empty tensor) is caught by the 4xx assertion.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import botocore.exceptions
import pytest

from .resources.helpers import upload_tarball


def _build_decodejxl_model(saved_model_dir: str) -> None:
    """Export a SavedModel whose serving signature invokes DecodeJxl.

    The module wraps ``tf.raw_ops.DecodeJxl`` in a signature-exposed
    method taking a scalar ``tf.string`` input. Split out so the test
    body stays focused on the deploy/invoke boundary.
    """
    import tensorflow as tf

    class _DecodeJxlWrapper(tf.Module):
        """Serves a DecodeJxl call. TFS 2.20 does not know this op — the
        model exports fine but the first predict returns 4xx."""

        @tf.function(input_signature=[tf.TensorSpec([], dtype=tf.string)])
        def serve(self, jxl_bytes):
            return {"image": tf.raw_ops.DecodeJxl(contents=jxl_bytes)}

    model = _DecodeJxlWrapper()
    tf.saved_model.save(
        model,
        saved_model_dir,
        signatures={"serving_default": model.serve},
    )


def _package_tarball(model_root: Path, tar_filename: str = "model.tar.gz") -> str:
    """Package ``<model_root>/1/saved_model.pb`` into ``model.tar.gz`` in
    the SageMaker TFS layout (top-level numeric version directory).

    Same shape as ``build_sample_model._package_saved_model_tarball``.
    Inlined here (rather than reusing it) because that helper writes to
    ``<output_dir>/<version>`` from a specific caller shape and this
    test needs to control the SavedModel export directory directly.
    """
    import tarfile

    tar_path = model_root / tar_filename
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(str(model_root / "1"), arcname="1")
    return str(tar_path)


def test_decodejxl_op_not_registered_on_tfs_2_20(
    sagemaker_session,
    deploy_endpoint,
    unique_name,
    cleanup_endpoint,
):
    """Deploy a SavedModel containing DecodeJxl and invoke it.

    Expected: 4xx with ``"Op type not registered"`` in the body. If this
    ever flips to 2xx, TFS silently gained DecodeJxl support (or worse,
    silently accepted a kernel-missing op). Either way, we want a
    controlled failure at CI time, not at the customer.
    """
    # DecodeJxl is only available on TF >= 2.21. Skip when the runner is
    # on an older TF than that (rather than fail confusingly on module
    # attribute lookup). Not "skip silently on any TF" — we need this
    # to run against TF 2.21+ workers to prove the boundary.
    tf = pytest.importorskip("tensorflow")
    if not hasattr(tf.raw_ops, "DecodeJxl"):
        pytest.skip(
            "tf.raw_ops.DecodeJxl not present on this test host; needs TF 2.21+ "
            "to author the boundary SavedModel"
        )

    with tempfile.TemporaryDirectory(prefix="tf220-decodejxl-") as workdir:
        model_root = Path(workdir)
        saved_model_dir = model_root / "1"
        saved_model_dir.mkdir(parents=True, exist_ok=True)
        _build_decodejxl_model(str(saved_model_dir))
        tar_path = _package_tarball(model_root)

        model_data = upload_tarball(
            sagemaker_session,
            tar_path,
            key_prefix=f"tf220-inference-tests/decodejxl/{unique_name('run')}",
        )
        endpoint, endpoint_name, model_name = deploy_endpoint(
            model_data_url=model_data,
            name_prefix="tf220-decodejxl",
        )
        cleanup_endpoint(endpoint_name, model_name=model_name)

        # Dummy JXL header bytes. The decode itself doesn't need to
        # succeed — the request only needs to reach the DecodeJxl op so
        # TFS raises "Op type not registered" before any decode logic
        # runs.
        with pytest.raises(botocore.exceptions.ClientError) as excinfo:
            endpoint.invoke(
                body=b"\xff\x0a\x00\x00",  # JXL container signature bytes
                content_type="application/octet-stream",
                accept="application/json",
            )

        response = excinfo.value.response
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        body_obj = response.get("Body", b"")
        if hasattr(body_obj, "read"):
            body_obj = body_obj.read()
        if isinstance(body_obj, bytes):
            body_text = body_obj.decode("utf-8", errors="replace")
        else:
            body_text = str(response)

        assert 400 <= status < 500, (
            f"expected 4xx from TFS on unregistered DecodeJxl op, got status {status}: {response!r}"
        )
        # Loose match — TFS's exact error phrasing has drifted across
        # versions. Any of these three markers proves the request
        # reached the op-lookup path and was correctly rejected.
        markers = ("DecodeJxl", "Op type not registered", "not registered")
        assert any(m in body_text for m in markers), (
            f"expected an 'op not registered' surface in the response body, got: {body_text!r}"
        )
