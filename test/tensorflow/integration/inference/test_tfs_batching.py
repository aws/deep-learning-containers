"""TFS batching-config integration test for TF 2.20 inference DLC.

Deploys an endpoint with ``SAGEMAKER_TFS_ENABLE_BATCHING=true`` (and the
knobs the DLC handler forwards to TFS's ``--batching_parameters_file``)
and verifies the endpoint still returns numerically-correct predictions —
i.e. the DLC handler wired the batching parameters through without
breaking the request path.

Direct verification of "TFS actually batched the requests" would require
reading the TFS process stdout, which is not accessible from a SageMaker
managed endpoint. This test covers the wire contract; a deeper end-to-end
assertion (measured throughput inflection under load) belongs in a
performance benchmark, not integration tests. Covers audit finding G6.
"""

from __future__ import annotations

import json
import tempfile

import pytest

from .resources.build_sample_model import build_sample_model
from .resources.helpers import read_predictions, upload_tarball


def test_tfs_batching_enabled_wire_contract(
    sagemaker_session,
    deploy_endpoint,
    unique_name,
    cleanup_endpoint,
):
    """Batching env vars set → endpoint still responds correctly.

    The env-var names are read by ``serve.py`` in the container:
    ``SAGEMAKER_TFS_ENABLE_BATCHING`` toggles the ``--enable_batching`` TFS
    flag; the other knobs feed ``batching_parameters_file`` (see
    ``tfs_utils.py::create_tfs_config_file`` / ``batching_parameters``)."""
    with tempfile.TemporaryDirectory(prefix="tf220-batching-") as workdir:
        tar_path = build_sample_model(output_dir=workdir, multiplier=2.0)
        model_data = upload_tarball(
            sagemaker_session,
            tar_path,
            key_prefix=f"tf220-inference-tests/batching/{unique_name('run')}",
        )
        endpoint, endpoint_name, model_name = deploy_endpoint(
            model_data_url=model_data,
            name_prefix="tf220-batching",
            container_env={
                "SAGEMAKER_TFS_ENABLE_BATCHING": "true",
                # Small knobs so batching engages promptly on the tiny
                # request rate this test generates. Real customer defaults
                # depend on their throughput profile.
                "SAGEMAKER_TFS_MAX_BATCH_SIZE": "8",
                "SAGEMAKER_TFS_BATCH_TIMEOUT_MICROS": "1000",
                "SAGEMAKER_TFS_NUM_BATCH_THREADS": "1",
                "SAGEMAKER_TFS_MAX_ENQUEUED_BATCHES": "10000",
            },
        )
        cleanup_endpoint(endpoint_name, model_name=model_name)

        payload = json.dumps({"instances": [[1.0, 2.0, 3.0]]})
        result = endpoint.invoke(
            body=payload,
            content_type="application/json",
            accept="application/json",
        )
        # If batching env vars broke the request path (e.g. malformed
        # batching_parameters_file → TFS exits at model load), we'd see 5xx
        # from InvokeEndpoint before ever reaching this assertion.
        assert read_predictions(result) == pytest.approx([2.0, 4.0, 6.0]), (
            "endpoint with batching enabled returned wrong predictions — "
            "batching config likely broke TFS's model-load or request path"
        )
