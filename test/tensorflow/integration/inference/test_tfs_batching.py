"""TFS batching-config integration test for TF 2.20 inference DLC.

Deploys with SAGEMAKER_TFS_ENABLE_BATCHING=true and verifies predictions
are still numerically correct — proves the wire contract. Measuring actual
batch behavior would need TFS stdout, which SM managed endpoints don't expose.
"""

from __future__ import annotations

import json
import tempfile

import pytest
from test_utils import random_suffix_name

from .resources.build_sample_model import build_sample_model
from .resources.helpers import read_predictions, upload_tarball


def test_tfs_batching_enabled_wire_contract(
    sagemaker_session,
    deploy_endpoint,
):
    """Batching env vars set -> endpoint still responds correctly."""
    with tempfile.TemporaryDirectory(prefix="tf220-batching-") as workdir:
        tar_path = build_sample_model(output_dir=workdir, multiplier=2.0)
        model_data = upload_tarball(
            sagemaker_session,
            tar_path,
            key_prefix=f"tf220-inference-tests/batching/{random_suffix_name('run', 63)}",
        )
        endpoint, endpoint_name, model_name = deploy_endpoint(
            model_data_url=model_data,
            name_prefix="tf220-batching",
            container_env={
                "SAGEMAKER_TFS_ENABLE_BATCHING": "true",
                # Small knobs so batching engages promptly.
                "SAGEMAKER_TFS_MAX_BATCH_SIZE": "8",
                "SAGEMAKER_TFS_BATCH_TIMEOUT_MICROS": "1000",
                "SAGEMAKER_TFS_NUM_BATCH_THREADS": "1",
                "SAGEMAKER_TFS_MAX_ENQUEUED_BATCHES": "10000",
            },
        )

        payload = json.dumps({"instances": [[1.0, 2.0, 3.0]]})
        result = endpoint.invoke(
            body=payload,
            content_type="application/json",
            accept="application/json",
        )
        # A malformed batching_parameters_file would 5xx before this line.
        assert read_predictions(result) == pytest.approx([2.0, 4.0, 6.0]), (
            "endpoint with batching enabled returned wrong predictions — "
            "batching config likely broke TFS's model-load or request path"
        )
