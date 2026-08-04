"""Model-versioning test for TF 2.20 inference DLC.

Verifies TFS picks the highest numeric version when a tarball contains
multiple <version>/saved_model.pb layouts. Both versions use the same
multiplier so the check doesn't depend on TFS picking a specific one.
"""

from __future__ import annotations

import json
import tempfile

import pytest

from .resources.build_sample_model import build_sample_model
from .resources.helpers import read_predictions, upload_tarball


def test_two_version_saved_model(
    sagemaker_session,
    deploy_endpoint,
    unique_name,
    cleanup_endpoint,
):
    with tempfile.TemporaryDirectory(prefix="tf220-versions-") as workdir:
        # versions=(1, 2) writes both under 1/ and 2/; TFS picks version 2.
        tar_path = build_sample_model(
            output_dir=workdir,
            multiplier=2.0,
            versions=(1, 2),
        )
        model_data = upload_tarball(
            sagemaker_session,
            tar_path,
            key_prefix=f"tf220-inference-tests/versions/{unique_name('run')}",
        )
        endpoint, endpoint_name, model_name = deploy_endpoint(
            model_data_url=model_data,
            name_prefix="tf220-versions",
        )
        cleanup_endpoint(endpoint_name, model_name=model_name)

        payload = json.dumps({"instances": [[1.0, 2.0, 3.0]]})
        result = endpoint.invoke(
            body=payload,
            content_type="application/json",
            accept="application/json",
        )
        assert read_predictions(result) == pytest.approx([2.0, 4.0, 6.0]), (
            "multi-version SavedModel tarball produced wrong predictions — "
            "either build_sample_model tarball layout regressed or TFS "
            "failed to load the highest-numbered version"
        )
