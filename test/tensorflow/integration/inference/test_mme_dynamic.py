"""MME dynamic load / miss-path tests for TF 2.20 inference DLC.

Extends test_multi_model_endpoint.py with two scenarios:
  1. Target-model miss — must return 4xx, not hang.
  2. Late-add dynamic load — upload a model to the MME prefix after
     the endpoint is InService, then invoke it.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from .resources.build_sample_model import build_sample_model
from .resources.helpers import read_predictions
from test_utils import random_suffix_name


def test_mme_target_model_not_found(
    boto_session,
    sagemaker_session,
    deploy_endpoint,
    cleanup_endpoint,
):
    """Invoke with a target_model that isn't in the MME S3 prefix — must 4xx."""
    with tempfile.TemporaryDirectory(prefix="tf220-mme-miss-") as workdir:
        workdir_path = Path(workdir)
        model1_tar = build_sample_model(
            output_dir=workdir_path / "m1",
            multiplier=2.0,
            tar_filename="model1.tar.gz",
        )
        bucket = sagemaker_session.default_bucket()
        s3_key_prefix = f"tf220-inference-tests/mme-miss/{random_suffix_name('run', 63)}"
        sagemaker_session.upload_data(path=model1_tar, bucket=bucket, key_prefix=s3_key_prefix)
        s3_model_prefix = f"s3://{bucket}/{s3_key_prefix}/"

        endpoint, endpoint_name, model_name = deploy_endpoint(
            model_data_url=s3_model_prefix,
            mode="MultiModel",
            name_prefix="tf220-mme-miss",
        )
        cleanup_endpoint(endpoint_name, model_name=model_name)

        payload = json.dumps({"instances": [[1.0, 2.0, 3.0]]})
        with pytest.raises(ClientError) as excinfo:
            endpoint.invoke(
                body=payload,
                content_type="application/json",
                accept="application/json",
                target_model="does_not_exist.tar.gz",
            )
        status = excinfo.value.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        assert 400 <= status < 500, (
            f"expected 4xx on unknown target_model, got status {status}: {excinfo.value.response!r}"
        )


def test_mme_late_dynamic_load(
    boto_session,
    sagemaker_session,
    deploy_endpoint,
    cleanup_endpoint,
):
    """Deploy MME with one model, upload a second after InService, invoke it."""
    with tempfile.TemporaryDirectory(prefix="tf220-mme-late-") as workdir:
        workdir_path = Path(workdir)
        model1_tar = build_sample_model(
            output_dir=workdir_path / "m1",
            multiplier=2.0,
            tar_filename="model1.tar.gz",
        )
        bucket = sagemaker_session.default_bucket()
        s3_key_prefix = f"tf220-inference-tests/mme-late/{random_suffix_name('run', 63)}"
        sagemaker_session.upload_data(path=model1_tar, bucket=bucket, key_prefix=s3_key_prefix)
        s3_model_prefix = f"s3://{bucket}/{s3_key_prefix}/"

        endpoint, endpoint_name, model_name = deploy_endpoint(
            model_data_url=s3_model_prefix,
            mode="MultiModel",
            name_prefix="tf220-mme-late",
        )
        cleanup_endpoint(endpoint_name, model_name=model_name)

        # Sanity check: existing model responds.
        payload = json.dumps({"instances": [[1.0, 2.0, 3.0]]})
        r1 = endpoint.invoke(
            body=payload,
            content_type="application/json",
            accept="application/json",
            target_model="model1.tar.gz",
        )
        assert read_predictions(r1) == pytest.approx([2.0, 4.0, 6.0])

        # Upload a second model to the same S3 prefix while endpoint is running.
        model2_tar = build_sample_model(
            output_dir=workdir_path / "m2",
            multiplier=3.0,
            tar_filename="model2.tar.gz",
        )
        sagemaker_session.upload_data(path=model2_tar, bucket=bucket, key_prefix=s3_key_prefix)

        r2 = endpoint.invoke(
            body=payload,
            content_type="application/json",
            accept="application/json",
            target_model="model2.tar.gz",
        )
        assert read_predictions(r2) == pytest.approx([3.0, 6.0, 9.0]), (
            "late-loaded model2 did not respond with 3x — MME dynamic load path broken"
        )
