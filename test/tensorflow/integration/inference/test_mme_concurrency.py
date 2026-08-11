"""MME concurrency: concurrent invokes to different target_model values on
a single MME endpoint must all succeed with correct per-model predictions.

Exercises:
    - multi_model_utils.lock() under concurrent load
    - _mme_tfs_instances_status cross-worker consistency via pickle
    - Per-model TFS instance routing
    - SAGEMAKER_GUNICORN_WORKERS > 1 request handling

Guards against:
    - Non-atomic pickle write bricking MME state
    - Blocking fcntl.lockf stalling other greenlets
    - Ghost dict entries after failed loads returning permanent 409
"""

from __future__ import annotations

import concurrent.futures
import json
import tempfile
from pathlib import Path

import pytest

from .resources.build_sample_model import build_sample_model
from .resources.helpers import read_predictions
from test_utils import random_suffix_name

_NUM_MODELS = 3
_NUM_CONCURRENT_INVOKES = 8


def test_mme_concurrent_invoke_distinct_models(
    sagemaker_session,
    deploy_endpoint,
    cleanup_endpoint,
):
    """Deploy MME with N models, fire N*K concurrent invokes across all
    target_models, assert each returns its own model's multiplier."""

    with tempfile.TemporaryDirectory(prefix="tf220-mme-conc-") as workdir:
        workdir_path = Path(workdir)

        # N models with distinct multipliers so cross-model routing is verifiable.
        # multiplier index i => value (i+2), i.e. 2.0, 3.0, 4.0 for _NUM_MODELS=3.
        models: list[tuple[str, float, str]] = []
        for i in range(_NUM_MODELS):
            mult = float(i + 2)
            filename = f"model{i}.tar.gz"
            tar_path = build_sample_model(
                output_dir=workdir_path / f"m{i}",
                multiplier=mult,
                tar_filename=filename,
            )
            models.append((filename, mult, tar_path))

        # Upload all under a shared MME S3 prefix.
        bucket = sagemaker_session.default_bucket()
        run_id = random_suffix_name("mme-conc", 63)
        s3_key_prefix = f"tf220-inference-tests/mme-conc/{run_id}"
        for _filename, _mult, tar_path in models:
            sagemaker_session.upload_data(
                path=tar_path,
                bucket=bucket,
                key_prefix=s3_key_prefix,
            )
        s3_model_prefix = f"s3://{bucket}/{s3_key_prefix}/"

        endpoint, endpoint_name, model_name = deploy_endpoint(
            model_data_url=s3_model_prefix,
            mode="MultiModel",
            container_env={
                # Force cross-worker sync — pickle atomicity + lock coordination
                # only matter when >1 gunicorn worker races on MME state.
                "SAGEMAKER_GUNICORN_WORKERS": "2",
            },
            name_prefix="tf220-mme-conc",
        )
        cleanup_endpoint(endpoint_name, model_name=model_name)

        # Round-robin (filename, expected) across N*K invokes.
        payload_body = json.dumps({"instances": [[1.0, 2.0, 3.0]]})
        tasks: list[tuple[str, float, list[float]]] = []
        for k in range(_NUM_CONCURRENT_INVOKES):
            filename, mult, _tar = models[k % _NUM_MODELS]
            expected = [1.0 * mult, 2.0 * mult, 3.0 * mult]
            tasks.append((filename, mult, expected))

        def _invoke(task):
            filename, mult, expected = task
            result = endpoint.invoke(
                body=payload_body,
                content_type="application/json",
                accept="application/json",
                target_model=filename,
            )
            values = read_predictions(result)
            # read_predictions returns the first prediction; single-instance
            # payload => that's the row we want.
            return filename, mult, expected, values

        # Fire concurrent invokes.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=_NUM_CONCURRENT_INVOKES,
        ) as pool:
            results = list(pool.map(_invoke, tasks))

        # Every result must match its own model's multiplier — proves per-request
        # cross-worker sync survived concurrency.
        for filename, mult, expected, values in results:
            assert values == pytest.approx(expected), (
                f"MME concurrent invoke wrong: target_model={filename!r}, "
                f"expected {expected} (multiplier {mult}), got {values!r}"
            )
