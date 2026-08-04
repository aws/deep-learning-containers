"""GPU-gated LSTM integration test for TF 2.20 inference DLC.

Sanity tests check libcudnn_adv.so.9 presence (RNN library) but no request
exercises a cuDNN RNN kernel. This deploys a tiny LSTM SavedModel to
ml.g6.4xlarge so a real LSTM routes through cuDNN's RNN dispatch — HTTP
200 with a finite, non-zero, deterministic value proves the RNN path is
ABI-compatible with the TFS binary.

Complements test_conv_gpu.py (which exercises libcudnn_ops + libcudnn_cnn
via Conv2D). LSTM is the canonical customer-facing RNN model.
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import pytest

from .resources.build_sample_model import build_lstm_sample_model
from .resources.helpers import read_predictions, upload_tarball

# Explicit skipif so the skip decision is visible at collection time.
_device = os.environ.get("SM_DEVICE_TYPE", "").lower()
assert _device in {"cpu", "gpu"}, f"SM_DEVICE_TYPE must be 'cpu' or 'gpu'; got {_device!r}"
pytestmark = pytest.mark.skipif(
    _device != "gpu",
    reason="cuDNN LSTM smoke requires SM_DEVICE_TYPE=gpu (workflow sets this for GPU device-type only)",
)


def _extract_scalar(values) -> float:
    """LSTM(units=1) yields a length-1 output per instance; unwrap to a scalar.

    Tolerates TFS's occasional list-shape flattening ([[v]] vs [v]).
    """
    if isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
        values = values[0]
    assert isinstance(values, list) and len(values) == 1, (
        f"expected 1-element output, got {values!r}"
    )
    (scalar,) = values
    assert isinstance(scalar, (int, float)), (
        f"expected numeric output, got {type(scalar).__name__}: {scalar!r}"
    )
    return float(scalar)


def test_lstm_gpu_predict(
    sagemaker_session,
    deploy_endpoint,
    unique_name,
    cleanup_endpoint,
):
    """Deploy a tiny LSTM SavedModel; prove cuDNN RNN executes at request time.

    LSTM(units=1) with constant-init weights + all-ones (1, 3, 2) input is
    deterministic. Two invokes must return the same finite, non-zero value.
    Regression modes this catches:
        - NaN out of cuDNN dispatch (drift-broken kernel)
        - 0.0 out (stubbed / bypassed kernel, same class of bug as PR #6418)
        - Non-deterministic output (algorithm-selection bug or races)
    """
    with tempfile.TemporaryDirectory(prefix="tf220-lstm-") as workdir:
        tar_path = build_lstm_sample_model(output_dir=workdir)
        model_data = upload_tarball(
            sagemaker_session,
            tar_path,
            key_prefix=f"tf220-inference-tests/lstm-gpu/{unique_name('run')}",
        )
        endpoint, endpoint_name, model_name = deploy_endpoint(
            model_data_url=model_data,
            name_prefix="tf220-lstm-gpu",
        )
        cleanup_endpoint(endpoint_name, model_name=model_name)

        # (batch=1, timesteps=3, features=2) all-ones payload. Model pins
        # LSTM weights to constants -> single deterministic answer.
        instance = [[1.0, 1.0]] * 3
        payload = json.dumps({"instances": [instance]})

        # Two invokes for determinism check.
        scalars: list[float] = []
        for _ in range(2):
            result = endpoint.invoke(
                body=payload,
                content_type="application/json",
                accept="application/json",
            )
            values = read_predictions(result)
            scalars.append(_extract_scalar(values))

        # 1. Finite (NaN + Inf guard — a broken cuDNN dispatch shows up here).
        for i, scalar in enumerate(scalars):
            assert not math.isnan(scalar), f"invoke {i}: NaN output from LSTM: {scalar!r}"
            assert not math.isinf(scalar), f"invoke {i}: Inf output from LSTM: {scalar!r}"

        # 2. Non-zero. LSTM(units=1, kernel=1, recurrent=1, bias=0) with
        # all-ones input has all internal gates activated — output cannot
        # be exactly zero unless the kernel is stubbed / bypassed (same
        # failure class as the cuDNN-only-stub bug from PR #6418).
        assert scalars[0] != 0.0, (
            f"LSTM output was 0.0 — cuDNN RNN kernel likely bypassed or stubbed. "
            f"scalar={scalars[0]!r}"
        )

        # 3. Deterministic. Constant weights + fixed input must give the same
        # output on every invoke. Divergence points at cuDNN algorithm-selection
        # nondeterminism or a race in the RNN dispatch layer.
        assert scalars[0] == pytest.approx(scalars[1], rel=1e-6, abs=1e-6), (
            f"LSTM output non-deterministic across invokes: {scalars[0]!r} vs {scalars[1]!r}. "
            "cuDNN RNN dispatch should be deterministic for constant weights + fixed input."
        )
