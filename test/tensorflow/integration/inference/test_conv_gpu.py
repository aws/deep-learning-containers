"""GPU-gated Conv2D integration test for the TF 2.20 inference DLC.

Sanity tests assert ``libcudnn.so.*`` is present in ``ldconfig -p`` and
appears in ``ldd tensorflow_model_server``, but no request in the current
suite actually invokes a cuDNN kernel. A cuDNN ABI drift (library present
but incompatible with the TFS binary) would pass sanity and fault on the
first customer Conv/RNN/LSTM request. This test closes that gap by
deploying a tiny 117-param Conv2D SavedModel and issuing a real prediction
against ``ml.g6.4xlarge`` — TFS routes the Conv2D op through cuDNN's
``cudnnConvolutionForward`` on GPU, so a HTTP 200 with numeric predictions
proves the ``libcudnn.so`` on the image is ABI-compatible with the TFS
binary end-to-end.

Skip mechanism mirrors how the workflow forwards device info to conftest:
``.github/workflows/tensorflow.tests-sagemaker-inference.yml`` sets
``SM_DEVICE_TYPE`` from ``inputs.device-type`` (line 87). The
``sm_instance_type`` fixture reads that env var — cpu -> ``ml.c5.xlarge``,
gpu -> ``ml.g6.4xlarge``. This test skips whenever ``SM_DEVICE_TYPE`` is
not ``gpu``, so the CPU-image job in the same pipeline skips it cleanly
without running (and paying for) a GPU host that can't test cuDNN anyway.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from .resources.build_sample_model import build_conv_sample_model
from .resources.helpers import read_predictions, upload_tarball

# Explicit skipif over reading the fixture value inside the test body so the
# skip decision is visible at collection time. Matches the semantic the
# reusable workflow already forwards via SM_DEVICE_TYPE (see conftest.py).
_device = os.environ.get("SM_DEVICE_TYPE", "").lower()
assert _device in {"cpu", "gpu"}, f"SM_DEVICE_TYPE must be 'cpu' or 'gpu'; got {_device!r}"
pytestmark = pytest.mark.skipif(
    _device != "gpu",
    reason="cuDNN Conv2D smoke requires SM_DEVICE_TYPE=gpu (workflow sets this for GPU device-type only)",
)


def test_conv2d_gpu_predict(
    sagemaker_session,
    deploy_endpoint,
    unique_name,
    cleanup_endpoint,
):
    """Deploy a 117-param Conv2D SavedModel and prove cuDNN executes at
    request time via the Keras 3 ``model.export()`` path customers on
    TF >= 2.16 use.

    Payload shape mirrors the model's serving signature — ``(1, 8, 8, 3)``
    of zeros. Correctness of the numeric output is not asserted (this is
    a cuDNN kernel-execution smoke, not a numerical test); we assert only
    that TFS returned HTTP 200 with a well-formed ``predictions`` list and
    a finite numeric result. A CUDNN_STATUS_VERSION_MISMATCH or missing
    symbol at load time would surface as an endpoint failure and abort
    ``endpoint.wait_for_status('InService')``; a runtime-only failure
    (e.g. no compatible algo) would surface as a 5xx from
    ``endpoint.invoke``.
    """
    with tempfile.TemporaryDirectory(prefix="tf220-conv-") as workdir:
        tar_path = build_conv_sample_model(output_dir=workdir)
        model_data = upload_tarball(
            sagemaker_session,
            tar_path,
            key_prefix=f"tf220-inference-tests/conv-gpu/{unique_name('run')}",
        )
        endpoint, endpoint_name, model_name = deploy_endpoint(
            model_data_url=model_data,
            name_prefix="tf220-conv-gpu",
        )
        cleanup_endpoint(endpoint_name, model_name=model_name)

        # (batch=1, H=8, W=8, C=3) — all-ones payload. The model pins
        # every trainable parameter to a constant (Conv2D kernel=1.0/
        # bias=0.0, Dense kernel=1.0/bias=0.0) so the forward pass has
        # exactly one correct answer. See ``_build_conv_sequential`` in
        # ``resources/build_sample_model.py`` for the closed-form
        # derivation.
        instance = [[[1.0, 1.0, 1.0]] * 8] * 8
        payload = json.dumps({"instances": [instance]})

        result = endpoint.invoke(
            body=payload,
            content_type="application/json",
            accept="application/json",
        )
        values = read_predictions(result)

        # Dense(1) yields a single scalar per instance. Accept either
        # ``[[v]]`` or ``[v]`` depending on TFS's list-shape flattening.
        if isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
            values = values[0]
        assert isinstance(values, list) and len(values) == 1, (
            f"expected 1-element output, got {values!r}"
        )
        (scalar,) = values
        assert isinstance(scalar, (int, float)), (
            f"expected numeric output, got {type(scalar).__name__}: {scalar!r}"
        )
        # Finite check — catches NaN/Inf blow-ups from a broken cuDNN
        # path before the closed-form assertion runs (comparing NaN
        # against 108.0 would fail with a confusing "expected 108.0,
        # got nan" instead of pointing at the NaN itself).
        assert scalar == scalar, f"NaN output from Conv2D forward pass: {scalar!r}"
        assert scalar not in (float("inf"), float("-inf")), f"Inf output: {scalar!r}"
        # Deterministic closed-form check.
        # With Conv2D kernel=1.0, bias=0.0, Dense kernel=1.0, bias=0.0,
        # all-ones (1, 8, 8, 3) input:
        #   Conv2D: each of 4 filters sums 27 ones -> pre-activation 27.0
        #   ReLU: 27.0 (positive, unchanged)
        #   GAP: [27.0, 27.0, 27.0, 27.0]
        #   Dense(1): 27+27+27+27 = 108.0
        # A dead / stubbed cuDNN convolution yields Dense = 0.0 ->
        # assertion fires loudly. This replaces an earlier
        # ``scalar != 0.0`` guard that had a 100 % false-negative rate:
        # with a Dense bias floor of 1.0, a zero feature map still
        # produced 1.0 at the output, which passed the guard.
        assert scalar == pytest.approx(108.0, abs=0.5), (
            f"expected 108.0 from closed-form conv (Conv2D kernel=1.0 x all-ones "
            f"8x8x3 input), got {scalar!r}. A stubbed or bypassed cuDNN kernel "
            f"produces 0.0 here."
        )
