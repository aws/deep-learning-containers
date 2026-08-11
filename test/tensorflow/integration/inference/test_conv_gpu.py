"""GPU-gated Conv2D integration test for TF 2.20 inference DLC.

Sanity tests check libcudnn.so presence but no request exercises a cuDNN
kernel. This deploys a tiny Conv2D SavedModel to ml.g6.4xlarge so a real
Conv routes through cuDNN's cudnnConvolutionForward — HTTP 200 proves
the libcudnn on the image is ABI-compatible with the TFS binary.
"""

from __future__ import annotations

import json
import tempfile

import pytest

from .resources.build_sample_model import build_conv_sample_model
from .resources.helpers import read_predictions, upload_tarball
from test_utils import random_suffix_name

pytestmark = pytest.mark.gpu


def test_conv2d_gpu_predict(
    sagemaker_session,
    deploy_endpoint,
):
    """Deploy a 117-param Conv2D SavedModel; prove cuDNN executes at request time."""
    with tempfile.TemporaryDirectory(prefix="tf220-conv-") as workdir:
        tar_path = build_conv_sample_model(output_dir=workdir)
        model_data = upload_tarball(
            sagemaker_session,
            tar_path,
            key_prefix=f"tf220-inference-tests/conv-gpu/{random_suffix_name('run', 63)}",
        )
        endpoint, endpoint_name, model_name = deploy_endpoint(
            model_data_url=model_data,
            name_prefix="tf220-conv-gpu",
        )

        # (1, 8, 8, 3) all-ones payload. Model pins Conv2D/Dense to
        # constant weights => single correct answer (108.0). See
        # _build_conv_sequential for the derivation.
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
        # NaN/Inf guard so the closed-form assertion doesn't hide a broken
        # cuDNN path behind a confusing "expected 108.0, got nan" message.
        assert scalar == scalar, f"NaN output from Conv2D forward pass: {scalar!r}"
        assert scalar not in (float("inf"), float("-inf")), f"Inf output: {scalar!r}"
        # Closed-form: Conv2D kernel=1/bias=0 x all-ones (1,8,8,3) -> 108.0.
        # A dead/stubbed cuDNN yields 0.0 here.
        assert scalar == pytest.approx(108.0, abs=0.5), (
            f"expected 108.0 from closed-form conv (Conv2D kernel=1.0 x all-ones "
            f"8x8x3 input), got {scalar!r}. A stubbed or bypassed cuDNN kernel "
            f"produces 0.0 here."
        )
