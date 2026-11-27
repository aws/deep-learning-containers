"""GPU-gated Conv2D + BatchNorm integration test for the TF inference DLC.

Sanity tests check libcudnn.so presence but no request exercises a cuDNN
kernel. This deploys a tiny Conv2D + BatchNormalization SavedModel to a GPU
endpoint so a real Conv routes through cuDNN's cudnnConvolutionForward (HTTP
200 proves the libcudnn on the image is ABI-compatible with the TFS binary) and
BatchNormalization's Rsqrt runs a kernel TensorFlow compiles at runtime, which
needs ptxas and libdevice (cuda-nvcc) in the image. The endpoint
log must show TensorFlow Serving created the GPU device, because TensorFlow
silently falls back to CPU when it cannot load the GPU libraries or driver,
and the closed-form output alone would still pass.
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import pytest
from test_utils import random_suffix_name

from .resources.build_sample_model import CONV_SAMPLE_EXPECTED, build_conv_sample_model
from .resources.helpers import read_predictions, upload_tarball

pytestmark = pytest.mark.skipif(
    os.environ.get("SM_DEVICE_TYPE", "").lower() != "gpu",
    reason="GPU-only test (requires SM_DEVICE_TYPE=gpu)",
)

# TensorFlow logs this when it registers the first GPU, e.g.
# "Created device /job:localhost/replica:0/task:0/device:GPU:0 with 21458 MB
# memory:  -> device: 0, name: NVIDIA L4, pci bus id: ..., compute capability: 8.9"
GPU_DEVICE_MARKER = "device:GPU:0"

# Blackwell runs (SM_GPU_INSTANCE_TYPE=ml.g7*) must land on a cc 12.0 GPU.
_EXPECTED_CAPABILITY = {"ml.g7.": "compute capability: 12.0", "ml.g7e.": "compute capability: 12.0"}


def _gpu_device_log_lines(aws_session, endpoint_name, timeout=300):
    """Return endpoint log lines that mention the GPU device, polling for delivery lag."""
    logs = aws_session.session.client("logs")
    group = f"/aws/sagemaker/Endpoints/{endpoint_name}"
    deadline = time.time() + timeout
    while True:
        lines = []
        try:
            paginator = logs.get_paginator("filter_log_events")
            for page in paginator.paginate(
                logGroupName=group, filterPattern=f'"{GPU_DEVICE_MARKER}"'
            ):
                lines.extend(e["message"] for e in page.get("events", []))
        except logs.exceptions.ResourceNotFoundException:
            pass
        if lines or time.time() >= deadline:
            return lines
        time.sleep(15)


def test_conv2d_gpu_predict(
    aws_session,
    sagemaker_session,
    deploy_endpoint,
):
    """Deploy a Conv2D + BatchNorm SavedModel; prove it runs on the GPU at request time."""
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

        # (1, 8, 8, 3) all-ones payload. Model pins Conv2D/Dense to constant
        # weights => single correct answer (CONV_SAMPLE_EXPECTED, ~107.946). See
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
        # GPU path behind a confusing "expected 107.946, got nan" message.
        assert scalar == scalar, f"NaN output from Conv2D forward pass: {scalar!r}"
        assert scalar not in (float("inf"), float("-inf")), f"Inf output: {scalar!r}"
        # A dead/stubbed cuDNN yields 0.0 here.
        assert scalar == pytest.approx(CONV_SAMPLE_EXPECTED, abs=1e-2), (
            f"expected {CONV_SAMPLE_EXPECTED:.3f} from the closed-form Conv2D + BatchNorm "
            f"model on an all-ones 8x8x3 input, got {scalar!r}. A stubbed or bypassed "
            f"cuDNN kernel produces 0.0 here."
        )

        # The right answer alone does not prove GPU execution: TensorFlow runs
        # the same graph on CPU when it cannot load the GPU libraries.
        lines = _gpu_device_log_lines(aws_session, endpoint_name)
        assert lines, (
            f"no '{GPU_DEVICE_MARKER}' line in /aws/sagemaker/Endpoints/{endpoint_name}: "
            f"TensorFlow Serving did not register the GPU, so the model ran on CPU"
        )
        print(f"GPU device: {lines[0].strip()}")
        pinned = os.environ.get("SM_GPU_INSTANCE_TYPE", "").strip()
        for prefix, capability in _EXPECTED_CAPABILITY.items():
            if pinned.startswith(prefix):
                assert any(capability in line for line in lines), (
                    f"expected a GPU with {capability} on {pinned}, got: {lines}"
                )
