"""runai-model-streamer (with the S3 backend) ships in the vLLM & sglang images.

This is the package behind `--load-format runai_streamer`, which lets vLLM / sglang
stream safetensors directly from S3 into GPU memory with no /tmp staging (set
MODEL_ID=s3://… + VLLM_LOAD_FORMAT/SGLANG_LOAD_FORMAT=runai_streamer on the handler).

The functional end-to-end S3-streaming serving check runs on real Lambda managed GPU
via test/lambda/platform/run_runai_s3.py. This unit test guards the cheaper contract:
the package and its S3 extra are installed and importable in the image.
"""

import importlib
import importlib.metadata as md

import pytest


def test_runai_model_streamer_importable():
    importlib.import_module("runai_model_streamer")


@pytest.mark.parametrize("dist", ["runai-model-streamer", "runai-model-streamer-s3"])
def test_runai_streamer_dist_installed(dist):
    # PackageNotFoundError if the dist (or its [s3] extra) is missing from the image.
    assert md.version(dist)
