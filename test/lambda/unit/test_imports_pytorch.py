"""Verify key packages import successfully — pytorch image."""

import importlib

import pytest

REQUIRED_PACKAGES = [
    "accelerate",
    "av",
    "awslambdaric",
    "boto3",
    "cv2",
    "diffusers",
    "librosa",
    "numpy",
    "PIL",
    "safetensors",
    "scipy",
    "soundfile",
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
]


@pytest.mark.parametrize("package", REQUIRED_PACKAGES)
def test_import(package):
    importlib.import_module(package)


def test_sam2_importable():
    from sam2.build_sam import build_sam2_video_predictor  # noqa: F401
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: F401
