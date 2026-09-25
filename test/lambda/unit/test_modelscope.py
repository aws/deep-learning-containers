"""modelscope ships in the vllm image for ModelScope hub model loading."""

import importlib
import importlib.metadata as md


def test_modelscope_importable():
    importlib.import_module("modelscope")


def test_modelscope_below_incompatible_release():
    major, minor = (int(p) for p in md.version("modelscope").split(".")[:2])
    assert (major, minor) < (1, 38)
