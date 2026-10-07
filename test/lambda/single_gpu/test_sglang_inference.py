"""Validate SGLang CUDA and offline-engine inference on a single GPU — sglang image."""

import os

import torch


def test_cuda_available():
    assert torch.cuda.is_available()


def test_device_count():
    assert torch.cuda.device_count() >= 1


def test_sglang_native_imports():
    """The sglang stack (engine + native kernels + attention backend) imports."""
    import flashinfer  # noqa: F401
    import sgl_kernel  # noqa: F401
    import sglang  # noqa: F401


def test_sglang_engine_generate():
    """The offline sgl.Engine loads a small model and generates non-empty text.

    Mirrors what handler.py does at cold start. Uses a tiny model and a low
    memory fraction to fit a single small GPU.
    """
    import sglang as sgl

    model_id = os.environ.get("SGLANG_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    engine = sgl.Engine(model_path=model_id, mem_fraction_static=0.7)
    try:
        out = engine.generate("The capital of France is", {"max_new_tokens": 16})
        text = out["text"] if isinstance(out, dict) else out[0]["text"]
        assert isinstance(text, str) and text.strip()
    finally:
        engine.shutdown()
