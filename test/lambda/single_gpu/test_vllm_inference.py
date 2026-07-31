"""Validate vLLM CUDA and offline-engine inference on a single GPU — vllm image."""

import os

import torch


def test_cuda_available():
    assert torch.cuda.is_available()


def test_device_count():
    assert torch.cuda.device_count() >= 1


def test_vllm_native_imports():
    """The vLLM stack (engine + compiled ops) imports."""
    import vllm  # noqa: F401
    from vllm import LLM, SamplingParams  # noqa: F401


def test_vllm_engine_generate():
    """The offline vllm.LLM loads a small model and generates non-empty text.

    Mirrors what handler.py does at cold start. Uses a tiny model and a low
    memory fraction to fit a single small GPU.
    """
    from vllm import LLM, SamplingParams

    model_id = os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    llm = LLM(model=model_id, gpu_memory_utilization=0.7, max_model_len=2048)
    sampling_params = SamplingParams(max_tokens=16)
    outputs = llm.generate(["The capital of France is"], sampling_params)
    text = outputs[0].outputs[0].text
    assert isinstance(text, str) and text.strip()
