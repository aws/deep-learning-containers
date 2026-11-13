"""FlashInfer kernels ship precompiled in the vllm image, so cold start doesn't JIT-build them.

Without them vLLM builds FlashInfer's sampling kernels on every new execution
environment (~80 s), which pushes Init toward Lambda's limit. CPU-only: reads the
installed kernel providers' manifests, loads nothing onto a GPU.
"""

import importlib.metadata

import flashinfer_jit_cache as jit_cache
import pytest

# The GPU generations the image is built for: torch_cuda_arch_list in
# .github/config/image/lambda/vllm/vllm.yml (8.0 8.6 8.9 12.0). sm86 has no package of
# its own; whether it uses the sm80 kernels is not verified yet.
EXPECTED_ARCHITECTURES = {"sm80", "sm89", "sm120f"}


@pytest.fixture(scope="module")
def providers():
    return jit_cache.get_jit_cache_providers()


def test_kernels_cover_every_supported_gpu_architecture(providers):
    installed = set().union(*(p.cuda_architectures for p in providers))
    assert EXPECTED_ARCHITECTURES <= installed, f"missing {EXPECTED_ARCHITECTURES - installed}"


def test_kernels_include_the_sampling_module(providers):
    # The module vLLM JIT-built at every cold start before the cache was installed.
    for provider in providers:
        assert any("sampling" in m for m in provider.modules), provider.provider_id


def test_kernel_cache_matches_the_flashinfer_version(providers):
    flashinfer = importlib.metadata.version("flashinfer-python").split("+")[0]
    for provider in providers:
        assert provider.version.split("+")[0] == flashinfer, provider.provider_id
        assert provider.jit_cache_dir.is_dir(), provider.jit_cache_dir
