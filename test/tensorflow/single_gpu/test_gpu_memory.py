"""GPU memory-allocator smoke test.

TF's BFC allocator deliberately retains pool memory across `clear_session()` /
`gc.collect()` — `get_memory_info(device)["current"]` reflects allocator-held
bytes, not user-held bytes. Unlike PyTorch's `torch.cuda.empty_cache()`, TF
exposes no public API that forces the pool back to the OS at runtime, so a
"memory after free == memory before alloc" assertion is unreliable.

Instead, assert the property we actually care about: when a tensor is freed,
the next allocation of the same size REUSES the existing pool rather than
growing it. Peak memory across two alloc/free cycles should be flat.
"""

import gc

import pytest
import tensorflow as tf

pytestmark = pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="GPU only")

# 5% growth tolerance between cycles for minor allocator bookkeeping.
PEAK_GROWTH_TOLERANCE = 1.05


def _peak_for_cycle(device: str, shape) -> int:
    """Reset peak stats, alloc + free, return the cycle's peak in bytes."""
    tf.config.experimental.reset_memory_stats(device)
    t = tf.zeros(shape)
    peak = tf.config.experimental.get_memory_info(device)["peak"]
    del t
    gc.collect()
    return peak


def test_gpu_memory_reuses_pool_across_cycles():
    device = "GPU:0"
    shape = [1000, 1000, 100]  # ~400 MB float32

    peak1 = _peak_for_cycle(device, shape)
    assert peak1 > 0, "allocation did not register in peak memory"

    peak2 = _peak_for_cycle(device, shape)
    assert peak2 <= peak1 * PEAK_GROWTH_TOLERANCE, (
        f"BFC pool grew on identical re-allocation: peak1={peak1}, peak2={peak2} "
        f"(tolerance {PEAK_GROWTH_TOLERANCE}x). Suggests freed memory was not reused."
    )
