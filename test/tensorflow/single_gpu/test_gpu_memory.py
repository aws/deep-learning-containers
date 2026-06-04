"""GPU memory-allocator smoke test — allocate, free, verify no large leak."""

import gc

import pytest
import tensorflow as tf

pytestmark = pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="GPU only")

# 50 MB tolerance for runtime overhead; TF caches GPU memory so an exact match isn't expected.
LEAK_TOLERANCE_BYTES = 50 * 1024 * 1024


def test_gpu_memory_returns_to_baseline():
    device = "GPU:0"

    # Warm up so steady-state allocator overhead is included in the baseline.
    warmup = tf.zeros([1024, 1024])
    del warmup
    gc.collect()

    baseline = tf.config.experimental.get_memory_info(device)["current"]

    # Allocate ~400 MB: 1000 * 1000 * 100 float32 elements.
    big = tf.zeros([1000, 1000, 100])
    peak = tf.config.experimental.get_memory_info(device)["current"]
    assert peak > baseline, f"allocation did not raise current memory: {baseline} -> {peak}"

    del big
    tf.keras.backend.clear_session()
    gc.collect()

    after = tf.config.experimental.get_memory_info(device)["current"]
    leaked = after - baseline
    assert leaked <= LEAK_TOLERANCE_BYTES, (
        f"memory not released: baseline={baseline}, after={after}, leaked={leaked} bytes"
    )
