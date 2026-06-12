"""tf.data pipeline smoke tests — runs on CPU and GPU containers."""

import numpy as np
import tensorflow as tf


def test_dataset_multiworker_prefetch():
    """Build a parallel-mapped, prefetched pipeline and verify all elements are seen."""
    n = 1000
    data = np.arange(n).reshape(n, 1).astype(np.float32)
    dataset = (
        tf.data.Dataset.from_tensor_slices(data)
        .batch(32)
        .map(lambda x: x * 2, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    seen = 0
    for batch in dataset:
        seen += int(batch.shape[0])

    assert seen == n, f"expected {n} elements, saw {seen}"


def test_dataset_interleave():
    """Split data into per-shard datasets and interleave — verify all elements produced."""
    per_shard = 200
    num_shards = 4
    n = per_shard * num_shards
    data = np.arange(n).reshape(num_shards, per_shard, 1).astype(np.float32)

    # Outer dataset yields each shard as a tensor; interleave fans out to inner datasets.
    dataset = tf.data.Dataset.from_tensor_slices(data).interleave(
        lambda shard: tf.data.Dataset.from_tensor_slices(shard).batch(16),
        cycle_length=num_shards,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    seen = 0
    for batch in dataset:
        seen += int(batch.shape[0])

    assert seen == n, f"expected {n} elements, saw {seen}"
