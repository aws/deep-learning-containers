"""Table generators for documentation."""

from . import (
    autogluon_table,
    base_table,
    djl_table,
    huggingface_pytorch_table,
    huggingface_tensorflow_table,
    neuron_table,
    pytorch_table,
    sglang_table,
    support_policy_table,
    tensorflow_table,
    triton_table,
    vllm_table,
)

IMAGE_TABLE_GENERATORS = [
    base_table,
    pytorch_table,
    tensorflow_table,
    huggingface_pytorch_table,
    huggingface_tensorflow_table,
    neuron_table,
    autogluon_table,
    djl_table,
    vllm_table,
    sglang_table,
    triton_table,
]

__all__ = [
    "support_policy_table",
    "IMAGE_TABLE_GENERATORS",
]
