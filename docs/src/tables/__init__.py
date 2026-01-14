# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
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
    sglang_table,
    vllm_table,
    pytorch_table,
    tensorflow_table,
    huggingface_pytorch_table,
    huggingface_tensorflow_table,
    neuron_table,
    autogluon_table,
    djl_table,
    triton_table,
]

__all__ = [
    "support_policy_table",
    "IMAGE_TABLE_GENERATORS",
]
