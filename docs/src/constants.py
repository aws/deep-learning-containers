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
"""Constants for documentation generation."""

# Custom sections that can be added to release notes.
# To add a new section type, add its key here and it will be rendered if present in config.
ALLOWED_SECTIONS = [
    "known_issues",
    "deprecations",
]

# Display names for sections (key -> title)
SECTION_TITLES = {
    "known_issues": "Known Issues",
    "deprecations": "Deprecations",
}

# Framework display names
FRAMEWORK_NAMES = {
    "base": "Base",
    "sglang": "SGLang",
    "vllm": "vLLM",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
}

# Repository display names for available_images tables
REPOSITORY_NAMES = {
    "base": "Base",
    "sglang": "SGLang",
    "vllm": "vLLM",
    "vllm-arm64": "vLLM (ARM64)",
    "pytorch-training": "PyTorch Training",
    "pytorch-inference": "PyTorch Inference",
    "pytorch-training-arm64": "PyTorch Training (ARM64)",
    "pytorch-inference-arm64": "PyTorch Inference (ARM64)",
    "tensorflow-training": "TensorFlow Training",
    "tensorflow-inference": "TensorFlow Inference",
    "tensorflow-inference-arm64": "TensorFlow Inference (ARM64)",
}

# Repository ordering for available_images tables
REPOSITORY_ORDER = [
    "base",
    "sglang",
    "vllm",
    "vllm-arm64",
    "pytorch-training",
    "pytorch-inference",
    "pytorch-training-arm64",
    "pytorch-inference-arm64",
    "tensorflow-training",
    "tensorflow-inference",
    "tensorflow-inference-arm64",
]

# Framework ordering for support policy
FRAMEWORK_ORDER = ["base", "sglang", "vllm", "pytorch", "tensorflow"]

# Tutorials repository
TUTORIALS_REPO = "https://github.com/aws-samples/sample-aws-deep-learning-containers"
