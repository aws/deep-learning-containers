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
"""Documentation global variables for mkdocs-macros-plugin."""

from constants import GLOBAL_CONFIG
from image_config import get_latest_image


def define_env(env):
    """Define variables for mkdocs-macros-plugin."""
    # Expose all global config variables to mkdocs macros
    for key, value in GLOBAL_CONFIG.items():
        if isinstance(value, str):
            env.variables[key] = value

    # Image helpers
    env.variables["images"] = {
        "latest_pytorch_training_ec2": get_latest_image("pytorch-training", "ec2"),
        "latest_vllm_sagemaker": get_latest_image("vllm", "sagemaker"),
        "latest_sglang_sagemaker": get_latest_image("sglang", "sagemaker"),
    }
