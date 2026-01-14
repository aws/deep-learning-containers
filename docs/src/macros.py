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

import os

from utils import load_yaml

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SRC_DIR, "data", "images.yml")


def get_latest_image(repo: str, platform: str) -> str:
    """Get the latest image URI for a repository and platform."""
    data = load_yaml(DATA_FILE)
    repo_data = data.get("images", {}).get(repo, {})
    tags = repo_data.get("tags", [])
    for tag in tags:
        if tag.endswith(platform):
            return f"763104351884.dkr.ecr.us-west-2.amazonaws.com/{repo}:{tag}"
    raise ValueError(
        f"Image not found for {repo} with platform {platform}. Docs must be fixed to use a valid image."
    )


def define_env(env):
    """Define variables for mkdocs-macros-plugin."""
    env.variables["images"] = {
        "latest_pytorch_training_ec2": get_latest_image("pytorch-training", "-ec2"),
        "latest_vllm_sagemaker": get_latest_image("vllm", "-sagemaker"),
        "latest_sglang_sagemaker": get_latest_image("sglang", "-sagemaker"),
    }
