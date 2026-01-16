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
"""Common utilities for documentation generation."""

import os
import subprocess

import yaml


def load_yaml(path: str) -> dict:
    """Load and return YAML data."""
    with open(path) as f:
        return yaml.safe_load(f)


def clone_git_repository(git_repository: str, target_dir: str) -> None:
    """Clone sample tutorials repository into docs/tutorials."""
    if os.path.exists(target_dir):
        return

    subprocess.run(["git", "clone", "--depth", "1", git_repository, target_dir], check=True)
