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

import logging
import os
import subprocess
from pathlib import Path

import yaml
from constants import GLOBAL_CONFIG, TABLES_DIR
from packaging.version import Version

LOGGER = logging.getLogger(__name__)


def load_yaml(path: str | Path) -> dict:
    """Load YAML file and return parsed content."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_table_config(repository: str) -> dict:
    """Load table column configuration for a repository.

    Raises:
        FileNotFoundError: If table config file does not exist.
    """
    path = TABLES_DIR / f"{repository}.yml"
    if not path.exists():
        raise FileNotFoundError(f"Table config not found: {path}")
    return load_yaml(path)


def load_jinja2(path: str | Path) -> str:
    """Load and return Jinja2 template file content."""
    with open(path) as f:
        return f.read()


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render headers and rows as a markdown table string."""
    if not rows:
        return ""
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, separator] + row_lines)


def write_output(path: str | Path, content: str) -> None:
    """Write content to output file."""
    with open(path, "w") as f:
        f.write(content)


def parse_version(version_str: str | None) -> Version:
    """Parse version string safely, returning Version("0") on failure."""
    try:
        return Version(version_str or "0")
    except Exception:
        return Version("0")


def clone_git_repository(git_repository: str, target_dir: str | Path) -> None:
    """Clone a git repository to target directory if it doesn't exist."""
    if os.path.exists(target_dir):
        return
    subprocess.run(["git", "clone", "--depth", "1", git_repository, target_dir], check=True)


def build_public_registry_note(repository: str) -> str:
    """Build markdown note linking to ECR Public Gallery for a repository."""
    url = f"{GLOBAL_CONFIG['public_gallery_url']}/{repository}"
    return f"These images are also available in ECR Public Gallery: [{repository}]({url})\n"


def check_public_registry(images: list, repository: str) -> bool:
    """Check if repository images are available in public registry."""
    if all(img.get("public_registry") for img in images):
        return True
    if any(img.get("public_registry") for img in images):
        LOGGER.warning(
            f"{repository} contains images with mixed public_registry values. "
            "Please check if this is a mistake."
        )
        return True
    return False
