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
