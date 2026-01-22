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
"""File loading utilities for documentation generation."""

from pathlib import Path

import yaml
from constants import GLOBAL_CONFIG_PATH, LEGACY_DIR, TABLES_DIR
from omegaconf import OmegaConf


def load_yaml(path: str | Path) -> dict:
    """Load YAML file and return parsed content."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_global_config() -> dict:
    """Load and resolve global.yml configuration using OmegaConf."""
    cfg = OmegaConf.load(GLOBAL_CONFIG_PATH)
    return OmegaConf.to_container(cfg, resolve=True)


def load_table_config(repository: str) -> dict:
    """Load table column configuration for a repository.

    Raises:
        FileNotFoundError: If table config file does not exist.
    """
    path = TABLES_DIR / f"{repository}.yml"
    if not path.exists():
        raise FileNotFoundError(f"Table config not found: {path}")
    return load_yaml(path)


def load_legacy_support() -> dict[tuple[str, str], dict]:
    """Load legacy support policy data from legacy_support.yml.

    Returns:
        Mapping of (framework, version) tuple to {ga, eop} dict.
    """
    path = LEGACY_DIR / "legacy_support.yml"
    if not path.exists():
        return {}

    data = load_yaml(path)
    result = {}
    for framework, entries in data.items():
        for entry in entries:
            key = (framework, entry["version"])
            result[key] = {"ga": entry["ga"], "eop": entry["eop"]}
    return result


def load_jinja2(path: str | Path) -> str:
    """Load and return Jinja2 template file content."""
    with open(path) as f:
        return f.read()
