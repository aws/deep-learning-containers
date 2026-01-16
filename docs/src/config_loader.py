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
"""Configuration loader for release configs."""

from pathlib import Path

import yaml

RELEASES_DIR = Path(__file__).parent / "data" / "releases"

REQUIRED_FIELDS = [
    "metadata.framework",
    "metadata.job_type",
    "metadata.version",
    "metadata.accelerator",
    "metadata.platform",
    "metadata.architecture",
    "metadata.ga_date",
    "metadata.eop_date",
    "environment.python",
    "environment.os",
    "image.tags",
]


def _get_nested(data: dict, path: str):
    """Get nested value from dict using dot notation."""
    keys = path.split(".")
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def validate_config(config: dict, filepath: str) -> None:
    """Validate config has all required fields."""
    missing = [f for f in REQUIRED_FIELDS if _get_nested(config, f) is None]
    if missing:
        raise ValueError(f"{filepath}: Missing required fields: {missing}")


def load_config(filepath: str) -> dict:
    """Load single YAML config, inject repository from parent directory."""
    with open(filepath) as f:
        config = yaml.safe_load(f)

    # Inject repository from parent directory name
    config["repository"] = Path(filepath).parent.name
    config["_filepath"] = filepath

    validate_config(config, filepath)
    return config


def load_all_configs() -> list[dict]:
    """Load all configs from releases directories."""
    configs = []
    for repo_dir in RELEASES_DIR.iterdir():
        if not repo_dir.is_dir():
            continue
        for config_file in repo_dir.glob("*.yml"):
            configs.append(load_config(str(config_file)))
    return configs
