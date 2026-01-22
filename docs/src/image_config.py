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
"""ImageConfig class and image loading utilities."""

from datetime import date
from pathlib import Path
from typing import Any

from constants import DATA_DIR
from file_loader import load_yaml


class ImageConfig:
    """Dynamic configuration object for DLC images.

    Attributes are determined by YAML fields and accessible via obj.field or getattr().
    """

    def __init__(self, repository: str, **kwargs: Any) -> None:
        self._repository = repository
        self._data = kwargs

    @classmethod
    def from_yaml(cls, path: Path, repository: str) -> "ImageConfig":
        """Create ImageConfig from a YAML file."""
        data = load_yaml(path)
        return cls(repository, **data)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"ImageConfig has no attribute '{name}'")

    def get(self, field: str, default: Any = None) -> Any:
        """Get field value with optional default."""
        return self._data.get(field, default)

    @property
    def repository(self) -> str:
        """Repository name for this image."""
        return self._repository

    def is_supported(self, today: date | None = None) -> bool:
        """Check if image is still supported based on EOP date."""
        eop = self._data.get("eop")
        if not eop:
            return True
        today = today or date.today()
        return date.fromisoformat(eop) >= today

    def has_support_dates(self) -> bool:
        """Check if image has GA and EOP dates defined."""
        return "ga" in self._data and "eop" in self._data

    def get_display_name(self, global_config: dict) -> str:
        """Get human-readable display name for the repository.

        Raises:
            KeyError: If repository not found in display_names.
        """
        display_names = global_config.get("display_names", {})
        if self._repository not in display_names:
            raise KeyError(f"Display name not found for: {self._repository}")
        return display_names[self._repository]

    def get_framework_group(self, global_config: dict) -> str | None:
        """Get framework group key if repository belongs to one."""
        framework_groups = global_config.get("framework_groups", {})
        for group_key, repos in framework_groups.items():
            if self._repository in repos:
                return group_key
        return None


def load_image(path: Path, repository: str) -> ImageConfig:
    """Load a single image configuration from YAML file."""
    return ImageConfig.from_yaml(path, repository)


def load_repository_images(repository: str) -> list[ImageConfig]:
    """Load all image configurations for a repository."""
    repo_dir = DATA_DIR / repository
    if not repo_dir.exists():
        return []
    return [load_image(f, repository) for f in sorted(repo_dir.glob("*.yml"))]


def load_all_images() -> dict[str, list[ImageConfig]]:
    """Load all image configurations across all repositories."""
    result = {}
    for repo_dir in DATA_DIR.iterdir():
        if repo_dir.is_dir():
            images = load_repository_images(repo_dir.name)
            if images:
                result[repo_dir.name] = images
    return result


def get_field_display(img: ImageConfig, field: str, global_config: dict) -> str:
    """Get display value for a field with mappings applied.

    Handles special fields: framework_version, example_url, platform, accelerator.
    """
    if field == "framework_version":
        return f"{img.get('framework', '')} {img.get('version', '')}"

    if field == "example_url":
        account = img.get("example_ecr_account", global_config["example_ecr_account"])
        tag = img.get("tag", "")
        return f"`{account}.dkr.ecr.<region>.amazonaws.com/{img.repository}:{tag}`"

    if field == "platform":
        platforms = global_config.get("platforms", {})
        return platforms.get(img.get("platform", ""), img.get("platform", "-"))

    if field == "accelerator":
        accelerators = global_config.get("accelerators", {})
        return accelerators.get(img.get("accelerator", ""), img.get("accelerator", "-").upper())

    # Default: return raw value or "-"
    value = img.get(field)
    return str(value) if value is not None else "-"


def build_image_row(img: ImageConfig, columns: list[dict], global_config: dict) -> list[str]:
    """Build a table row from an ImageConfig using column definitions."""
    return [get_field_display(img, col["field"], global_config) for col in columns]


def sort_images_for_table(images: list[ImageConfig]) -> list[ImageConfig]:
    """Sort images for available_images table: version desc, sagemaker first, gpu first."""
    from utils import parse_version

    def sort_key(img: ImageConfig):
        ver = parse_version(img.get("version"))
        platform_order = 0 if img.get("platform") == "sagemaker" else 1
        accel = img.get("accelerator", "").lower()
        accel_order = 0 if accel == "gpu" else 1 if accel == "neuronx" else 2
        return (-ver.major, -ver.minor, -ver.micro, platform_order, accel_order)

    return sorted(images, key=sort_key)


def sort_support_entries(entries: list[dict], table_order: list[str]) -> list[dict]:
    """Sort support policy entries by table_order then version descending."""
    from utils import parse_version

    def sort_key(item):
        repo = item["_sort_repo"]
        order = table_order.index(repo) if repo in table_order else len(table_order)
        ver = parse_version(item["version"])
        return (order, -ver.major, -ver.minor, -ver.micro)

    return sorted(entries, key=sort_key)


def get_latest_image(repo: str, platform: str) -> str:
    """Get the latest image URI for a repository and platform.

    Raises:
        ValueError: If no image found for the repository and platform combination.
    """
    from file_loader import load_global_config
    from utils import parse_version

    global_config = load_global_config()
    images = load_repository_images(repo)

    matching = [img for img in images if img.get("platform") == platform]
    if not matching:
        raise ValueError(f"Image not found for {repo} with platform {platform}")

    latest = max(matching, key=lambda img: parse_version(img.get("version")))
    account = latest.get("example_ecr_account", global_config["example_ecr_account"])
    return f"{account}.dkr.ecr.us-west-2.amazonaws.com/{repo}:{latest.tag}"
