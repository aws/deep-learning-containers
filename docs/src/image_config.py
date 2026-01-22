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

from constants import DATA_DIR, GLOBAL_CONFIG
from utils import load_yaml


class ImageConfig:
    """Dynamic configuration object for DLC images.

    Attributes are determined by YAML fields and accessible via obj.field or getattr().
    """

    def __init__(self, repository: str, **kwargs: Any) -> None:
        self._repository = repository
        self._data = kwargs
        # Compute framework_group from GLOBAL_CONFIG, default to repository
        self._framework_group = repository
        for group_key, repos in GLOBAL_CONFIG.get("framework_groups", {}).items():
            if repository in repos:
                self._framework_group = group_key
                break

    @classmethod
    def from_yaml(cls, path: Path, repository: str) -> "ImageConfig":
        """Create ImageConfig from a YAML file."""
        data = load_yaml(path)
        return cls(repository, **data)

    def __getattr__(self, name: str) -> Any:
        """Get field value by reference YAML data."""
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

    @property
    def framework_group(self) -> str:
        """Framework group key (or repository if not in a group)."""
        return self._framework_group

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

    def get_display_name(self) -> str:
        """Get human-readable display name for the repository."""
        display_names = GLOBAL_CONFIG.get("display_names", {})
        if self._repository not in display_names:
            raise KeyError(f"Display name not found for: {self._repository}")
        return display_names[self._repository]


def load_repository_images(repository: str) -> list[ImageConfig]:
    """Load all image configurations for a repository."""
    repo_dir = DATA_DIR / repository
    if not repo_dir.exists():
        return []
    return [ImageConfig.from_yaml(f, repository) for f in sorted(repo_dir.glob("*.yml"))]


def get_field_display(img: ImageConfig, field: str) -> str:
    """Get display value for a field with mappings applied."""
    if field == "framework_version":
        return f"{img.get('framework', '')} {img.get('version', '')}"

    if field == "example_url":
        account = img.get("example_ecr_account", GLOBAL_CONFIG["example_ecr_account"])
        tag = img.get("tag", "")
        return f"`{account}.dkr.ecr.<region>.amazonaws.com/{img.repository}:{tag}`"

    if field == "platform":
        platforms = GLOBAL_CONFIG.get("platforms", {})
        return platforms.get(img.get("platform", ""), img.get("platform", "-"))

    if field == "accelerator":
        accelerators = GLOBAL_CONFIG.get("accelerators", {})
        return accelerators.get(img.get("accelerator", ""), img.get("accelerator", "-").upper())

    # Default: return raw value or "-"
    value = img.get(field)
    return str(value) if value is not None else "-"


def build_image_row(img: ImageConfig, columns: list[dict]) -> list[str]:
    """Build a table row from an ImageConfig using column definitions."""
    return [get_field_display(img, col["field"]) for col in columns]


def sort_by_version(
    images: list[ImageConfig],
    tiebreakers: list[callable] | None = None,
) -> list[ImageConfig]:
    """Sort ImageConfig objects by version descending with optional tiebreakers."""
    from utils import parse_version

    def sort_key(img: ImageConfig):
        ver = parse_version(img.get("version"))
        keys = [-ver.major, -ver.minor, -ver.micro]
        if tiebreakers:
            keys.extend(fn(img) for fn in tiebreakers)
        return tuple(keys)

    return sorted(images, key=sort_key)


def get_framework_order() -> list[str]:
    """Derive framework order from table_order, collapsing framework groups."""
    table_order = GLOBAL_CONFIG.get("table_order", [])
    framework_groups = GLOBAL_CONFIG.get("framework_groups", {})

    # Build reverse mapping: repo -> framework group
    repo_to_group = {}
    for group, repos in framework_groups.items():
        for repo in repos:
            repo_to_group[repo] = group

    seen = set()
    result = []
    for repo in table_order:
        key = repo_to_group.get(repo, repo)
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def get_legacy_images() -> dict[str, list[ImageConfig]]:
    """Get legacy support policy data as ImageConfig objects.

    Returns:
        Mapping of framework key to list of ImageConfig objects.
    """
    from constants import LEGACY_DIR

    path = LEGACY_DIR / "legacy_support.yml"
    if not path.exists():
        return {}

    data = load_yaml(path)
    return {
        framework: [
            ImageConfig(framework, version=e["version"], ga=e["ga"], eop=e["eop"]) for e in entries
        ]
        for framework, entries in data.items()
    }


def get_latest_image(repo: str, platform: str) -> str:
    """Get the latest image URI for a repository and platform.

    Raises:
        ValueError: If no image found for the repository and platform combination.
    """
    from utils import parse_version

    images = load_repository_images(repo)

    matching = [img for img in images if img.get("platform") == platform]
    if not matching:
        raise ValueError(f"Image not found for {repo} with platform {platform}")

    latest = max(matching, key=lambda img: parse_version(img.get("version")))
    account = latest.get("example_ecr_account", GLOBAL_CONFIG["example_ecr_account"])
    return f"{account}.dkr.ecr.us-west-2.amazonaws.com/{repo}:{latest.tag}"
