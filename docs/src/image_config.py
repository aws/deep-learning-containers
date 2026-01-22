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

import logging
from datetime import date
from pathlib import Path
from typing import Any

from constants import DATA_DIR, GLOBAL_CONFIG, LEGACY_DIR
from utils import load_yaml, parse_version

LOGGER = logging.getLogger(__name__)


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

    @property
    def is_supported(self) -> bool:
        """Check if image is still supported based on EOP date."""
        eop = self._data.get("eop")
        if not eop:
            return True
        return date.fromisoformat(eop) >= date.today()

    @property
    def has_support_dates(self) -> bool:
        """Check if image has GA and EOP dates defined."""
        return "ga" in self._data and "eop" in self._data

    @property
    def display_repository(self) -> str:
        """Get human-readable display name for the repository."""
        display_names = GLOBAL_CONFIG.get("display_names", {})
        if self._repository not in display_names:
            raise KeyError(f"Display name not found for: {self._repository}")
        return display_names[self._repository]

    @property
    def display_framework_group(self) -> str:
        """Get human-readable display name for the framework group."""
        display_names = GLOBAL_CONFIG.get("display_names", {})
        if self._framework_group not in display_names:
            raise KeyError(f"Display name not found for: {self._framework_group}")
        return display_names[self._framework_group]

    @property
    def display_framework_version(self) -> str:
        """Framework and version combined for table display."""
        return f"{self.get('framework', '')} {self.get('version', '')}"

    @property
    def display_example_url(self) -> str:
        """Example ECR URL for table display."""
        account = self.get("example_ecr_account", GLOBAL_CONFIG["example_ecr_account"])
        return (
            f"`{account}.dkr.ecr.<region>.amazonaws.com/{self._repository}:{self.get('tag', '')}`"
        )

    @property
    def display_platform(self) -> str:
        """Platform display value with mapping applied."""
        platforms = GLOBAL_CONFIG.get("platforms", {})
        return platforms.get(self.get("platform", ""), self.get("platform", "-"))

    @property
    def display_accelerator(self) -> str:
        """Accelerator display value with mapping applied."""
        accelerators = GLOBAL_CONFIG.get("accelerators", {})
        return accelerators.get(self.get("accelerator", ""), self.get("accelerator", "-").upper())

    def get_display(self, field: str) -> str:
        """Get display value for a field, using display_<field> property if available."""
        display_attr = f"display_{field}"
        if hasattr(self, display_attr):
            return getattr(self, display_attr)
        value = self.get(field)
        return str(value) if value is not None else "-"


def build_image_row(img: ImageConfig, columns: list[dict]) -> list[str]:
    """Build a table row from an ImageConfig using column definitions.

    In tables/<table>.yml, the <field> name will map to img.<field> / img.get(<field>) attribute.
    If you need to do string manipulation on the field, create a new property with convention display_<field>.
    """
    return [img.get_display(col.get("data", col["field"])) for col in columns]


def load_repository_images(repository: str) -> list[ImageConfig]:
    """Load all image configurations for a repository."""
    repo_dir = DATA_DIR / repository
    if not repo_dir.exists():
        return []
    return [ImageConfig.from_yaml(f, repository) for f in sorted(repo_dir.glob("*.yml"))]


def load_legacy_images() -> dict[str, list[ImageConfig]]:
    """Load legacy support policy data as ImageConfig objects.

    Returns:
        Mapping of framework key to list of ImageConfig objects.
    """
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


def sort_by_version(
    images: list[ImageConfig],
    tiebreakers: list[callable] | None = None,
) -> list[ImageConfig]:
    """Sort ImageConfig objects by version descending with optional tiebreakers."""

    def sort_key(img: ImageConfig):
        ver = parse_version(img.get("version"))
        keys = [-ver.major, -ver.minor, -ver.micro]
        if tiebreakers:
            keys.extend(fn(img) for fn in tiebreakers)
        return tuple(keys)

    return sorted(images, key=sort_key)


def check_public_registry(images: list[ImageConfig], repository: str) -> bool:
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


def get_latest_image(repo: str, platform: str) -> str:
    """Get the latest image URI for a repository and platform.

    Raises:
        ValueError: If no image found for the repository and platform combination.
    """
    images = load_repository_images(repo)

    matching = [img for img in images if img.get("platform") == platform]
    if not matching:
        raise ValueError(f"Image not found for {repo} with platform {platform}")

    latest = sort_by_version(matching)[0]
    account = latest.get("example_ecr_account", GLOBAL_CONFIG["example_ecr_account"])
    return f"{account}.dkr.ecr.us-west-2.amazonaws.com/{repo}:{latest.tag}"
