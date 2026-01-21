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
from collections import defaultdict
from pathlib import Path

import yaml
from constants import DATA_DIR, GLOBAL_CONFIG_PATH, TABLES_DIR
from omegaconf import OmegaConf
from packaging.version import Version

LOGGER = logging.getLogger(__name__)


def load_global_config() -> dict:
    """Load and resolve global.yml configuration using OmegaConf."""
    cfg = OmegaConf.load(GLOBAL_CONFIG_PATH)
    return OmegaConf.to_container(cfg, resolve=True)


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


def load_image_configs(repository: str) -> list[dict]:
    """Load all image configuration files for a repository.

    Each config dict includes '_repository' key with the repository name.
    Returns empty list if repository directory does not exist.
    """
    repo_dir = DATA_DIR / repository
    if not repo_dir.exists():
        return []

    configs = []
    for yml_file in sorted(repo_dir.glob("*.yml")):
        config = load_yaml(yml_file)
        config["_repository"] = repository
        configs.append(config)
    return configs


def load_all_image_configs() -> dict[str, list[dict]]:
    """Load all image configurations across all repositories.

    Returns dict mapping repository name to list of image configs.
    """
    result = {}
    for repo_dir in DATA_DIR.iterdir():
        if repo_dir.is_dir():
            configs = load_image_configs(repo_dir.name)
            if configs:
                result[repo_dir.name] = configs
    return result


def get_display_name(global_config: dict, repository: str) -> str:
    """Get human-readable display name for a repository.

    Raises:
        KeyError: If repository not found in global config display_names.
    """
    display_names = global_config.get("display_names", {})
    if repository not in display_names:
        raise KeyError(
            f"Display name not found for repository: {repository}. Add it to global.yml display_names."
        )
    return display_names[repository]


def build_ecr_url(image_config: dict, global_config: dict, repository: str) -> str:
    """Build ECR URL string for an image with region placeholder."""
    account = image_config.get("example_ecr_account", global_config["example_ecr_account"])
    tag = image_config["tag"]
    return f"`{account}.dkr.ecr.<region>.amazonaws.com/{repository}:{tag}`"


def build_public_registry_note(repository: str, global_config: dict) -> str:
    """Build markdown note linking to ECR Public Gallery for a repository."""
    url = f"{global_config['public_gallery_url']}/{repository}"
    return f"These images are also available in ECR Public Gallery: [{repository}]({url})\n"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render headers and rows as a markdown table string.

    Returns empty string if rows is empty.
    """
    if not rows:
        return ""
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, separator] + row_lines)


def read_template(path: str | Path) -> str:
    """Read and return template file content."""
    with open(path) as f:
        return f.read()


def write_output(path: str | Path, content: str) -> None:
    """Write content to output file."""
    with open(path, "w") as f:
        f.write(content)


def clone_git_repository(git_repository: str, target_dir: str | Path) -> None:
    """Clone a git repository to target directory if it doesn't exist."""
    if os.path.exists(target_dir):
        return
    subprocess.run(["git", "clone", "--depth", "1", git_repository, target_dir], check=True)


def get_field_value(image: dict, field: str, global_config: dict, repository: str) -> str:
    """Get formatted field value from image config.

    Handles computed fields: framework_version, example_url, platform, accelerator.
    Returns '-' for missing fields.
    """
    field_handlers = {
        "framework_version": lambda: f"{image.get('framework', '')} {image.get('version', '')}",
        "example_url": lambda: build_ecr_url(image, global_config, repository),
        "platform": lambda: global_config.get("platforms", {}).get(
            image.get("platform", ""), image.get("platform", "")
        ),
        "accelerator": lambda: image.get("accelerator", "").upper(),
    }
    return field_handlers.get(field, lambda: str(image.get(field, "-")))()


def group_images_by_version(all_images: dict[str, list[dict]]) -> dict[tuple[str, str], dict]:
    """Group images by (repository, version) and validate GA/EOP consistency.

    Returns dict mapping (repository, version) tuple to dict with ga, eop, and images.

    Raises:
        ValueError: If GA or EOP dates are inconsistent within a version group.
    """
    version_data = defaultdict(lambda: {"ga": None, "eop": None, "images": []})

    for repository, images in all_images.items():
        for image in images:
            if "ga" not in image or "eop" not in image:
                continue

            key = (repository, image["version"])
            data = version_data[key]

            if data["ga"] is not None and data["ga"] != image["ga"]:
                raise ValueError(
                    f"Inconsistent GA date for {repository} {image['version']}: "
                    f"{data['ga']} vs {image['ga']}"
                )
            if data["eop"] is not None and data["eop"] != image["eop"]:
                raise ValueError(
                    f"Inconsistent EOP date for {repository} {image['version']}: "
                    f"{data['eop']} vs {image['eop']}"
                )

            data["ga"] = image["ga"]
            data["eop"] = image["eop"]
            data["images"].append(image)

    return version_data


def check_public_registry(images: list[dict], repository: str) -> bool:
    """Check if repository images are available in public registry.

    Logs warning if some but not all images have public_registry set.
    """
    if all(img.get("public_registry") for img in images):
        return True
    if any(img.get("public_registry") for img in images):
        LOGGER.warning(
            f"{repository} contain images with public_registry: true, and public_registry: false."
            "Please check if this is a mistake or typo."
        )
        return True
    return False


def build_image_table(
    images: list[dict], columns: list[dict], global_config: dict, repository: str
) -> str:
    """Build markdown table for a list of images using column configuration."""
    headers = [col["header"] for col in columns]
    rows = [
        [get_field_value(image, col["field"], global_config, repository) for col in columns]
        for image in images
    ]
    return render_table(headers, rows)


def get_latest_image(repo: str, platform: str) -> str:
    """Get the latest image URI for a repository and platform.

    Finds the image with the highest version number matching the platform.

    Raises:
        ValueError: If no image found for the repository and platform combination.
    """
    global_config = load_global_config()
    images = load_image_configs(repo)

    matching = [img for img in images if img.get("platform") == platform]
    if not matching:
        raise ValueError(
            f"Image not found for {repo} with platform {platform}. Docs must be fixed to use a valid image."
        )

    def version_key(img: dict) -> Version:
        try:
            return Version(img.get("version", "0"))
        except Exception:
            return Version("0")

    latest = max(matching, key=version_key)
    account = latest.get("example_ecr_account", global_config["example_ecr_account"])
    return f"{account}.dkr.ecr.us-west-2.amazonaws.com/{repo}:{latest['tag']}"
