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
from datetime import date
from pathlib import Path

import yaml
from constants import DATA_DIR, GLOBAL_CONFIG_PATH, LEGACY_DIR, TABLES_DIR
from omegaconf import OmegaConf
from packaging.version import Version

LOGGER = logging.getLogger(__name__)


def load_global_config() -> dict:
    """Load and resolve global.yml configuration using OmegaConf.

    Returns:
        dict: Resolved configuration with all variable interpolations expanded.
    """
    cfg = OmegaConf.load(GLOBAL_CONFIG_PATH)
    return OmegaConf.to_container(cfg, resolve=True)


def load_yaml(path: str | Path) -> dict:
    """Load YAML file and return parsed content.

    Args:
        path: Path to the YAML file.

    Returns:
        dict: Parsed YAML content.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def load_table_config(repository: str) -> dict:
    """Load table column configuration for a repository.

    Args:
        repository: Name of the repository.

    Returns:
        dict: Table configuration with column definitions.

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
        dict[tuple[str, str], dict]: Mapping of (framework, version) tuple to {ga, eop} dict.
            Returns empty dict if file doesn't exist.
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


def is_image_supported(image: dict) -> bool:
    """Check if an image is still supported based on its EOP date.

    Args:
        image: Image configuration dict containing optional 'eop' field.

    Returns:
        bool: True if image has no 'eop' field or eop >= today.
    """
    eop = image.get("eop")
    if not eop:
        return True
    return date.fromisoformat(eop) >= date.today()


def load_image_configs(repository: str) -> list[dict]:
    """Load all image configuration files for a repository.

    Args:
        repository: Name of the repository directory.

    Returns:
        list[dict]: List of image configs, each with '_repository' key added.
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

    Returns:
        dict[str, list[dict]]: Mapping of repository name to list of image configs.
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

    Args:
        global_config: Global configuration dict containing display_names.
        repository: Name of the repository.

    Returns:
        str: Human-readable display name.

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
    """Build ECR URL string for an image with region placeholder.

    Args:
        image_config: Image configuration dict containing 'tag' and optional 'example_ecr_account'.
        global_config: Global configuration dict containing 'example_ecr_account'.
        repository: Name of the repository.

    Returns:
        str: Formatted ECR URL with region placeholder in markdown code format.
    """
    account = image_config.get("example_ecr_account", global_config["example_ecr_account"])
    tag = image_config["tag"]
    return f"`{account}.dkr.ecr.<region>.amazonaws.com/{repository}:{tag}`"


def build_public_registry_note(repository: str, global_config: dict) -> str:
    """Build markdown note linking to ECR Public Gallery for a repository.

    Args:
        repository: Name of the repository.
        global_config: Global configuration dict containing 'public_gallery_url'.

    Returns:
        str: Markdown formatted note with link to ECR Public Gallery.
    """
    url = f"{global_config['public_gallery_url']}/{repository}"
    return f"These images are also available in ECR Public Gallery: [{repository}]({url})\n"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render headers and rows as a markdown table string.

    Args:
        headers: List of column header strings.
        rows: List of row data, where each row is a list of cell values.

    Returns:
        str: Markdown formatted table. Returns empty string if rows is empty.
    """
    if not rows:
        return ""
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, separator] + row_lines)


def read_template(path: str | Path) -> str:
    """Read and return template file content.

    Args:
        path: Path to the template file.

    Returns:
        str: Template file content.
    """
    with open(path) as f:
        return f.read()


def write_output(path: str | Path, content: str) -> None:
    """Write content to output file.

    Args:
        path: Path to the output file.
        content: Content to write.
    """
    with open(path, "w") as f:
        f.write(content)


def clone_git_repository(git_repository: str, target_dir: str | Path) -> None:
    """Clone a git repository to target directory if it doesn't exist.

    Args:
        git_repository: Git repository URL to clone.
        target_dir: Target directory path for the clone.

    Raises:
        subprocess.CalledProcessError: If git clone command fails.
    """
    if os.path.exists(target_dir):
        return
    subprocess.run(["git", "clone", "--depth", "1", git_repository, target_dir], check=True)


def get_field_value(image: dict, field: str, global_config: dict, repository: str) -> str:
    """Get formatted field value from image config.

    Handles computed fields: framework_version, example_url, platform, accelerator.

    Args:
        image: Image configuration dict.
        field: Field name to retrieve.
        global_config: Global configuration dict.
        repository: Name of the repository.

    Returns:
        str: Formatted field value. Returns '-' for missing fields.
    """
    special_fields = {
        "framework_version": lambda: f"{image.get('framework', '')} {image.get('version', '')}",
        "example_url": lambda: build_ecr_url(image, global_config, repository),
        "platform": lambda: global_config.get("platforms", {}).get(
            image.get("platform", ""), image.get("platform", "-")
        ),
        "accelerator": lambda: global_config.get("accelerators", {}).get(
            image.get("accelerator", ""), image.get("accelerator", "-").upper()
        ),
    }
    # Default handlers to lambda: str(image.get(field), "-")
    return special_fields.get(field, lambda: str(image.get(field, "-")))()


def validate_date_consistency(
    current_ga: str | None,
    current_eop: str | None,
    new_ga: str,
    new_eop: str,
    msg_context: str,
) -> None:
    """Validate that GA/EOP dates are consistent with existing values.

    Args:
        current_ga: Existing GA date or None.
        current_eop: Existing EOP date or None.
        new_ga: New GA date to validate.
        new_eop: New EOP date to validate.
        context: Description for error message (e.g., "pytorch-training 2.9").

    Raises:
        ValueError: If dates are inconsistent.
    """
    if (current_ga is not None and current_ga != new_ga) or (
        current_eop is not None and current_eop != new_eop
    ):
        raise ValueError(
            f"Inconsistent dates for {msg_context}: GA=({current_ga} vs {new_ga}), EOP=({current_eop} vs {new_eop})"
        )


def group_images_by_version(all_images: dict[str, list[dict]]) -> dict[tuple[str, str], dict]:
    """Group images by (repository, version) and validate GA/EOP consistency.

    Args:
        all_images: Mapping of repository name to list of image configs.

    Returns:
        dict[tuple[str, str], dict]: Mapping of (repository, version) tuple to dict
            with 'ga', 'eop', and 'images' keys.

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

            validate_date_consistency(
                data["ga"],
                data["eop"],
                image["ga"],
                image["eop"],
                f"{repository} {image['version']}",
            )

            data["ga"] = image["ga"]
            data["eop"] = image["eop"]
            data["images"].append(image)

    return version_data


def check_public_registry(images: list[dict], repository: str) -> bool:
    """Check if repository images are available in public registry.

    Args:
        images: List of image configuration dicts.
        repository: Name of the repository.

    Returns:
        bool: True if any images have public_registry set to True.
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


def build_table(
    images: list[dict], columns: list[dict], global_config: dict, repository: str
) -> str:
    """Build markdown table for a list of images using column configuration.

    Args:
        images: List of image configuration dicts.
        columns: List of column definitions with 'field' and 'header' keys.
        global_config: Global configuration dict.
        repository: Name of the repository.

    Returns:
        str: Markdown formatted table.
    """
    headers = [col["header"] for col in columns]
    rows = [
        [get_field_value(image, col["field"], global_config, repository) for col in columns]
        for image in images
    ]
    return render_table(headers, rows)


def consolidate_support_entries(
    entries: list[dict],
    framework_groups: dict[str, list[str]],
    table_order: list[str],
    global_config: dict,
) -> list[dict]:
    """Consolidate support policy entries by framework when GA/EOP dates match.

    Groups entries by framework and version into a single row with the framework display name.

    Args:
        entries: List of support policy entries with framework, version, ga, eop, _repository keys.
        framework_groups: Mapping of group key to list of repository names.
        table_order: List defining repository ordering for positioning consolidated rows.
        global_config: Global configuration containing display_names.

    Returns:
        list[dict]: Consolidated list of entries.

    Raises:
        ValueError: If repositories in a framework group have different GA/EOP dates
            for the same version.
    """
    # Build reverse mapping: repo -> group_key
    repo_to_group = {}
    for group_key, repos in framework_groups.items():
        for repo in repos:
            repo_to_group[repo] = group_key

    # Separate grouped and ungrouped entries
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    ungrouped = []

    for entry in entries:
        repo = entry["_repository"]
        if repo in repo_to_group:
            key = (repo_to_group[repo], entry["version"])
            grouped[key].append(entry)
        else:
            ungrouped.append(entry)

    # Process each group
    result = list(ungrouped)
    for (group_key, version), group_entries in grouped.items():
        # Validate all entries have same GA/EOP dates
        first = group_entries[0]
        for entry in group_entries[1:]:
            validate_date_consistency(
                first["ga"],
                first["eop"],
                entry["ga"],
                entry["eop"],
                f"{get_display_name(global_config, group_key)} {version} "
                f"({first['_repository']} vs {entry['_repository']})",
            )

        # Consolidate to single row with display name
        display_name = get_display_name(global_config, group_key)
        first_repo = min(group_entries, key=lambda e: table_order.index(e["_repository"]))
        result.append(
            {
                "framework": display_name,
                "version": version,
                "ga": first["ga"],
                "eop": first["eop"],
                "_repository": first_repo["_repository"],
            }
        )

    return result


def parse_version(version_str: str | None) -> Version:
    """Parse version string safely, returning Version("0") on failure.

    Args:
        version_str: Version string to parse, or None.

    Returns:
        Version: Parsed version, or Version("0") if parsing fails.
    """
    try:
        return Version(version_str or "0")
    except Exception:
        return Version("0")


def make_support_policy_entry(
    framework: str, version: str, ga: str, eop: str, repository: str
) -> dict:
    """Create a support policy entry dict.

    Args:
        framework: Display name for the framework.
        version: Framework version.
        ga: General Availability date.
        eop: End of Patch date.
        repository: Repository name (used for sorting/grouping).

    Returns:
        dict: Support policy entry with framework, version, ga, eop, _repository keys.
    """
    return {
        "framework": framework,
        "version": version,
        "ga": ga,
        "eop": eop,
        "_repository": repository,
    }


def get_latest_image(repo: str, platform: str) -> str:
    """Get the latest image URI for a repository and platform.

    Finds the image with the highest version number matching the platform.

    Args:
        repo: Name of the repository.
        platform: Platform type (e.g., 'ec2', 'sagemaker').

    Returns:
        str: Full ECR image URI with us-west-2 region.

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

    latest = max(matching, key=lambda img: parse_version(img.get("version")))
    account = latest.get("example_ecr_account", global_config["example_ecr_account"])
    return f"{account}.dkr.ecr.us-west-2.amazonaws.com/{repo}:{latest['tag']}"
