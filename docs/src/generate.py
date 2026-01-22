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
"""Documentation generation functions."""

import logging

from constants import AVAILABLE_IMAGES_TABLE_HEADER, GLOBAL_CONFIG, REFERENCE_DIR, TEMPLATES_DIR
from image_config import (
    ImageConfig,
    build_image_row,
    check_public_registry,
    load_legacy_images,
    load_repository_images,
    sort_by_version,
)
from jinja2 import Template
from sorter import accelerator_sorter, platform_sorter
from utils import (
    get_framework_order,
    load_jinja2,
    load_table_config,
    render_table,
    write_output,
)

LOGGER = logging.getLogger(__name__)


def generate_support_policy(dry_run: bool = False) -> str:
    """Generate support_policy.md from image configs with GA/EOP dates."""
    output_path = REFERENCE_DIR / "support_policy.md"
    template_path = TEMPLATES_DIR / "reference" / "support_policy.template.md"
    LOGGER.debug(f"Generating {output_path}")

    framework_groups = GLOBAL_CONFIG.get("framework_groups", {})
    legacy_data = load_legacy_images()

    supported, unsupported = [], []

    for framework_key in get_framework_order():
        # Get repos for this framework (group or single repo)
        repos = framework_groups.get(framework_key, [framework_key])

        # Load images with support dates from all repos in group
        images = []
        for repo in repos:
            images.extend(img for img in load_repository_images(repo) if img.has_support_dates)

        # Deduplicate by version, validating date consistency
        version_map: dict[str, ImageConfig] = {}
        for img in images:
            existing = version_map.get(img.version)
            if existing and (existing.ga != img.ga or existing.eop != img.eop):
                raise ValueError(
                    f"Inconsistent dates for {framework_key} {img.version}: "
                    f"({existing.ga}, {existing.eop}) vs ({img.ga}, {img.eop})"
                )
            version_map[img.version] = img

        # Merge legacy entries for this framework
        for legacy_img in legacy_data.get(framework_key, []):
            if legacy_img.version not in version_map:
                version_map[legacy_img.version] = legacy_img

        # Sort by version descending and separate supported/unsupported
        for img in sort_by_version(list(version_map.values())):
            (supported if img.is_supported else unsupported).append(img)

    # Build tables
    table_config = load_table_config("support_policy")
    columns = table_config.get("columns", [])
    headers = [col["header"] for col in columns]

    supported_table = render_table(headers, [build_image_row(img, columns) for img in supported])
    unsupported_table = render_table(
        headers, [build_image_row(img, columns) for img in unsupported]
    )

    # Render template
    template = Template(load_jinja2(template_path))
    content = template.render(
        supported_table=supported_table,
        unsupported_table=unsupported_table,
        **GLOBAL_CONFIG,
    )

    if not dry_run:
        write_output(output_path, content)
        LOGGER.debug(f"Wrote {output_path}")

    LOGGER.info("Generated support_policy.md")
    return content


def generate_available_images(dry_run: bool = False) -> str:
    """Generate available_images.md from image configs and table configs."""
    output_path = REFERENCE_DIR / "available_images.md"
    template_path = TEMPLATES_DIR / "reference" / "available_images.template.md"
    LOGGER.debug(f"Generating {output_path}")

    display_names = GLOBAL_CONFIG.get("display_names", {})
    table_order = GLOBAL_CONFIG.get("table_order", [])
    tables_content = []

    for repository in table_order:
        images = [img for img in load_repository_images(repository) if img.is_supported]
        if not images:
            continue

        try:
            table_config = load_table_config(repository)
        except FileNotFoundError:
            LOGGER.warning(f"No table config for {repository}, skipping")
            continue

        display_name = display_names[repository]
        columns = table_config.get("columns", [])
        has_public_registry = check_public_registry(images, repository)

        # Sort images by version desc, platform, accelerator
        images = sort_by_version(images, tiebreakers=[platform_sorter, accelerator_sorter])

        # Build table
        headers = [col["header"] for col in columns]
        rows = [build_image_row(img, columns) for img in images]

        section = f"{AVAILABLE_IMAGES_TABLE_HEADER} {display_name}\n"
        if has_public_registry:
            url = f"{GLOBAL_CONFIG['public_gallery_url']}/{repository}"
            section += (
                f"\nThese images are also available in ECR Public Gallery: [{repository}]({url})\n"
            )
        section += f"\n{render_table(headers, rows)}"
        tables_content.append(section)

    # Render template
    template = Template(load_jinja2(template_path))
    content = template.render(
        tables_content="\n\n".join(tables_content),
        **GLOBAL_CONFIG,
    )

    if not dry_run:
        write_output(output_path, content)
        LOGGER.debug(f"Wrote {output_path}")

    LOGGER.info("Generated available_images.md")
    return content


def generate_all(dry_run: bool = False) -> None:
    """Generate all documentation files."""
    LOGGER.info("Loaded global config")

    generate_support_policy(dry_run)
    generate_available_images(dry_run)

    LOGGER.info("Documentation generation complete")
