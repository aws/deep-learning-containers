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

from constants import (
    AVAILABLE_IMAGES_TABLE_HEADER,
    GLOBAL_CONFIG,
    REFERENCE_DIR,
    RELEASE_NOTES_DIR,
    TEMPLATES_DIR,
)
from image_config import (
    ImageConfig,
    build_image_row,
    check_public_registry,
    load_images_by_framework_group,
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

    legacy_data = load_legacy_images()
    supported, unsupported = [], []

    # Load images with support dates, grouped by framework_group
    images_by_group = load_images_by_framework_group(lambda img: img.has_support_dates)

    if not images_by_group:
        LOGGER.info("No support policy tables to generate")
        return

    for framework_key in get_framework_order():
        images = images_by_group.get(framework_key, [])

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


def generate_release_notes(dry_run: bool = False) -> None:
    """Generate release notes from image configs with release notes fields."""
    LOGGER.debug("Generating release notes")

    template_path = TEMPLATES_DIR / "releasenotes" / "release_note.template.md"
    index_template_path = TEMPLATES_DIR / "releasenotes" / "index.template.md"
    display_names = GLOBAL_CONFIG.get("display_names", {})

    # Load images with release notes, grouped by framework_group
    images_by_group = load_images_by_framework_group(lambda img: img.has_release_notes)

    if not images_by_group:
        LOGGER.info("No release notes to generate")
        return

    release_template = Template(load_jinja2(template_path))
    index_template = Template(load_jinja2(index_template_path))

    for group in get_framework_order():
        images = images_by_group.get(group)
        if not images:
            continue

        group_dir = RELEASE_NOTES_DIR / group
        if not dry_run:
            group_dir.mkdir(parents=True, exist_ok=True)

        # Generate individual release notes
        index_entries: dict[str, list] = {}  # version -> list of entries
        for img in sort_by_version(images, tiebreakers=[platform_sorter, accelerator_sorter]):
            # Determine type from repository name
            repo_type = "Training" if "training" in img.repository else "Inference"
            if "training" not in img.repository and "inference" not in img.repository:
                repo_type = img.display_repository

            content = release_template.render(
                title=f"{img.display_repository} {img.version} on {img.display_platform}",
                framework=img.get("framework"),
                version=img.get("version"),
                platform_display=img.display_platform,
                announcement=img.get("announcement", []),
                packages=img.get("packages", {}),
                image_uris=img.get_image_uris(),
                known_issues=img.get("known_issues"),
                **GLOBAL_CONFIG,
            )

            output_path = group_dir / img.release_note_filename
            if not dry_run:
                write_output(output_path, content)
                LOGGER.debug(f"Wrote {output_path}")

            # Collect for index
            version = img.get("version")
            index_entries.setdefault(version, []).append(
                {
                    "platform": img.display_platform,
                    "type": repo_type,
                    "title": f"{img.display_repository} {version} on {img.display_platform}",
                    "filename": img.release_note_filename,
                }
            )

        # Generate index for this framework group
        index_content = index_template.render(
            framework_display=display_names.get(group, group),
            releases_by_version=index_entries,
            **GLOBAL_CONFIG,
        )

        index_path = group_dir / "index.md"
        if not dry_run:
            write_output(index_path, index_content)
            LOGGER.debug(f"Wrote {index_path}")

    LOGGER.info("Generated release notes")


def generate_all(dry_run: bool = False) -> None:
    """Generate all documentation files."""
    LOGGER.info("Loaded global config")

    generate_support_policy(dry_run)
    generate_available_images(dry_run)
    generate_release_notes(dry_run)

    LOGGER.info("Documentation generation complete")
