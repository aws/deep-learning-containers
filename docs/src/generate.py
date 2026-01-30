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
from pathlib import Path

from constants import (
    AVAILABLE_IMAGES_TABLE_HEADER,
    GLOBAL_CONFIG,
    PUBLIC_GALLERY_URL,
    REFERENCE_DIR,
    RELEASE_NOTES_DIR,
    RELEASE_NOTES_TABLE_HEADER,
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
from sorter import accelerator_sorter, platform_sorter, repository_sorter
from utils import (
    get_framework_order,
    load_jinja2,
    load_table_config,
    parse_version,
    render_table,
    write_output,
)

LOGGER = logging.getLogger(__name__)


def _generate_individual_release_note(
    img: ImageConfig, template: Template, output_dir: Path, dry_run: bool = False
) -> str:
    """Generate a single release note page for an image."""
    content = template.render(
        title=f"{img.display_repository} {img.version} on {img.display_platform}",
        framework=img.get("framework"),
        version=img.get("version"),
        platform_display=img.display_platform,
        announcements=img.get("announcements", []),
        packages=img.get("packages", {}),
        image_uris=img.get_image_uris(),
        optional=img.get("optional", {}),
        **GLOBAL_CONFIG,
    )

    if not dry_run:
        output_path = output_dir / img.release_note_filename
        write_output(output_path, content)
        LOGGER.debug(f"Wrote {output_path}")

    return content


def _generate_framework_index(
    framework_group: str,
    images: list[ImageConfig],
    template: Template,
    table_config: dict,
    output_dir: Path,
    dry_run: bool = False,
) -> str:
    """Generate the index page for a framework group's release notes."""
    display_names = GLOBAL_CONFIG.get("display_names", {})
    framework_display = display_names.get(framework_group, framework_group)
    columns = table_config.get("columns", [])
    headers = [col["header"] for col in columns]

    # Group images by major.minor version
    images_by_version: dict[str, list[ImageConfig]] = {}
    for img in images:
        ver = parse_version(img.version)
        major_minor = f"{ver.major}.{ver.minor}"
        images_by_version.setdefault(major_minor, []).append(img)

    # Sort framework versions descending
    sorted_versions = sorted(images_by_version.keys(), key=parse_version, reverse=True)

    # Build version-separated content for supported and deprecated
    supported_sections, deprecated_sections = [], []
    for version in sorted_versions:
        sorted_images = sort_by_version(
            images_by_version[version],
            tiebreakers=[repository_sorter, platform_sorter, accelerator_sorter],
        )

        supported = [img for img in sorted_images if img.is_supported]
        deprecated = [img for img in sorted_images if not img.is_supported]

        if supported:
            table = render_table(headers, [build_image_row(img, columns) for img in supported])
            supported_sections.append(
                f"{RELEASE_NOTES_TABLE_HEADER} {framework_display} {version}\n\n{table}"
            )

        if deprecated:
            table = render_table(headers, [build_image_row(img, columns) for img in deprecated])
            deprecated_sections.append(
                f"{RELEASE_NOTES_TABLE_HEADER} {framework_display} {version}\n\n{table}"
            )

    content = template.render(
        framework_display=framework_display,
        supported_content="\n\n".join(supported_sections),
        deprecated_content="\n\n".join(deprecated_sections),
        **GLOBAL_CONFIG,
    )

    if not dry_run:
        write_output(output_dir / "index.md", content)
        LOGGER.debug(f"Wrote {output_dir / 'index.md'}")

    return content


def generate_release_notes(dry_run: bool = False) -> None:
    """Generate release notes from image configs with release notes fields."""
    LOGGER.debug("Generating release notes")

    release_template = Template(
        load_jinja2(TEMPLATES_DIR / "releasenotes" / "release_note.template.md")
    )
    index_template = Template(load_jinja2(TEMPLATES_DIR / "releasenotes" / "index.template.md"))
    table_config = load_table_config("extra/release_notes")

    images_by_group = load_images_by_framework_group(lambda img: img.has_release_notes)
    if not images_by_group:
        LOGGER.info("No release notes to generate")
        return

    for framework_group in get_framework_order():
        group_images = images_by_group.get(framework_group, [])
        if not group_images:
            continue

        group_dir = RELEASE_NOTES_DIR / framework_group
        if not dry_run:
            group_dir.mkdir(parents=True, exist_ok=True)

        for img in group_images:
            _generate_individual_release_note(img, release_template, group_dir, dry_run)

        _generate_framework_index(
            framework_group, group_images, index_template, table_config, group_dir, dry_run
        )

    LOGGER.info("Generated release notes")


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

    for framework_group in get_framework_order():
        images = images_by_group.get(framework_group, [])
        if not images:
            continue

        # Group by major.minor, then decide display format based on date consistency
        major_minor_groups: dict[str, list[ImageConfig]] = {}
        for img in images:
            v = parse_version(img.version)
            major_minor = f"{v.major}.{v.minor}"
            major_minor_groups.setdefault(major_minor, []).append(img)

        version_map: dict[str, ImageConfig] = {}
        for major_minor, group in major_minor_groups.items():
            # Check if all images in group have same ga/eop
            first = group[0]
            all_same_dates = all(img.ga == first.ga and img.eop == first.eop for img in group)

            if all_same_dates:
                # Consolidate to major.minor display
                version_map[major_minor] = first
            else:
                # Keep full versions, warn about inconsistency
                versions_info = ", ".join(f"{img.version} ({img.ga}, {img.eop})" for img in group)
                LOGGER.warning(
                    f"Different GA/EOP dates for {framework_group} patch versions: {versions_info}"
                )
                # Keep each patch version as separate row with full version display
                for img in group:
                    existing = version_map.get(img.version)
                    # Error if same full version (e.g., X.Y.Z) has different dates across images
                    if existing and (existing.ga != img.ga or existing.eop != img.eop):
                        raise ValueError(
                            f"Inconsistent dates for {framework_group} {img.version}: \n"
                            f"\tExisting: {existing._repository}-{existing.version}-{existing.accelerator}-{existing.platform}\n"
                            f"\tImage: {img._repository}-{img.version}-{img.accelerator}-{img.platform}\n"
                            f"\t({existing.ga}, {existing.eop}) vs ({img.ga}, {img.eop})"
                        )
                    # Deduplicate same full version with same dates
                    version_map[img.version] = img

        # Merge legacy entries for this framework
        for legacy_img in legacy_data.get(framework_group, []):
            if legacy_img.version not in version_map:
                version_map[legacy_img.version] = legacy_img

        # Sort by version descending within this framework group
        # Key is the display version (major.minor if consolidated, full version otherwise)
        sorted_keys = sorted(
            version_map.keys(), key=lambda k: parse_version(version_map[k].version), reverse=True
        )
        for key in sorted_keys:
            img = version_map[key]
            (supported if img.is_supported else unsupported).append((img, key))

    # Build tables
    table_config = load_table_config("extra/support_policy")
    columns = table_config.get("columns", [])
    headers = [col["header"] for col in columns]

    supported_table = render_table(
        headers, [build_image_row(img, columns, {"version": ver}) for img, ver in supported]
    )
    unsupported_table = render_table(
        headers, [build_image_row(img, columns, {"version": ver}) for img, ver in unsupported]
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
            url = f"{PUBLIC_GALLERY_URL}/{repository}"
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
    generate_release_notes(dry_run)

    LOGGER.info("Documentation generation complete")
