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

import sorter as sorter_module
from constants import (
    AVAILABLE_IMAGES_TABLE_HEADER,
    DOCS_DIR,
    GLOBAL_CONFIG,
    PUBLIC_GALLERY_URL,
    README_PATH,
    REFERENCE_DIR,
    RELEASE_NOTES_DIR,
    RELEASE_NOTES_TABLE_HEADER,
    SITE_URL,
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
from utils import (
    get_framework_order,
    load_jinja2,
    load_table_config,
    parse_version,
    render_table,
    write_output,
)

DEFAULT_TIEBREAKERS = [sorter_module.platform_sorter, sorter_module.accelerator_sorter]


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
            tiebreakers=[
                sorter_module.repository_sorter,
                sorter_module.platform_sorter,
                sorter_module.accelerator_sorter,
            ],
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

        # Group by (major.minor, ga, eop) to allow different dates for same version
        # This enables training and inference to have different EOP dates
        date_groups: dict[tuple[str, str, str], list[ImageConfig]] = {}
        for img in images:
            v = parse_version(img.version)
            major_minor = f"{v.major}.{v.minor}"
            key = (major_minor, img.ga, img.eop)
            date_groups.setdefault(key, []).append(img)

        # Track which versions have multiple date groups (need repository-specific display)
        version_date_count: dict[str, int] = {}
        for (major_minor, ga, eop), group in date_groups.items():
            version_date_count[major_minor] = version_date_count.get(major_minor, 0) + 1

        version_map: dict[str, tuple[ImageConfig, bool]] = {}

        # Process each unique (version, ga, eop) combination
        for (major_minor, ga, eop), group in date_groups.items():
            # Check if all images in this date group have the same full version
            versions_in_group = {img.version for img in group}

            # Determine if this version needs repository-specific display
            needs_repo_display = version_date_count[major_minor] > 1

            if len(versions_in_group) == 1:
                # All images have same patch version
                first = group[0]

                if needs_repo_display:
                    # Same version exists with different dates - use repository-specific display
                    # Store with flag indicating we need to override framework_group display
                    repos_in_group = {img._repository for img in group}
                    # Create a unique key for this date group
                    repo_suffix = "-".join(sorted(repos_in_group))
                    display_key = f"{major_minor}:{repo_suffix}"
                    version_map[display_key] = (first, True)  # True = use repo display
                else:
                    # No conflict - use simple major.minor key with framework display
                    version_map[major_minor] = (first, False)  # False = use framework display
            else:
                # Multiple patch versions with same dates - warn and keep separate
                versions_info = ", ".join(sorted(versions_in_group))
                LOGGER.warning(
                    f"Different patch versions for {framework_group} with same GA/EOP dates: {versions_info}"
                )
                for img in group:
                    version_map[img.version] = (img, needs_repo_display)

        # Merge legacy entries for this framework
        for legacy_img in legacy_data.get(framework_group, []):
            if legacy_img.version not in version_map:
                version_map[legacy_img.version] = (
                    legacy_img,
                    False,
                )  # Legacy uses framework display

        # Sort by version descending within this framework group
        # Extract version for sorting from the tuple
        sorted_keys = sorted(
            version_map.keys(), key=lambda k: parse_version(version_map[k][0].version), reverse=True
        )
        for key in sorted_keys:
            img, use_repo_display = version_map[key]
            # Extract clean version for display (remove repo suffix if present)
            display_version = key.split(":")[0] if ":" in key else key
            (supported if img.is_supported else unsupported).append(
                (img, display_version, use_repo_display)
            )

    # Build tables
    table_config = load_table_config("extra/support_policy")
    columns = table_config.get("columns", [])
    headers = [col["header"] for col in columns]

    # Build rows with appropriate framework display
    supported_rows = []
    for img, ver, use_repo_display in supported:
        overrides = {"version": ver}
        if use_repo_display:
            # Find all repositories in this framework group with this version and same dates
            # to create a comprehensive display name
            all_repos_with_dates = [
                i
                for i in images_by_group.get(img.framework_group, [])
                if parse_version(i.version).major == parse_version(img.version).major
                and parse_version(i.version).minor == parse_version(img.version).minor
                and i.ga == img.ga
                and i.eop == img.eop
            ]
            unique_repos = sorted(set(i._repository for i in all_repos_with_dates))
            display_names = GLOBAL_CONFIG.get("display_names", {})

            # Determine the common prefix (e.g., "PyTorch") and suffix (e.g., "Training", "Inference")
            repo_displays = [display_names.get(repo, repo) for repo in unique_repos]

            # If all repos share a common framework prefix, consolidate intelligently
            # e.g., ["PyTorch Training", "PyTorch Training ARM64"] -> "PyTorch Training"
            # e.g., ["PyTorch Inference", "PyTorch Inference ARM64"] -> "PyTorch Inference"
            if len(repo_displays) > 1:
                # Check if we can consolidate (e.g., remove ARM64 variants)
                base_displays = set()
                for display in repo_displays:
                    # Remove " ARM64" suffix if present
                    base = display.replace(" ARM64", "").strip()
                    base_displays.add(base)

                if len(base_displays) == 1:
                    # All are variants of the same base (e.g., all "PyTorch Training")
                    overrides["framework_group"] = base_displays.pop()
                else:
                    # Multiple different bases - show them all
                    overrides["framework_group"] = ", ".join(sorted(base_displays))
            else:
                overrides["framework_group"] = repo_displays[0]
        supported_rows.append(build_image_row(img, columns, overrides))

    unsupported_rows = []
    for img, ver, use_repo_display in unsupported:
        overrides = {"version": ver}
        if use_repo_display:
            # Find all repositories in this framework group with this version and same dates
            all_repos_with_dates = [
                i
                for i in images_by_group.get(img.framework_group, [])
                if parse_version(i.version).major == parse_version(img.version).major
                and parse_version(i.version).minor == parse_version(img.version).minor
                and i.ga == img.ga
                and i.eop == img.eop
            ]
            unique_repos = sorted(set(i._repository for i in all_repos_with_dates))
            display_names = GLOBAL_CONFIG.get("display_names", {})

            repo_displays = [display_names.get(repo, repo) for repo in unique_repos]

            if len(repo_displays) > 1:
                base_displays = set()
                for display in repo_displays:
                    base = display.replace(" ARM64", "").strip()
                    base_displays.add(base)

                if len(base_displays) == 1:
                    overrides["framework_group"] = base_displays.pop()
                else:
                    overrides["framework_group"] = ", ".join(sorted(base_displays))
            else:
                overrides["framework_group"] = repo_displays[0]
        unsupported_rows.append(build_image_row(img, columns, overrides))

    supported_table = render_table(headers, supported_rows)
    unsupported_table = render_table(headers, unsupported_rows)

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

        # Sort images by version desc with tiebreakers from config or defaults
        tiebreaker_names = table_config.get("tiebreakers")
        tiebreakers = (
            [getattr(sorter_module, name) for name in tiebreaker_names]
            if tiebreaker_names
            else DEFAULT_TIEBREAKERS
        )
        images = sort_by_version(images, tiebreakers=tiebreakers)

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


def generate_index(dry_run: bool = False) -> str:
    """Generate docs/index.md from README.md content."""
    output_path = DOCS_DIR / "index.md"
    template_path = TEMPLATES_DIR / "index.template.md"
    LOGGER.debug(f"Generating {output_path}")

    readme_content = README_PATH.read_text()
    readme_content = readme_content.replace(SITE_URL, "")

    # Expand single logo into MkDocs theme-aware light/dark logos
    readme_logo = '<img src="assets/logos/AWS_logo_RGB.svg" alt="AWS Logo" width="30%">'
    mkdocs_logos = (
        '<img src="assets/logos/AWS_logo_RGB.svg#only-light" alt="AWS Logo" width="30%">\n'
        '<img src="assets/logos/AWS_logo_RGB_REV.svg#only-dark" alt="AWS Logo" width="30%">'
    )
    readme_content = readme_content.replace(readme_logo, mkdocs_logos)

    template = Template(load_jinja2(template_path))
    content = template.render(readme_content=readme_content)

    if not dry_run:
        write_output(output_path, content)
        LOGGER.debug(f"Wrote {output_path}")

    LOGGER.info("Generated index.md")
    return content


def generate_all(dry_run: bool = False) -> None:
    """Generate all documentation files."""
    LOGGER.info("Loaded global config")

    generate_index(dry_run)
    generate_support_policy(dry_run)
    generate_available_images(dry_run)
    generate_release_notes(dry_run)

    LOGGER.info("Documentation generation complete")
