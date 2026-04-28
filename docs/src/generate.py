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
    dates_agree,
    load_images_by_framework_group,
    load_legacy_images,
    load_repository_images,
    sort_by_version,
)
from jinja2 import Template
from utils import (
    build_repo_map,
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
    framework_display = GLOBAL_CONFIG["display_names"].get(framework_group, framework_group)
    columns = table_config["columns"]
    headers = [col["header"] for col in columns]

    # Split images by OS: AL2023 images first, then legacy (Ubuntu etc.)
    amzn2023_images = [img for img in images if img.get("os", "") == "amzn2023"]
    legacy_images = [img for img in images if img.get("os", "") != "amzn2023"]

    # Group images by major.minor version within each OS group
    def _group_by_version(img_list):
        by_version: dict[str, list[ImageConfig]] = {}
        for img in img_list:
            ver = parse_version(img.version)
            major_minor = f"{ver.major}.{ver.minor}"
            by_version.setdefault(major_minor, []).append(img)
        return by_version

    # Build ordered version list: AL2023 versions first (descending), then legacy (descending)
    amzn2023_by_version = _group_by_version(amzn2023_images)
    legacy_by_version = _group_by_version(legacy_images)
    sorted_versions = sorted(amzn2023_by_version.keys(), key=parse_version, reverse=True) + sorted(
        legacy_by_version.keys(), key=parse_version, reverse=True
    )
    images_by_version = {**amzn2023_by_version, **legacy_by_version}

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

        # Add OS label to header if images have mixed OS types in this framework
        os_label = ""
        version_display = version
        if amzn2023_images and legacy_images:
            sample_os = sorted_images[0].get("os", "")
            if sample_os == "amzn2023":
                os_label = " (Amazon Linux 2023)"
                version_display = f"v{version.split('.')[0]}"
            else:
                os_label = " (Ubuntu)"

        if supported:
            table = render_table(headers, [build_image_row(img, columns) for img in supported])
            supported_sections.append(
                f"{RELEASE_NOTES_TABLE_HEADER} {framework_display} {version_display}{os_label}\n\n{table}"
            )

        if deprecated:
            table = render_table(headers, [build_image_row(img, columns) for img in deprecated])
            deprecated_sections.append(
                f"{RELEASE_NOTES_TABLE_HEADER} {framework_display} {version_display}{os_label}\n\n{table}"
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


def _consolidate_framework_version(
    framework_group: str,
    full_ver: str,
    repo_imgs: list[ImageConfig],
) -> list[tuple[ImageConfig, dict[str, str]]]:
    """Consolidate images for a single framework version using hierarchical date agreement.

    Tries three levels of consolidation, stopping at the first that succeeds:
      1. Framework group — all repos agree → single row
      2. Sub-group — repos within a sub-group agree → one row per sub-group (nested groups only)
      3. Per-repo — no agreement → one row per repository
    """
    # All repos agree → single framework-level row (Level 1 consolidation)
    if dates_agree(repo_imgs):
        return [(repo_imgs[0], {"version": full_ver})]

    LOGGER.warning(
        f"GA/EOP mismatch in {framework_group} {full_ver} across repositories. "
        f"Splitting into sub-group/repository rows."
    )

    group_config = GLOBAL_CONFIG.get("framework_groups", {}).get(framework_group, [])

    # Flat group — no sub-groups, fall back directly to per-repo rows (Level 3 consolidation)
    if not isinstance(group_config, dict):
        return [
            (img, {"version": full_ver, "framework_group": img.display_repository})
            for img in repo_imgs
        ]

    # Nested group — try sub-group consolidation (Level 2 consolidation), per-repo fallback (Level 3 consolidation)
    repo_to_subgroup = build_repo_map(group_config)
    subgroup_imgs: dict[str, list[ImageConfig]] = {}
    for img in repo_imgs:
        subgroup_name = repo_to_subgroup.get(img._repository, img._repository)
        subgroup_imgs.setdefault(subgroup_name, []).append(img)

    entries: list[tuple[ImageConfig, dict[str, str]]] = []
    for subgroup_name, images in subgroup_imgs.items():
        if dates_agree(images):
            display_name = GLOBAL_CONFIG["display_names"].get(subgroup_name, subgroup_name)
            entries.append((images[0], {"version": full_ver, "framework_group": display_name}))
        else:
            entries.extend(
                (img, {"version": full_ver, "framework_group": img.display_repository})
                for img in images
            )

    return entries


def _collapse_minor_versions(
    entries: list[tuple[ImageConfig, dict[str, str]]],
) -> list[tuple[ImageConfig, dict[str, str]]]:
    """Collapse patch versions (e.g., A.B.C, A.B.D) into major.minor (A.B) when all share identical dates.

    Skips any major.minor that has split (per-repo) rows, since mixing collapsed and split rows
    under the same major.minor would be confusing.

    Args:
        entries: List of (image, overrides) tuples. Split rows have "framework_group" in overrides.

    Returns:
        New list with collapsible groups replaced by a single major.minor entry.
    """
    uncollapsible: set[str] = set()
    collapsible_groups: dict[str, list[int]] = {}
    for idx, (img, overrides) in enumerate(entries):
        version_obj = parse_version(img.version)
        mm = f"{version_obj.major}.{version_obj.minor}"
        if "framework_group" in overrides:
            # Find major.minors that have split rows by repository — these cannot be collapsed
            # Collapsing split rows will create ambiguity between patch versions
            uncollapsible.add(mm)
        else:
            collapsible_groups.setdefault(mm, []).append(idx)

    # Collapse: if all entries in a major.minor group share dates, keep one with major.minor display
    for mm, indices in collapsible_groups.items():
        if mm in uncollapsible:
            continue
        if len(indices) < 2:
            continue
        group_imgs = [entries[idx][0] for idx in indices]
        ref_img = group_imgs[0]
        if dates_agree(group_imgs):
            entries[indices[0]] = (ref_img, {"version": mm})
            for idx in indices[1:]:
                entries[idx] = None  # mark duplicates for removal
        else:
            LOGGER.warning(
                f"Cannot collapse {ref_img._repository} {mm}. "
                f"Please confirm images GA/EOP dates within this framework are intentional."
            )

    return [e for e in entries if e is not None]


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

        # Step 1: Group by full version, deduplicate per repo
        version_entries: dict[str, list[ImageConfig]] = {}
        for img in images:
            bucket = version_entries.setdefault(img.version, [])
            if not any(existing._repository == img._repository for existing in bucket):
                bucket.append(img)

        # Step 2: Consolidate across repos (framework → sub-group → per-repo fallback)
        entries: list[tuple[ImageConfig, dict[str, str]]] = []
        for full_ver, repo_imgs in version_entries.items():
            entries.extend(_consolidate_framework_version(framework_group, full_ver, repo_imgs))

        # Step 3: Collapse patch versions into major.minor where possible
        entries = _collapse_minor_versions(entries)

        # Merge legacy entries for this framework
        for legacy_img in legacy_data.get(framework_group, []):
            entries.append((legacy_img, {"version": legacy_img.version}))

        # Sort by version descending within this framework group
        entries.sort(key=lambda e: parse_version(e[0].version), reverse=True)
        for img, overrides in entries:
            (supported if img.is_supported else unsupported).append((img, overrides))

    # Build tables
    table_config = load_table_config("extra/support_policy")
    columns = table_config["columns"]
    headers = [col["header"] for col in columns]

    supported_table = render_table(
        headers, [build_image_row(img, columns, overrides) for img, overrides in supported]
    )
    unsupported_table = render_table(
        headers, [build_image_row(img, columns, overrides) for img, overrides in unsupported]
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

    display_names = GLOBAL_CONFIG["display_names"]
    table_order = GLOBAL_CONFIG["table_order"]
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
        columns = table_config["columns"]
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
        if table_config.get("note"):
            section += f"\n{table_config['note']}\n"
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
    # Fix empty links left after stripping SITE_URL (e.g. [text](SITE_URL) -> [text]())
    readme_content = readme_content.replace("]()", "](./)")
    readme_content = readme_content.replace('href=""', 'href="./"')

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
