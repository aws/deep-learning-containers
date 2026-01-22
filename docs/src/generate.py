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
from collections import defaultdict
from datetime import date

from constants import AVAILABLE_IMAGES_TABLE_HEADER, REFERENCE_DIR, TEMPLATES_DIR
from file_loader import load_global_config, load_jinja2, load_legacy_support, load_table_config
from image_config import (
    build_image_row,
    load_all_images,
    load_repository_images,
    sort_images_for_table,
    sort_support_entries,
)
from jinja2 import Template
from utils import (
    build_public_registry_note,
    check_public_registry,
    render_table,
    write_output,
)

LOGGER = logging.getLogger(__name__)


def generate_support_policy(global_config: dict, dry_run: bool = False) -> str:
    """Generate support_policy.md from image configs with GA/EOP dates."""
    output_path = REFERENCE_DIR / "support_policy.md"
    template_path = TEMPLATES_DIR / "reference" / "support_policy.template.md"
    LOGGER.debug(f"Generating {output_path}")

    all_images = load_all_images()
    today = date.today()
    table_order = global_config.get("table_order", [])
    framework_groups = global_config.get("framework_groups", {})

    # Group images by (framework_group or repository, version) and validate dates
    version_data: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"ga": None, "eop": None, "repository": None}
    )

    for repository, images in all_images.items():
        for img in images:
            if not img.has_support_dates():
                continue

            # Use framework group if available, otherwise repository
            group_key = img.get_framework_group(global_config) or repository
            key = (group_key, img.version)
            data = version_data[key]

            # Validate date consistency
            if data["ga"] is not None and (data["ga"] != img.ga or data["eop"] != img.eop):
                raise ValueError(
                    f"Inconsistent dates for {group_key} {img.version}: "
                    f"({data['ga']}, {data['eop']}) vs ({img.ga}, {img.eop})"
                )

            data["ga"] = img.ga
            data["eop"] = img.eop
            # Track first repository for sorting
            if data["repository"] is None:
                data["repository"] = repository

    # Separate supported and unsupported
    supported, unsupported = [], []
    for (group_key, version), data in version_data.items():
        eop_date = date.fromisoformat(data["eop"])
        display_name = global_config["display_names"].get(group_key, group_key)
        # For sorting: use first repo in framework group if it's a group, else the repository
        if group_key in framework_groups:
            sort_repo = framework_groups[group_key][0]
        else:
            sort_repo = data["repository"]
        entry = {
            "framework": display_name,
            "version": version,
            "ga": data["ga"],
            "eop": data["eop"],
            "_sort_repo": sort_repo,
        }
        (supported if eop_date >= today else unsupported).append(entry)

    # Add legacy entries (all unsupported)
    legacy_data = load_legacy_support()
    for (framework, version), data in legacy_data.items():
        display_name = global_config["display_names"].get(framework, framework)
        repos = framework_groups.get(framework, [])
        unsupported.append(
            {
                "framework": display_name,
                "version": version,
                "ga": data["ga"],
                "eop": data["eop"],
                "_sort_repo": repos[0] if repos else framework,
            }
        )

    # Sort by table_order then version descending
    supported = sort_support_entries(supported, table_order)
    unsupported = sort_support_entries(unsupported, table_order)

    # Build tables
    table_config = load_table_config("support_policy")
    columns = table_config.get("columns", [])
    headers = [col["header"] for col in columns]

    supported_rows = [[e[col["field"]] for col in columns] for e in supported]
    unsupported_rows = [[e[col["field"]] for col in columns] for e in unsupported]

    supported_table = render_table(headers, supported_rows)
    unsupported_table = render_table(headers, unsupported_rows)

    # Render template
    template = Template(load_jinja2(template_path))
    content = template.render(
        supported_table=supported_table,
        unsupported_table=unsupported_table,
        **global_config,
    )

    if not dry_run:
        write_output(output_path, content)
        LOGGER.debug(f"Wrote {output_path}")

    LOGGER.info("Generated support_policy.md")
    return content


def generate_available_images(global_config: dict, dry_run: bool = False) -> str:
    """Generate available_images.md from image configs and table configs."""
    output_path = REFERENCE_DIR / "available_images.md"
    template_path = TEMPLATES_DIR / "reference" / "available_images.template.md"
    LOGGER.debug(f"Generating {output_path}")

    table_order = global_config.get("table_order", [])
    tables_content = []

    for repository in table_order:
        images = load_repository_images(repository)
        if not images:
            continue

        # Filter supported images
        images = [img for img in images if img.is_supported()]
        if not images:
            continue

        try:
            table_config = load_table_config(repository)
        except FileNotFoundError:
            LOGGER.warning(f"No table config for {repository}, skipping")
            continue

        display_name = images[0].get_display_name(global_config)
        columns = table_config.get("columns", [])
        has_public_registry = check_public_registry(images, repository)

        # Sort images
        images = sort_images_for_table(images)

        # Build table
        headers = [col["header"] for col in columns]
        rows = [build_image_row(img, columns, global_config) for img in images]

        section = f"{AVAILABLE_IMAGES_TABLE_HEADER} {display_name}\n"
        if has_public_registry:
            section += f"\n{build_public_registry_note(repository, global_config)}"
        section += f"\n{render_table(headers, rows)}"
        tables_content.append(section)

    # Render template
    template = Template(load_jinja2(template_path))
    content = template.render(
        tables_content="\n\n".join(tables_content),
        **global_config,
    )

    if not dry_run:
        write_output(output_path, content)
        LOGGER.debug(f"Wrote {output_path}")

    LOGGER.info("Generated available_images.md")
    return content


def generate_all(dry_run: bool = False) -> None:
    """Generate all documentation files."""
    global_config = load_global_config()
    LOGGER.info("Loaded global config")

    generate_support_policy(global_config, dry_run)
    generate_available_images(global_config, dry_run)

    LOGGER.info("Documentation generation complete")
