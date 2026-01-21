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
from datetime import date

from constants import AVAILABLE_IMAGES_TABLE_HEADER, REFERENCE_DIR, TEMPLATES_DIR
from jinja2 import Template
from packaging.version import Version
from utils import (
    build_image_table,
    build_public_registry_note,
    check_public_registry,
    get_display_name,
    group_images_by_version,
    load_all_image_configs,
    load_global_config,
    load_image_configs,
    load_table_config,
    read_template,
    write_output,
)

LOGGER = logging.getLogger(__name__)


def generate_support_policy(global_config: dict, dry_run: bool = False) -> str:
    """Generate support_policy.md from image configs with GA/EOP dates."""
    output_path = REFERENCE_DIR / "support_policy.md"
    template_path = TEMPLATES_DIR / "reference" / "support_policy.template.md"
    LOGGER.debug(f"Generating {output_path}")

    all_images = load_all_image_configs()
    today = date.today()
    version_data = group_images_by_version(all_images)

    # Separate supported and unsupported
    supported = []
    unsupported = []
    table_order = global_config.get("table_order", [])

    for (repository, version), data in version_data.items():
        eop_date = date.fromisoformat(data["eop"])
        display_name = get_display_name(global_config, repository)

        entry = {
            "framework": display_name,
            "version": version,
            "ga": data["ga"],
            "eop": data["eop"],
            "_repository": repository,
        }

        if eop_date >= today:
            supported.append(entry)
        else:
            unsupported.append(entry)

    # Sort by table_order, then by version descending
    def sort_key(item):
        repo = item["_repository"]
        if repo in table_order:
            order = table_order.index(repo)
        else:
            raise ValueError(
                f"Table {repo} does not exists within the table order in global configuration."
            )
        try:
            ver = Version(item["version"])
        except Exception:
            ver = Version("0")
        return (order, -ver.major, -ver.minor, -ver.micro)

    supported.sort(key=sort_key)
    unsupported.sort(key=sort_key)

    # Render template
    template_content = read_template(template_path)
    template = Template(template_content)
    content = template.render(
        supported=supported,
        unsupported=unsupported,
        **global_config,
    )

    if not dry_run:
        write_output(output_path, content)
        LOGGER.debug(f"Wrote {output_path}")
    else:
        LOGGER.debug("Dry run - skipping write")

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
        images = load_image_configs(repository)
        if not images:
            continue

        try:
            table_config = load_table_config(repository)
        except FileNotFoundError:
            LOGGER.warning(f"No table config for {repository}, skipping")
            continue

        display_name = get_display_name(global_config, repository)
        columns = table_config.get("columns", [])
        has_public_registry = check_public_registry(images, repository)

        # Sort images: version desc, platform (sagemaker before ec2), accelerator (gpu before cpu)
        def sort_key(img):
            try:
                ver = Version(img.get("version"))
            except Exception:
                ver = Version("0")
            platform_order = 0 if img.get("platform") == "sagemaker" else 1
            accel = img.get("accelerator", "").lower()
            accel_order = 0 if accel == "gpu" else 1 if accel == "neuronx" else 2
            # Negate version for descending, keep others ascending
            return (-ver.major, -ver.minor, -ver.micro, platform_order, accel_order)

        images.sort(key=sort_key)

        # Build table section
        section = f"{AVAILABLE_IMAGES_TABLE_HEADER} {display_name}\n"
        if has_public_registry:
            section += f"\n{build_public_registry_note(repository, global_config)}"

        section += f"\n{build_image_table(images, columns, global_config, repository)}"
        tables_content.append(section)

    # Render template with tables
    template_content = read_template(template_path)
    template = Template(template_content)
    content = template.render(
        tables_content="\n\n".join(tables_content),
        **global_config,
    )

    if not dry_run:
        write_output(output_path, content)
        LOGGER.debug(f"Wrote {output_path}")
    else:
        LOGGER.debug("Dry run - skipping write")

    LOGGER.info("Generated available_images.md")
    return content


def generate_all(dry_run: bool = False) -> None:
    """Generate all documentation files."""
    global_config = load_global_config()
    LOGGER.info("Loaded global config")

    generate_support_policy(global_config, dry_run)
    generate_available_images(global_config, dry_run)

    LOGGER.info("Documentation generation complete")
