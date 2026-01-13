"""Documentation generation functions."""

import logging
import os

from tables import IMAGE_TABLE_GENERATORS, support_policy_table
from utils import read_template, write_output

LOGGER = logging.getLogger(__name__)

# Resolve paths relative to this file
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.dirname(SRC_DIR)
DATA_FILE = os.path.join(SRC_DIR, "data", "images.yml")
REFERENCE_DIR = os.path.join(DOCS_DIR, "reference")


def generate_support_policy(yaml_data: dict, dry_run: bool = False) -> str:
    """Generate support_policy.md from template and YAML data."""
    output_path = os.path.join(REFERENCE_DIR, "support_policy.md")
    LOGGER.debug(f"Generating {output_path}")

    template = read_template(os.path.join(REFERENCE_DIR, "support_policy.template.md"))
    tables = support_policy_table.generate(yaml_data)
    content = template + "\n\n" + tables

    if not dry_run:
        write_output(output_path, content)
        LOGGER.debug(f"Wrote {output_path}")
    else:
        LOGGER.debug("Dry run - skipping write")

    LOGGER.info("Generated support_policy.md")

    return content


def generate_available_images(yaml_data: dict, dry_run: bool = False) -> str:
    """Generate available_images.md from template and YAML data."""
    output_path = os.path.join(REFERENCE_DIR, "available_images.md")
    LOGGER.debug(f"Generating {output_path}")

    template = read_template(os.path.join(REFERENCE_DIR, "available_images.template.md"))

    sections = []
    for generator in IMAGE_TABLE_GENERATORS:
        output = generator.generate(yaml_data)
        if output:
            sections.append(output)

    content = template + "\n\n" + "\n\n".join(sections)

    if not dry_run:
        write_output(output_path, content)
        LOGGER.debug(f"Wrote {output_path}")
    else:
        LOGGER.debug("Dry run - skipping write")

    LOGGER.info("Generated available_images.md")

    return content


def generate_all(yaml_data: dict, dry_run: bool = False) -> None:
    """Generate all documentation files."""
    LOGGER.info(f"Loaded data from {DATA_FILE}")

    generate_support_policy(yaml_data, dry_run)
    generate_available_images(yaml_data, dry_run)

    LOGGER.info("Documentation generation complete")
