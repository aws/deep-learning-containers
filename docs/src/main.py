"""Documentation generation entry point.

Usage:
    python docs/src/main.py [--dry-run] [--verbose] [--support-policy-only] [--available-images-only]

MkDocs hook:
    Add to mkdocs.yaml: hooks: [docs/src/main.py]
"""

import argparse
import os

from tables import IMAGE_TABLE_GENERATORS, support_policy_table
from utils import load_yaml, read_template, write_output

# Resolve paths relative to this file
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.dirname(SRC_DIR)
DATA_FILE = os.path.join(SRC_DIR, "data", "images.yml")
REFERENCE_DIR = os.path.join(DOCS_DIR, "reference")


def generate_support_policy(yaml_data: dict, dry_run: bool = False, verbose: bool = False) -> str:
    """Generate support_policy.md from template and YAML data."""
    template = read_template(os.path.join(REFERENCE_DIR, "support_policy.template.md"))
    tables = support_policy_table.generate(yaml_data)
    content = template + "\n\n" + tables

    if verbose:
        print("Generated support_policy.md")

    if not dry_run:
        write_output(os.path.join(REFERENCE_DIR, "support_policy.md"), content)
    elif verbose:
        print(content)

    return content


def generate_available_images(yaml_data: dict, dry_run: bool = False, verbose: bool = False) -> str:
    """Generate available_images.md from template and YAML data."""
    template = read_template(os.path.join(REFERENCE_DIR, "available_images.template.md"))

    sections = []
    for generator in IMAGE_TABLE_GENERATORS:
        output = generator.generate(yaml_data)
        if output:
            sections.append(output)

    content = template + "\n\n" + "\n\n".join(sections)

    if verbose:
        print("Generated available_images.md")

    if not dry_run:
        write_output(os.path.join(REFERENCE_DIR, "available_images.md"), content)
    elif verbose:
        print(content)

    return content


def generate_all(dry_run: bool = False, verbose: bool = False) -> None:
    """Generate all documentation files."""
    yaml_data = load_yaml(DATA_FILE)

    if verbose:
        print(f"Loaded data from {DATA_FILE}")

    generate_support_policy(yaml_data, dry_run, verbose)
    generate_available_images(yaml_data, dry_run, verbose)

    if verbose:
        print("Documentation generation complete")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate DLC documentation from YAML source")
    parser.add_argument("--dry-run", action="store_true", help="Print output without writing files")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
    parser.add_argument(
        "--support-policy-only", action="store_true", help="Generate only support_policy.md"
    )
    parser.add_argument(
        "--available-images-only", action="store_true", help="Generate only available_images.md"
    )
    args = parser.parse_args()

    yaml_data = load_yaml(DATA_FILE)

    if args.verbose:
        print(f"Loaded data from {DATA_FILE}")

    if args.support_policy_only:
        generate_support_policy(yaml_data, args.dry_run, args.verbose)
    elif args.available_images_only:
        generate_available_images(yaml_data, args.dry_run, args.verbose)
    else:
        generate_support_policy(yaml_data, args.dry_run, args.verbose)
        generate_available_images(yaml_data, args.dry_run, args.verbose)

    if args.verbose:
        print("Done")


# MkDocs hook entry point
def on_startup(command=["build", "gh-deploy", "serve"], dirty=False):
    """MkDocs hook - runs before build."""
    generate_all(dry_run=False, verbose=False)


if __name__ == "__main__":
    main()
