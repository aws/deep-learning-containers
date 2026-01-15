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
"""Documentation generation entry point.

This file is mainly used for development process to test the doc generation functionality.
If you intend to run the code with mkdocs, see src/hooks.py.

Usage:
    python docs/src/main.py [-h] [--dry-run] [--verbose] [--support-policy-only | --available-images-only | --clone-tutorials]
"""

import argparse
import logging
import os
import sys

from constants import TUTORIALS_REPO
from generate import generate_all, generate_available_images, generate_support_policy
from logger import ColoredFormatter
from utils import clone_git_repository, load_yaml

# Configure root logger - all child loggers inherit this
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColoredFormatter())
console_handler.setLevel(logging.DEBUG)

root_logger.addHandler(console_handler)

LOGGER = logging.getLogger(__name__)

# Resolve paths relative to this file
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.dirname(SRC_DIR)
DATA_FILE = os.path.join(SRC_DIR, "data", "images.yml")
REFERENCE_DIR = os.path.join(DOCS_DIR, "reference")
TUTORIALS_DIR = os.path.join(DOCS_DIR, "tutorials")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate DLC documentation from YAML source")
    parser.add_argument("--dry-run", action="store_true", help="Print output without writing files")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")

    exclusive_group = parser.add_mutually_exclusive_group()
    exclusive_group.add_argument(
        "--support-policy-only", action="store_true", help="Generate only support_policy.md"
    )
    exclusive_group.add_argument(
        "--available-images-only", action="store_true", help="Generate only available_images.md"
    )
    exclusive_group.add_argument(
        "--clone-tutorials", action="store_true", help="Clone only aws-samples tutorials repository"
    )
    args = parser.parse_args()

    if args.verbose:
        root_logger.setLevel(logging.DEBUG)

    yaml_data = load_yaml(DATA_FILE)
    LOGGER.info(f"Loaded data from {DATA_FILE}")

    actions = {
        "support_policy_only": lambda: generate_support_policy(yaml_data, args.dry_run),
        "available_images_only": lambda: generate_available_images(yaml_data, args.dry_run),
        "clone_tutorials": lambda: clone_git_repository(TUTORIALS_REPO, TUTORIALS_DIR),
    }

    for flag, action in actions.items():
        if getattr(args, flag):
            action()
            break
    else:
        clone_git_repository(TUTORIALS_REPO, TUTORIALS_DIR)
        generate_all(yaml_data, args.dry_run)

    LOGGER.info("Done")


if __name__ == "__main__":
    main()
