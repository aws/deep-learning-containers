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

Usage:
    python docs/src/main.py [--dry-run] [--verbose] [--support-policy-only] [--available-images-only]
"""

import argparse
import logging
import os

from generate import generate_all, generate_available_images, generate_support_policy
from utils import load_yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

# Resolve paths relative to this file
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.dirname(SRC_DIR)
DATA_FILE = os.path.join(SRC_DIR, "data", "images.yml")
REFERENCE_DIR = os.path.join(DOCS_DIR, "reference")


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
        logging.getLogger().setLevel(logging.DEBUG)

    LOGGER.info(f"Loaded data from {DATA_FILE}")

    if args.support_policy_only:
        generate_support_policy(yaml_data, args.dry_run)
    elif args.available_images_only:
        generate_available_images(yaml_data, args.dry_run)
    else:
        generate_all(yaml_data, args.dry_run)

    LOGGER.info("Done")


if __name__ == "__main__":
    main()
