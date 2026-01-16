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
    python docs/src/main.py [-h] [--dry-run] [--verbose]
"""

import argparse
import logging
import os
import sys

from constants import TUTORIALS_REPO
from generate import generate_all
from utils import clone_git_repository

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
LOGGER = logging.getLogger(__name__)

# Resolve paths relative to this file
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.dirname(SRC_DIR)
TUTORIALS_DIR = os.path.join(DOCS_DIR, "tutorials")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate DLC documentation from config files")
    parser.add_argument("--dry-run", action="store_true", help="Print output without writing files")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    clone_git_repository(TUTORIALS_REPO, TUTORIALS_DIR)
    generate_all(args.dry_run)
    LOGGER.info("Done")


if __name__ == "__main__":
    main()
