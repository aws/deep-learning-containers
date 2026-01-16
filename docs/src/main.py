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
