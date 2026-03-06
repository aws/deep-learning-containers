"""Documentation generation entry point via mkdocs hooks.

MkDocs hook:
    Add to mkdocs.yaml: hooks: [docs/src/hooks.py]
"""

import logging
import sys

from constants import TUTORIALS_DIR, TUTORIALS_REPO
from generate import generate_all
from logger import ColoredFormatter
from utils import clone_git_repository

# Configure root logger - all child loggers inherit this
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColoredFormatter())
console_handler.setLevel(logging.DEBUG)

root_logger.addHandler(console_handler)

LOGGER = logging.getLogger(__name__)


# MkDocs hook entry point
def on_startup(command=["build", "gh-deploy", "serve"], dirty=False):
    """MkDocs hook - runs before build."""
    clone_git_repository(TUTORIALS_REPO, TUTORIALS_DIR)
    generate_all(dry_run=False)
