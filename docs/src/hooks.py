"""Documentation generation entry point via mkdocs hooks.

MkDocs hook:
    Add to mkdocs.yaml: hooks: [docs/src/hooks.py]
"""

import os

from constants import TUTORIALS_REPO
from generate import generate_all
from utils import clone_git_repository

# Resolve paths relative to this file
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.dirname(SRC_DIR)
TUTORIALS_DIR = os.path.join(DOCS_DIR, "tutorials")


def on_startup(command=["build", "gh-deploy", "serve"], dirty=False):
    """MkDocs hook - runs before build."""
    clone_git_repository(TUTORIALS_REPO, TUTORIALS_DIR)
    generate_all(dry_run=False)
