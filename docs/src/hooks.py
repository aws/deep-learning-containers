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
"""Documentation generation entry point via mkdocs hooks.

MkDocs hook:
    Add to mkdocs.yaml: hooks: [docs/src/hooks.py]
"""

from constants import TUTORIALS_DIR, TUTORIALS_REPO
from generate import generate_all
from utils import clone_git_repository


# MkDocs hook entry point
def on_startup(command=["build", "gh-deploy", "serve"], dirty=False):
    """MkDocs hook - runs before build."""
    clone_git_repository(TUTORIALS_REPO, TUTORIALS_DIR)
    generate_all(dry_run=False)
