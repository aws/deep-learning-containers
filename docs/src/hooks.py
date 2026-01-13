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

import logging
import os
import subprocess

from generate import generate_all
from utils import load_yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

# Resolve paths relative to this file
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.dirname(SRC_DIR)
DATA_FILE = os.path.join(SRC_DIR, "data", "images.yml")


def clone_tutorials():
    """Clone sample tutorials repository into docs/tutorials."""
    tutorials_dir = os.path.join(DOCS_DIR, "tutorials")
    if os.path.exists(tutorials_dir):
        return

    repo_url = "https://github.com/aws-samples/sample-aws-deep-learning-containers"
    subprocess.run(["git", "clone", "--depth", "1", repo_url, tutorials_dir], check=True)


# MkDocs hook entry point
def on_startup(command=["build", "gh-deploy", "serve"], dirty=False):
    """MkDocs hook - runs before build."""
    yaml_data = load_yaml(DATA_FILE)
    clone_tutorials()
    generate_all(yaml_data, dry_run=False)
