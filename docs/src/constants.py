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
"""Global variables for documentation generation."""

from pathlib import Path

from omegaconf import OmegaConf

# Path constants
SRC_DIR = Path(__file__).parent
DOCS_DIR = SRC_DIR.parent
DATA_DIR = SRC_DIR / "data"
LEGACY_DIR = SRC_DIR / "legacy"
TABLES_DIR = SRC_DIR / "tables"
TEMPLATES_DIR = SRC_DIR / "templates"
REFERENCE_DIR = DOCS_DIR / "reference"
RELEASE_NOTES_DIR = DOCS_DIR / "releasenotes"
TUTORIALS_DIR = DOCS_DIR / "tutorials"

# Release notes configuration
RELEASE_NOTES_REQUIRED_FIELDS = ["announcement", "packages"]
GLOBAL_CONFIG_PATH = SRC_DIR / "global.yml"

AVAILABLE_IMAGES_TABLE_HEADER = "##"
RELEASE_NOTES_TABLE_HEADER = "###"
TUTORIALS_REPO = "https://github.com/aws-samples/sample-aws-deep-learning-containers"

# Load global config once at import time
global_cfg = OmegaConf.load(GLOBAL_CONFIG_PATH)
GLOBAL_CONFIG = OmegaConf.to_container(global_cfg, resolve=True)
