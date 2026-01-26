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
"""Pytest fixtures for documentation generation tests."""

import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add docs/src to path for imports
DOCS_SRC_DIR = Path(__file__).parent.parent.parent / "docs" / "src"
MOCK_DATA_DIR = Path(__file__).parent / "mock_data"
sys.path.insert(0, str(DOCS_SRC_DIR))

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


@pytest.fixture(scope="module")
def mock_data_dir():
    """Return path to mock data directory."""
    LOGGER.debug(f"Mock data directory: {MOCK_DATA_DIR}")
    return MOCK_DATA_DIR


@pytest.fixture(scope="function")
def mock_paths():
    """Patch DATA_DIR, LEGACY_DIR, TABLES_DIR to use mock data."""
    import constants
    import image_config
    import utils

    LOGGER.debug("Patching paths to use mock data directory")

    original = {
        "constants.DATA_DIR": constants.DATA_DIR,
        "constants.LEGACY_DIR": constants.LEGACY_DIR,
        "constants.TABLES_DIR": constants.TABLES_DIR,
        "image_config.DATA_DIR": image_config.DATA_DIR,
        "image_config.LEGACY_DIR": image_config.LEGACY_DIR,
        "utils.TABLES_DIR": utils.TABLES_DIR,
    }

    constants.DATA_DIR = MOCK_DATA_DIR / "data"
    constants.LEGACY_DIR = MOCK_DATA_DIR / "legacy"
    constants.TABLES_DIR = MOCK_DATA_DIR / "tables"
    image_config.DATA_DIR = MOCK_DATA_DIR / "data"
    image_config.LEGACY_DIR = MOCK_DATA_DIR / "legacy"
    utils.TABLES_DIR = MOCK_DATA_DIR / "tables"

    LOGGER.debug(f"DATA_DIR set to: {constants.DATA_DIR}")

    yield

    LOGGER.debug("Restoring original paths")
    constants.DATA_DIR = original["constants.DATA_DIR"]
    constants.LEGACY_DIR = original["constants.LEGACY_DIR"]
    constants.TABLES_DIR = original["constants.TABLES_DIR"]
    image_config.DATA_DIR = original["image_config.DATA_DIR"]
    image_config.LEGACY_DIR = original["image_config.LEGACY_DIR"]
    utils.TABLES_DIR = original["utils.TABLES_DIR"]


@pytest.fixture(scope="function")
def mock_display_names(mock_paths):
    """Patch display_names to include mock-repo and mock-framework."""
    from constants import GLOBAL_CONFIG

    LOGGER.debug("Patching GLOBAL_CONFIG with mock display names")

    patched_config = {
        **GLOBAL_CONFIG,
        "display_names": {
            **GLOBAL_CONFIG.get("display_names", {}),
            "mock-repo": "Mock Repository",
            "mock-framework": "Mock Framework",
        },
        "table_order": ["mock-repo"],
        "framework_groups": {},
    }
    with patch("generate.GLOBAL_CONFIG", patched_config):
        with patch("utils.GLOBAL_CONFIG", patched_config):
            with patch("image_config.GLOBAL_CONFIG", patched_config):
                yield patched_config
