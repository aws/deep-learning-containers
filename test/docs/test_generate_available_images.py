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
"""Tests for generate_available_images function."""

import logging
from unittest.mock import patch

import pytest
from constants import GLOBAL_CONFIG
from generate import generate_available_images

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


@pytest.mark.parametrize(
    "expected",
    [
        "# Available Images",
        "## Region Availability",
        "us-east-1",
        "Mock Repository",
        "| Framework |",
        "| Python |",
    ],
    ids=[
        "title",
        "region_header",
        "region_code",
        "repo_section",
        "framework_col",
        "python_col",
    ],
)
def test_contains_sections(mock_display_names, expected):
    LOGGER.debug(f"Testing available_images contains: {expected}")
    content = generate_available_images(dry_run=True)
    assert expected in content


def test_public_registry_note(mock_display_names):
    LOGGER.debug("Testing public registry note")
    content = generate_available_images(dry_run=True)
    assert "ECR Public Gallery" in content or "gallery.ecr.aws" in content
    LOGGER.info("available_images public registry test passed")


def test_filters_unsupported(mock_display_names):
    """Version 1.0 has EOP 2025-01-01 (past), should not appear in tables."""
    LOGGER.debug("Testing unsupported images are filtered")
    content = generate_available_images(dry_run=True)
    tables_section = content.split("## Region Availability")[-1]
    assert "1.0.0-cpu-py311" not in tables_section
    LOGGER.info("available_images filters unsupported test passed")


def test_empty_repo_skipped(mock_paths):
    LOGGER.debug("Testing empty repository is skipped")
    patched_config = {
        **GLOBAL_CONFIG,
        "table_order": ["nonexistent-repo"],
        "display_names": {**GLOBAL_CONFIG.get("display_names", {}), "nonexistent-repo": "None"},
    }
    with patch("generate.GLOBAL_CONFIG", patched_config):
        with patch("utils.GLOBAL_CONFIG", patched_config):
            content = generate_available_images(dry_run=True)
            assert "None" not in content.split("## Region Availability")[-1]
