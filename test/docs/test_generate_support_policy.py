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
"""Tests for generate_support_policy function."""

import logging

import pytest
from generate import generate_support_policy

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


@pytest.mark.parametrize(
    "expected",
    [
        "# Framework Support Policy",
        "## Supported Frameworks",
        "## Unsupported Frameworks",
        "## Glossary",
        "GA (General Availability)",
        "EOP (End of Patch)",
    ],
    ids=["title", "supported_header", "unsupported_header", "glossary", "ga_term", "eop_term"],
)
def test_contains_sections(mock_display_names, expected):
    LOGGER.debug(f"Testing support_policy contains: {expected}")
    content = generate_support_policy(dry_run=True)
    assert expected in content


def test_contains_table_structure(mock_display_names):
    LOGGER.debug("Testing support_policy table structure")
    content = generate_support_policy(dry_run=True)
    assert "| Framework |" in content or "| ---" in content
    LOGGER.info("support_policy table structure test passed")


def test_supported_images(mock_display_names):
    """Version 2.0 has EOP 2500-01-01 (far future), should be in supported section."""
    LOGGER.debug("Testing supported images appear in supported section")
    content = generate_support_policy(dry_run=True)
    supported_section = content.split("## Unsupported Frameworks")[0]
    assert "2.0" in supported_section or "Mock" in supported_section
    LOGGER.info("support_policy supported images test passed")
