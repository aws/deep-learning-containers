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
"""Tests for generate_release_notes function.

We test the helper functions (_generate_individual_release_note, _generate_framework_index)
directly because generate_release_notes() returns None and writes files. Testing helpers
allows us to verify rendered content without filesystem side effects.
"""

import logging

import pytest
from constants import TEMPLATES_DIR
from generate import _generate_framework_index, _generate_individual_release_note
from image_config import ImageConfig
from jinja2 import Template
from utils import load_jinja2, load_table_config

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


@pytest.fixture(scope="module")
def release_template():
    """Load the release note template."""
    return Template(load_jinja2(TEMPLATES_DIR / "releasenotes" / "release_note.template.md"))


@pytest.fixture(scope="module")
def index_template():
    """Load the framework index template."""
    return Template(load_jinja2(TEMPLATES_DIR / "releasenotes" / "index.template.md"))


@pytest.fixture(scope="module")
def table_config(mock_paths):
    """Load release notes table config."""
    return load_table_config("extra/release_notes")


@pytest.fixture(scope="module")
def supported_image():
    """Create a supported image with release notes fields (EOP 2500-01-01)."""
    return ImageConfig(
        "mock-repo",
        framework="MockFramework",
        version="3.0",
        accelerator="gpu",
        platform="ec2",
        python="py312",
        eop="2500-01-01",
        tags=["3.0.0-gpu-py312-ec2"],
        announcement=["Introduced MockFramework 3.0", "Added GPU support"],
        packages={"python": "3.12", "mockframework": "3.0.0"},
    )


@pytest.fixture(scope="module")
def deprecated_image():
    """Create a deprecated image with release notes fields (EOP 2025-01-01)."""
    return ImageConfig(
        "mock-repo",
        framework="MockFramework",
        version="0.5",
        accelerator="cpu",
        platform="ec2",
        python="py39",
        eop="2025-01-01",
        tags=["0.5.0-cpu-py39-ec2"],
        announcement=["Initial release"],
        packages={"python": "3.9", "mockframework": "0.5.0"},
    )


def test_individual_release_note_content(
    mock_display_names, release_template, supported_image, tmp_path
):
    """Test that individual release notes contain all expected sections."""
    content = _generate_individual_release_note(
        supported_image, release_template, tmp_path, dry_run=True
    )

    # Title components
    assert "Mock Repository" in content
    assert "3.0" in content
    assert "EC2" in content

    # Announcement section
    assert "Introduced MockFramework 3.0" in content
    assert "Added GPU support" in content

    # Packages table
    assert "| Package | Version |" in content
    assert "3.12" in content
    assert "3.0.0" in content

    # Other sections
    assert "Security Advisory" in content
    assert "Docker Image URIs" in content


def test_framework_index_title(
    mock_display_names, index_template, table_config, supported_image, tmp_path
):
    """Test framework index contains framework display name."""
    content = _generate_framework_index(
        "mock-repo", [supported_image], index_template, table_config, tmp_path, dry_run=True
    )
    assert "Mock Repository" in content
    assert "Release Notes" in content


def test_framework_index_supported_section(
    mock_display_names, index_template, table_config, supported_image, tmp_path
):
    """Test supported images appear in main content (not in warning)."""
    content = _generate_framework_index(
        "mock-repo", [supported_image], index_template, table_config, tmp_path, dry_run=True
    )
    # Supported images should appear before any warning admonition
    warning_idx = content.find("!!! warning")
    version_idx = content.find("3.0")
    # Make sure version string is found, returns -1 if not found
    assert version_idx != -1
    # Make sure warning string is not found, or warning string is after version string
    assert warning_idx == -1 or version_idx < warning_idx


def test_framework_index_deprecated_section(
    mock_display_names, index_template, table_config, deprecated_image, tmp_path
):
    """Test deprecated images appear in warning admonition."""
    content = _generate_framework_index(
        "mock-repo", [deprecated_image], index_template, table_config, tmp_path, dry_run=True
    )
    assert '!!! warning "Deprecated Images"' in content
    # Version should appear after the warning
    warning_idx = content.find("!!! warning")
    version_idx = content.find("0.5", warning_idx)
    assert version_idx > warning_idx


def test_framework_index_mixed_support(
    mock_display_names,
    index_template,
    table_config,
    supported_image,
    deprecated_image,
    tmp_path,
):
    """Test index correctly separates supported and deprecated images."""
    content = _generate_framework_index(
        "mock-repo",
        [supported_image, deprecated_image],
        index_template,
        table_config,
        tmp_path,
        dry_run=True,
    )
    # Both versions should appear
    assert "3.0" in content
    assert "0.5" in content
    # Warning section should exist for deprecated
    assert '!!! warning "Deprecated Images"' in content


def test_framework_index_empty_images(mock_display_names, index_template, table_config, tmp_path):
    """Test framework index handles empty image list."""
    content = _generate_framework_index(
        "mock-repo", [], index_template, table_config, tmp_path, dry_run=True
    )
    # Should still render template without tables
    assert "Release Notes" in content
    assert "| ---" not in content
