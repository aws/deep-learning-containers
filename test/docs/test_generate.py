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
"""Tests for documentation generation functions."""

import logging
from unittest.mock import patch

import pytest
from constants import GLOBAL_CONFIG
from generate import generate_all, generate_available_images, generate_support_policy
from image_config import ImageConfig

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class TestGenerateSupportPolicy:
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
    def test_contains_sections(self, mock_display_names, expected):
        LOGGER.debug(f"Testing support_policy contains: {expected}")
        content = generate_support_policy(dry_run=True)
        assert expected in content

    def test_contains_table_structure(self, mock_display_names):
        LOGGER.debug("Testing support_policy table structure")
        content = generate_support_policy(dry_run=True)
        assert "| Framework |" in content or "| ---" in content
        LOGGER.info("support_policy table structure test passed")

    def test_supported_images(self, mock_display_names):
        """Version 2.0 has EOP 2027-01-01 (future), should be in supported section."""
        LOGGER.debug("Testing supported images appear in supported section")
        content = generate_support_policy(dry_run=True)
        supported_section = content.split("## Unsupported Frameworks")[0]
        assert "2.0" in supported_section or "Mock" in supported_section
        LOGGER.info("support_policy supported images test passed")


class TestGenerateAvailableImages:
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
    def test_contains_sections(self, mock_display_names, expected):
        LOGGER.debug(f"Testing available_images contains: {expected}")
        content = generate_available_images(dry_run=True)
        assert expected in content

    def test_public_registry_note(self, mock_display_names):
        LOGGER.debug("Testing public registry note")
        content = generate_available_images(dry_run=True)
        assert "ECR Public Gallery" in content or "gallery.ecr.aws" in content
        LOGGER.info("available_images public registry test passed")

    def test_filters_unsupported(self, mock_display_names):
        """Version 1.0 has EOP 2025-01-01 (past), should not appear in tables."""
        LOGGER.debug("Testing unsupported images are filtered")
        content = generate_available_images(dry_run=True)
        tables_section = content.split("## Region Availability")[-1]
        assert "1.0.0-cpu-py311" not in tables_section
        LOGGER.info("available_images filters unsupported test passed")

    def test_empty_repo_skipped(self, mock_paths):
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


class TestDateConsistencyValidation:
    @pytest.mark.parametrize(
        "repo_a_dates,repo_b_dates,should_raise",
        [
            (("2025-01-01", "2026-01-01"), ("2025-01-01", "2026-01-01"), False),
            (("2025-01-01", "2026-01-01"), ("2025-01-01", "2026-06-01"), True),
            (("2025-01-01", "2026-01-01"), ("2025-02-01", "2026-01-01"), True),
            (("2025-01-01", "2026-01-01"), ("2025-06-01", "2027-01-01"), True),
        ],
        ids=["consistent", "inconsistent_eop", "inconsistent_ga", "both_inconsistent"],
    )
    def test_validation(self, mock_paths, repo_a_dates, repo_b_dates, should_raise):
        LOGGER.debug(f"Testing date consistency: repo_a={repo_a_dates}, repo_b={repo_b_dates}")
        patched_config = {
            **GLOBAL_CONFIG,
            "table_order": ["repo-a", "repo-b"],
            "display_names": {
                **GLOBAL_CONFIG.get("display_names", {}),
                "test-group": "Test Group",
                "repo-a": "Repo A",
                "repo-b": "Repo B",
            },
            "framework_groups": {"test-group": ["repo-a", "repo-b"]},
        }

        def mock_load(repo):
            if repo == "repo-a":
                return [ImageConfig(repo, version="1.0", ga=repo_a_dates[0], eop=repo_a_dates[1])]
            elif repo == "repo-b":
                return [ImageConfig(repo, version="1.0", ga=repo_b_dates[0], eop=repo_b_dates[1])]
            return []

        with patch("generate.GLOBAL_CONFIG", patched_config):
            with patch("utils.GLOBAL_CONFIG", patched_config):
                with patch("image_config.GLOBAL_CONFIG", patched_config):
                    with patch("generate.load_repository_images", mock_load):
                        if should_raise:
                            LOGGER.debug("Expecting ValueError for inconsistent dates")
                            with pytest.raises(ValueError, match="Inconsistent dates"):
                                generate_support_policy(dry_run=True)
                        else:
                            content = generate_support_policy(dry_run=True)
                            assert "Test Group" in content
                            LOGGER.info("Date consistency validation passed")


class TestGenerateAll:
    def test_calls_both(self, mock_display_names):
        LOGGER.debug("Testing generate_all calls both generators")
        with patch("generate.generate_support_policy") as mock_support:
            with patch("generate.generate_available_images") as mock_images:
                generate_all(dry_run=True)
                mock_support.assert_called_once_with(True)
                mock_images.assert_called_once_with(True)
        LOGGER.info("generate_all test passed")

    def test_dry_run_no_writes(self, mock_display_names, tmp_path):
        LOGGER.debug(f"Testing dry_run doesn't write to: {tmp_path}")
        with patch("generate.REFERENCE_DIR", tmp_path):
            generate_all(dry_run=True)
            assert list(tmp_path.iterdir()) == []
        LOGGER.info("generate_all dry_run test passed")
