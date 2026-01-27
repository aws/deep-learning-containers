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
"""Tests for generate_all and cross-cutting generation concerns."""

import logging
from unittest.mock import patch

import pytest
from constants import GLOBAL_CONFIG
from generate import generate_all, generate_support_policy
from image_config import ImageConfig

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class TestGenerateAll:
    def test_calls_all_generators(self):
        """Test generate_all calls all three generators."""
        LOGGER.debug("Testing generate_all calls all generators")
        with patch("generate.generate_support_policy") as mock_support:
            with patch("generate.generate_available_images") as mock_images:
                with patch("generate.generate_release_notes") as mock_release:
                    generate_all(dry_run=True)
                    mock_support.assert_called_once_with(True)
                    mock_images.assert_called_once_with(True)
                    mock_release.assert_called_once_with(True)
        LOGGER.info("generate_all test passed")

    def test_dry_run_no_writes(self, mock_display_names, tmp_path):
        LOGGER.debug(f"Testing dry_run doesn't write to: {tmp_path}")
        with patch("generate.REFERENCE_DIR", tmp_path):
            with patch("generate.RELEASE_NOTES_DIR", tmp_path / "releasenotes"):
                generate_all(dry_run=True)
                assert list(tmp_path.iterdir()) == []
        LOGGER.info("generate_all dry_run test passed")


class TestDateConsistencyValidation:
    @pytest.fixture(scope="function")
    def mock_repo_images(self, request):
        """Fixture that patches load_repository_images based on parametrized dates."""
        repo_a_dates, repo_b_dates = request.param

        mock_images = {
            "repo-a": [
                ImageConfig(
                    "repo-a",
                    version="1.0",
                    ga=repo_a_dates[0],
                    eop=repo_a_dates[1],
                    accelerator="gpu",
                    platform="ec2",
                )
            ],
            "repo-b": [
                ImageConfig(
                    "repo-b",
                    version="1.0",
                    ga=repo_b_dates[0],
                    eop=repo_b_dates[1],
                    accelerator="gpu",
                    platform="ec2",
                )
            ],
        }

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

        with patch("generate.GLOBAL_CONFIG", patched_config):
            with patch("utils.GLOBAL_CONFIG", patched_config):
                with patch("image_config.GLOBAL_CONFIG", patched_config):
                    with patch(
                        "image_config.load_repository_images",
                        lambda repo: mock_images.get(repo, []),
                    ):
                        yield

    @pytest.mark.parametrize(
        "mock_repo_images",
        [
            (("2025-01-01", "2500-01-01"), ("2025-01-01", "2500-01-01")),
        ],
        indirect=True,
        ids=["consistent"],
    )
    def test_consistent_dates(self, mock_paths, mock_repo_images):
        """Test that consistent dates across repos in same framework group pass validation."""
        content = generate_support_policy(dry_run=True)
        assert "Test Group" in content

    @pytest.mark.parametrize(
        "mock_repo_images",
        [
            (("2025-01-01", "2500-01-01"), ("2025-01-01", "2500-06-01")),
            (("2025-01-01", "2500-01-01"), ("2025-02-01", "2500-01-01")),
            (("2025-01-01", "2500-01-01"), ("2025-06-01", "2500-06-01")),
        ],
        indirect=True,
        ids=["inconsistent_eop", "inconsistent_ga", "both_inconsistent"],
    )
    def test_inconsistent_dates_raises(self, mock_paths, mock_repo_images):
        """Test that inconsistent dates across repos in same framework group raise ValueError."""
        with pytest.raises(ValueError, match="Inconsistent dates"):
            generate_support_policy(dry_run=True)
