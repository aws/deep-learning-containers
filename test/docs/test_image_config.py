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
"""Tests for ImageConfig class and helper functions."""

import logging
from unittest.mock import patch

import pytest
from image_config import (
    ImageConfig,
    build_image_row,
    check_public_registry,
    get_latest_image_uri,
    load_legacy_images,
    load_repository_images,
    sort_by_version,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class TestImageConfigInit:
    def test_basic(self):
        LOGGER.debug("Testing ImageConfig basic initialization")
        img = ImageConfig("test-repo", version="1.0", framework="TestFramework")
        assert img.repository == "test-repo"
        assert img.version == "1.0"
        assert img.framework == "TestFramework"
        LOGGER.info("ImageConfig basic initialization test passed")

    def test_from_yaml(self, mock_data_dir):
        path = mock_data_dir / "data" / "mock-repo" / "2.0-gpu-ec2.yml"
        LOGGER.debug(f"Loading ImageConfig from: {path}")
        img = ImageConfig.from_yaml(path, "mock-repo")
        assert img.repository == "mock-repo"
        assert img.version == "2.0"
        assert img.accelerator == "gpu"
        assert img.cuda == "cu121"
        LOGGER.info("ImageConfig from_yaml test passed")


class TestImageConfigAttrAccess:
    @pytest.mark.parametrize("attr,expected", [("version", "1.0"), ("python", "py312")])
    def test_getattr_valid(self, attr, expected):
        img = ImageConfig("repo", version="1.0", python="py312")
        assert getattr(img, attr) == expected

    def test_getattr_invalid(self):
        img = ImageConfig("repo", version="1.0")
        with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
            _ = img.nonexistent

    def test_getattr_private(self):
        img = ImageConfig("repo", version="1.0")
        with pytest.raises(AttributeError):
            _ = img._private

    @pytest.mark.parametrize(
        "field,default,expected",
        [("cuda", None, "cu121"), ("missing", None, None), ("missing", "-", "-")],
    )
    def test_get(self, field, default, expected):
        img = ImageConfig("repo", cuda="cu121")
        assert img.get(field, default) == expected


class TestImageConfigProperties:
    @pytest.mark.parametrize(
        "repo,expected_group",
        [("pytorch-training", "pytorch"), ("standalone-repo", "standalone-repo")],
    )
    def test_framework_group(self, repo, expected_group):
        img = ImageConfig(repo, version="1.0")
        LOGGER.debug(f"Testing framework_group for repo={repo}")
        assert img.framework_group == expected_group

    @pytest.mark.parametrize(
        "eop,expected",
        [("2099-12-31", True), ("2020-01-01", False), (None, True)],
        ids=["future_eop", "past_eop", "no_eop"],
    )
    def test_is_supported(self, eop, expected):
        kwargs = {"eop": eop} if eop else {}
        img = ImageConfig("repo", **kwargs)
        LOGGER.debug(f"Testing is_supported with eop={eop}")
        assert img.is_supported is expected

    @pytest.mark.parametrize(
        "ga,eop,expected",
        [
            ("2025-01-01", "2026-01-01", True),
            (None, "2026-01-01", False),
            ("2025-01-01", None, False),
            (None, None, False),
        ],
        ids=["both_present", "missing_ga", "missing_eop", "both_missing"],
    )
    def test_has_support_dates(self, ga, eop, expected):
        kwargs = {}
        if ga:
            kwargs["ga"] = ga
        if eop:
            kwargs["eop"] = eop
        img = ImageConfig("repo", **kwargs)
        assert img.has_support_dates is expected


class TestImageConfigDisplayProperties:
    def test_display_repository(self):
        img = ImageConfig("pytorch-training", version="2.0")
        assert img.display_repository == "PyTorch Training"

    def test_display_repository_missing(self):
        img = ImageConfig("unknown-repo", version="1.0")
        with pytest.raises(KeyError, match="Display name not found"):
            _ = img.display_repository

    def test_display_framework_group(self):
        img = ImageConfig("pytorch-training", version="2.0")
        assert img.display_framework_group == "PyTorch"

    def test_display_framework_version(self):
        img = ImageConfig("repo", framework="PyTorch", version="2.0")
        assert img.display_framework_version == "PyTorch 2.0"

    @pytest.mark.parametrize(
        "kwargs,expected_substr",
        [
            ({"tag": "2.0-gpu-py312"}, "763104351884"),
            ({"tag": "1.0-cpu", "example_ecr_account": "123456789012"}, "123456789012"),
        ],
        ids=["default_account", "custom_account"],
    )
    def test_display_example_url(self, kwargs, expected_substr):
        img = ImageConfig("pytorch-training", **kwargs)
        assert expected_substr in img.display_example_url

    @pytest.mark.parametrize(
        "platform,expected", [("ec2", "EC2, ECS, EKS"), ("sagemaker", "SageMaker")]
    )
    def test_display_platform(self, platform, expected):
        img = ImageConfig("repo", platform=platform)
        assert img.display_platform == expected

    @pytest.mark.parametrize(
        "accelerator,expected", [("gpu", "GPU"), ("cpu", "CPU"), ("neuronx", "NeuronX")]
    )
    def test_display_accelerator(self, accelerator, expected):
        img = ImageConfig("repo", accelerator=accelerator)
        assert img.display_accelerator == expected


class TestImageConfigGetDisplay:
    @pytest.mark.parametrize(
        "kwargs,field,expected",
        [
            ({"platform": "ec2"}, "platform", "EC2, ECS, EKS"),
            ({"python": "py312"}, "python", "py312"),
            ({}, "cuda", "-"),
        ],
        ids=["with_display_property", "without_display_property", "missing_field"],
    )
    def test_get_display(self, kwargs, field, expected):
        img = ImageConfig("repo", **kwargs)
        assert img.get_display(field) == expected


class TestBuildImageRow:
    @pytest.mark.parametrize(
        "columns,expected",
        [
            (
                [
                    {"field": "framework_version", "header": "Framework"},
                    {"field": "python", "header": "Python"},
                ],
                ["PyTorch 2.0", "py312"],
            ),
        ],
    )
    def test_build_row(self, columns, expected):
        img = ImageConfig("repo", framework="PyTorch", version="2.0", python="py312")
        assert build_image_row(img, columns) == expected


class TestLoadRepositoryImages:
    def test_existing_repo(self, mock_paths):
        LOGGER.debug("Testing load_repository_images for existing repo")
        images = load_repository_images("mock-repo")
        LOGGER.debug(f"Loaded {len(images)} images")
        assert len(images) == 4
        assert all(isinstance(img, ImageConfig) for img in images)
        LOGGER.info("load_repository_images test passed")

    def test_nonexistent_repo(self, mock_paths):
        LOGGER.debug("Testing load_repository_images for nonexistent repo")
        assert load_repository_images("nonexistent-repo") == []


class TestLoadLegacyImages:
    def test_with_file(self, mock_paths):
        LOGGER.debug("Testing load_legacy_images with existing file")
        legacy = load_legacy_images()
        assert "mock-framework" in legacy
        assert len(legacy["mock-framework"]) == 2
        assert legacy["mock-framework"][0].version == "0.5"
        LOGGER.info("load_legacy_images test passed")

    def test_no_file(self, mock_paths, mock_data_dir):
        with patch("image_config.LEGACY_DIR", mock_data_dir / "nonexistent"):
            assert load_legacy_images() == {}


class TestSortByVersion:
    def test_descending(self):
        images = [
            ImageConfig("repo", version="1.0"),
            ImageConfig("repo", version="2.0"),
            ImageConfig("repo", version="1.5"),
        ]
        versions = [img.version for img in sort_by_version(images)]
        LOGGER.debug(f"Sorted versions: {versions}")
        assert versions == ["2.0", "1.5", "1.0"]

    def test_with_tiebreakers(self):
        images = [
            ImageConfig("repo", version="2.0", platform="ec2"),
            ImageConfig("repo", version="2.0", platform="sagemaker"),
            ImageConfig("repo", version="1.0", platform="ec2"),
        ]
        sorted_imgs = sort_by_version(
            images, tiebreakers=[lambda img: 0 if img.get("platform") == "sagemaker" else 1]
        )
        assert sorted_imgs[0].platform == "sagemaker"
        assert sorted_imgs[1].platform == "ec2"
        assert sorted_imgs[1].version == "2.0"

    def test_invalid_version(self):
        images = [ImageConfig("repo", version="invalid"), ImageConfig("repo", version="1.0")]
        assert sort_by_version(images)[0].version == "1.0"


class TestCheckPublicRegistry:
    @pytest.mark.parametrize(
        "registry_values,expected",
        [([True, True], True), ([None, None], False)],
        ids=["all_public", "none_public"],
    )
    def test_check(self, registry_values, expected):
        images = [ImageConfig("repo", public_registry=v) for v in registry_values]
        assert check_public_registry(images, "repo") is expected

    def test_mixed(self, caplog):
        LOGGER.debug("Testing check_public_registry with mixed values")
        images = [
            ImageConfig("repo", public_registry=True),
            ImageConfig("repo", public_registry=False),
        ]
        assert check_public_registry(images, "test-repo") is True
        assert "mixed public_registry" in caplog.text


class TestGetLatestImage:
    def test_success(self, mock_paths):
        LOGGER.debug("Testing get_latest_image_uri for valid repo/platform")
        uri = get_latest_image_uri("mock-repo", "ec2")
        LOGGER.debug(f"Latest image URI: {uri}")
        assert "mock-repo" in uri
        assert "2.0.0-gpu-py312" in uri
        LOGGER.info("get_latest_image_uri test passed")

    @pytest.mark.parametrize(
        "repo,platform",
        [("mock-repo", "nonexistent-platform"), ("nonexistent-repo", "ec2")],
        ids=["invalid_platform", "invalid_repo"],
    )
    def test_not_found(self, mock_paths, repo, platform):
        with pytest.raises(ValueError, match="Image not found"):
            get_latest_image_uri(repo, platform)
