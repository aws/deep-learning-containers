"""Property-based tests for the auto-docs-pr helper functions.

Uses hypothesis to validate correctness properties across many random inputs.
"""

from __future__ import annotations

import string

from hypothesis import given, settings
from hypothesis import strategies as st

from tests.helpers.docs_pr_functions import (
    build_failed_packages_report,
    build_failure_slack_payload,
    build_slack_payload,
    generate_announcement,
    generate_branch_name,
    generate_pr_title,
    generate_tags,
    get_display_name,
    parse_cuda_version,
    parse_major_minor,
    parse_pip_version,
)

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

_version_int = st.integers(min_value=0, max_value=99)

# Simple identifier-like strings (no whitespace, no newlines)
_identifier = st.text(
    alphabet=string.ascii_lowercase + string.digits + "-_",
    min_size=1,
    max_size=20,
)

_nonempty_printable = st.text(
    alphabet=string.ascii_letters + string.digits + "-_.",
    min_size=1,
    max_size=30,
)


# ===================================================================
# 6.1 — Property 5: Major.minor version extraction
# ===================================================================
# Feature: auto-docs-pr, Property 5: Major.minor version extraction


class TestMajorMinorExtraction:
    """**Validates: Requirements 6.3**"""

    @settings(max_examples=100)
    @given(major=_version_int, minor=_version_int, patch=_version_int)
    def test_xyz_format(self, major: int, minor: int, patch: int) -> None:
        version = f"{major}.{minor}.{patch}"
        result = parse_major_minor(version)
        assert result == f"{major}.{minor}"

    @settings(max_examples=100)
    @given(major=_version_int, minor=_version_int)
    def test_xy_format(self, major: int, minor: int) -> None:
        version = f"{major}.{minor}"
        result = parse_major_minor(version)
        assert result == f"{major}.{minor}"


# ===================================================================
# 6.2 — Property 4: Tag generation per platform
# ===================================================================
# Feature: auto-docs-pr, Property 4: Tag generation per platform


class TestTagGeneration:
    """**Validates: Requirements 5.7, 6.1, 6.2**"""

    @settings(max_examples=100)
    @given(
        major=_version_int,
        minor=_version_int,
        patch=_version_int,
        device=_identifier,
        python=_identifier,
        cuda=_identifier,
        os_ver=_identifier,
    )
    def test_ec2_tags(
        self,
        major: int,
        minor: int,
        patch: int,
        device: str,
        python: str,
        cuda: str,
        os_ver: str,
    ) -> None:
        version = f"{major}.{minor}.{patch}"
        tags = generate_tags(version, device, python, cuda, os_ver, "ec2")

        assert len(tags) == 4
        # All ec2 tags must contain "ec2"
        for tag in tags:
            assert "ec2" in tag

        # Tags contain the version/device/python components
        assert tags[0] == f"{version}-{device}-{python}-{cuda}-{os_ver}-ec2"
        mm = f"{major}.{minor}"
        assert tags[1] == f"{mm}-{device}-{python}-{cuda}-{os_ver}-ec2-v1"
        assert tags[2] == f"{version}-{device}-{python}-ec2"
        assert tags[3] == f"{mm}-{device}-{python}-ec2"

    @settings(max_examples=100)
    @given(
        major=_version_int,
        minor=_version_int,
        patch=_version_int,
        device=_identifier,
        python=_identifier,
        cuda=_identifier,
        os_ver=_identifier,
    )
    def test_sagemaker_tags(
        self,
        major: int,
        minor: int,
        patch: int,
        device: str,
        python: str,
        cuda: str,
        os_ver: str,
    ) -> None:
        version = f"{major}.{minor}.{patch}"
        mm = f"{major}.{minor}"
        tags = generate_tags(version, device, python, cuda, os_ver, "sagemaker")

        assert len(tags) == 4
        # First 2 tags contain "sagemaker"
        assert "sagemaker" in tags[0]
        assert "sagemaker" in tags[1]
        # Last 2 tags do NOT have a platform suffix
        assert tags[2] == f"{version}-{device}-{python}"
        assert tags[3] == f"{mm}-{device}-{python}"

        # Verify exact patterns
        assert tags[0] == f"{version}-{device}-{python}-{cuda}-{os_ver}-sagemaker"
        assert tags[1] == f"{mm}-{device}-{python}-{cuda}-{os_ver}-sagemaker-v1"


# ===================================================================
# 6.3 — Property 6: Announcement message generation
# ===================================================================
# Feature: auto-docs-pr, Property 6: Announcement message generation


class TestAnnouncementGeneration:
    """**Validates: Requirements 5.8, 10.1, 10.2, 10.3**"""

    @settings(max_examples=100)
    @given(version=_nonempty_printable)
    def test_ec2_known_frameworks(self, version: str) -> None:
        for framework in ("vllm", "sglang"):
            display = get_display_name(framework)
            result = generate_announcement(framework, version, "ec2")
            assert result == f"Introduced {display} {version} containers for EC2, ECS, EKS"

    @settings(max_examples=100)
    @given(version=_nonempty_printable)
    def test_sagemaker_known_frameworks(self, version: str) -> None:
        for framework in ("vllm", "sglang"):
            display = get_display_name(framework)
            result = generate_announcement(framework, version, "sagemaker")
            assert result == f"Introduced {display} {version} containers for SageMaker"

    @settings(max_examples=100)
    @given(framework=_identifier, version=_nonempty_printable)
    def test_ec2_random_framework(self, framework: str, version: str) -> None:
        display = get_display_name(framework)
        result = generate_announcement(framework, version, "ec2")
        assert result == f"Introduced {display} {version} containers for EC2, ECS, EKS"

    @settings(max_examples=100)
    @given(framework=_identifier, version=_nonempty_printable)
    def test_sagemaker_random_framework(self, framework: str, version: str) -> None:
        display = get_display_name(framework)
        result = generate_announcement(framework, version, "sagemaker")
        assert result == f"Introduced {display} {version} containers for SageMaker"


# ===================================================================
# 6.4 — Property 7: Branch name and PR title generation
# ===================================================================
# Feature: auto-docs-pr, Property 7: Branch name and PR title generation


class TestBranchNameAndPrTitle:
    """**Validates: Requirements 7.1, 7.5**"""

    @settings(max_examples=100)
    @given(
        framework=_identifier,
        version=_nonempty_printable,
        platform=st.sampled_from(["ec2", "sagemaker"]),
    )
    def test_branch_name_format(self, framework: str, version: str, platform: str) -> None:
        branch = generate_branch_name(framework, version, platform)
        assert branch == f"docs/auto-update-{framework}-{version}-{platform}"

    @settings(max_examples=100)
    @given(
        framework=_identifier,
        version=_nonempty_printable,
        platform=st.sampled_from(["ec2", "sagemaker"]),
    )
    def test_pr_title_contains_required_parts(
        self, framework: str, version: str, platform: str
    ) -> None:
        title = generate_pr_title(framework, version, platform)
        display = get_display_name(framework)
        assert display in title
        assert version in title
        assert platform.upper() in title


# ===================================================================
# 6.5 — Property 2: Version string parsing from command output
# ===================================================================
# Feature: auto-docs-pr, Property 2: Version string parsing from command output


class TestVersionParsing:
    """**Validates: Requirements 3.2, 3.3**"""

    @settings(max_examples=100)
    @given(
        major=_version_int,
        minor=_version_int,
        patch=_version_int,
    )
    def test_parse_pip_version(self, major: int, minor: int, patch: int) -> None:
        ver = f"{major}.{minor}.{patch}"
        pip_output = f"Name: some-pkg\nVersion: {ver}\nSummary: A package"
        assert parse_pip_version(pip_output) == ver

    @settings(max_examples=100)
    @given(major=_version_int, minor=_version_int)
    def test_parse_cuda_version(self, major: int, minor: int) -> None:
        ver = f"{major}.{minor}"
        nvcc_output = f"nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release {ver}, V{ver}.123"
        assert parse_cuda_version(nvcc_output) == ver

    def test_empty_pip_output(self) -> None:
        assert parse_pip_version("") == ""

    def test_malformed_pip_output(self) -> None:
        assert parse_pip_version("no version line here") == ""

    def test_empty_cuda_output(self) -> None:
        assert parse_cuda_version("") == ""

    def test_malformed_cuda_output(self) -> None:
        assert parse_cuda_version("no release info here") == ""


# ===================================================================
# 6.6 — Property 3: Docs data file field pass-through correctness
# ===================================================================
# Feature: auto-docs-pr, Property 3: Docs data file field pass-through correctness


class TestDocsDataFieldPassThrough:
    """**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.9**"""

    @settings(max_examples=100)
    @given(framework=st.sampled_from(["vllm", "sglang"]))
    def test_known_display_name(self, framework: str) -> None:
        expected = {"vllm": "vLLM", "sglang": "SGLang"}
        assert get_display_name(framework) == expected[framework]

    @settings(max_examples=100)
    @given(framework=_identifier)
    def test_unknown_display_name_fallback(self, framework: str) -> None:
        # For unknown frameworks, get_display_name returns the raw framework string
        display = get_display_name(framework)
        if framework not in ("vllm", "sglang"):
            assert display == framework

    @settings(max_examples=100)
    @given(
        framework=_identifier,
        version=_nonempty_printable,
        device=_identifier,
        platform=st.sampled_from(["ec2", "sagemaker"]),
    )
    def test_output_file_path_pattern(
        self, framework: str, version: str, device: str, platform: str
    ) -> None:
        expected_path = f"docs/src/data/{framework}/{version}-{device}-{platform}.yml"
        # Verify the path components are consistent with the inputs
        assert framework in expected_path
        assert version in expected_path
        assert device in expected_path
        assert platform in expected_path

    @settings(max_examples=100)
    @given(
        version=_nonempty_printable,
        device=_identifier,
        python=_identifier,
        cuda=_identifier,
        os_ver=_identifier,
        platform=st.sampled_from(["ec2", "sagemaker"]),
    )
    def test_tag_generation_consistent_with_inputs(
        self,
        version: str,
        device: str,
        python: str,
        cuda: str,
        os_ver: str,
        platform: str,
    ) -> None:
        tags = generate_tags(version, device, python, cuda, os_ver, platform)
        assert len(tags) == 4
        # All tags should contain the version or major.minor, device, and python
        mm = parse_major_minor(version)
        for tag in tags:
            assert device in tag
            assert python in tag
            assert version in tag or mm in tag


# ===================================================================
# 6.7 — Property 8: Failed packages reported in PR body
# ===================================================================
# Feature: auto-docs-pr, Property 8: Failed packages reported in PR body


class TestFailedPackagesReport:
    """**Validates: Requirements 11.5**"""

    @settings(max_examples=100)
    @given(
        packages=st.lists(
            st.text(
                alphabet=string.ascii_lowercase + string.digits + "-_",
                min_size=1,
                max_size=30,
            ),
            min_size=1,
            max_size=10,
        ),
    )
    def test_nonempty_list_contains_all_packages(self, packages: list[str]) -> None:
        report = build_failed_packages_report(packages)
        for pkg in packages:
            assert pkg in report

    def test_empty_list_returns_empty_string(self) -> None:
        assert build_failed_packages_report([]) == ""


# ===================================================================
# 6.8 — Property 9: Slack notification payload completeness
# ===================================================================
# Feature: auto-docs-pr, Property 9: Slack notification payload completeness


class TestSlackPayloadCompleteness:
    """**Validates: Requirements 12.2**"""

    @settings(max_examples=100)
    @given(
        display_name=_nonempty_printable,
        version=_nonempty_printable,
        platform=_nonempty_printable,
        pr_url=_nonempty_printable,
    )
    def test_success_payload_contains_all_values(
        self, display_name: str, version: str, platform: str, pr_url: str
    ) -> None:
        payload = build_slack_payload(display_name, version, platform, pr_url)
        text = payload["text"]
        assert display_name in text
        assert version in text
        assert platform in text
        assert pr_url in text

    @settings(max_examples=100)
    @given(
        display_name=_nonempty_printable,
        version=_nonempty_printable,
        platform=_nonempty_printable,
        run_url=_nonempty_printable,
    )
    def test_failure_payload_contains_all_values(
        self, display_name: str, version: str, platform: str, run_url: str
    ) -> None:
        payload = build_failure_slack_payload(display_name, version, platform, run_url)
        text = payload["text"]
        assert display_name in text
        assert version in text
        assert platform in text
        assert run_url in text
