"""Unit tests for the auto-docs-pr helper functions.

Tests known release configs against existing docs data files and covers
edge cases and error handling.
"""

from __future__ import annotations

from tests.helpers.docs_pr_functions import (
    build_failed_packages_report,
    generate_announcement,
    generate_branch_name,
    generate_pr_title,
    generate_tags,
    get_display_name,
    parse_cuda_version,
    parse_pip_version,
)

# ===================================================================
# 7.1 — Unit tests for known release configs against existing docs data
# ===================================================================


class TestVllmEc2KnownConfig:
    """Verify helpers match docs/src/data/vllm/0.17.1-gpu-ec2.yml exactly."""

    def test_tags(self) -> None:
        tags = generate_tags("0.17.1", "gpu", "py312", "cu129", "ubuntu22.04", "ec2")
        assert tags == [
            "0.17.1-gpu-py312-cu129-ubuntu22.04-ec2",
            "0.17-gpu-py312-cu129-ubuntu22.04-ec2-v1",
            "0.17.1-gpu-py312-ec2",
            "0.17-gpu-py312-ec2",
        ]

    def test_announcement(self) -> None:
        result = generate_announcement("vllm", "0.17.1", "ec2")
        assert result == "Introduced vLLM 0.17.1 containers for EC2, ECS, EKS"


class TestVllmSagemakerKnownConfig:
    """Verify helpers match docs/src/data/vllm/0.17.1-gpu-sagemaker.yml exactly."""

    def test_tags(self) -> None:
        tags = generate_tags("0.17.1", "gpu", "py312", "cu129", "ubuntu22.04", "sagemaker")
        assert tags == [
            "0.17.1-gpu-py312-cu129-ubuntu22.04-sagemaker",
            "0.17-gpu-py312-cu129-ubuntu22.04-sagemaker-v1",
            "0.17.1-gpu-py312",
            "0.17-gpu-py312",
        ]

    def test_announcement(self) -> None:
        result = generate_announcement("vllm", "0.17.1", "sagemaker")
        assert result == "Introduced vLLM 0.17.1 containers for SageMaker"


class TestSglangSagemakerKnownConfig:
    """Verify helpers match docs/src/data/sglang/0.5.9-gpu-sagemaker.yml exactly."""

    def test_tags(self) -> None:
        tags = generate_tags("0.5.9", "gpu", "py312", "cu129", "ubuntu24.04", "sagemaker")
        assert tags == [
            "0.5.9-gpu-py312-cu129-ubuntu24.04-sagemaker",
            "0.5-gpu-py312-cu129-ubuntu24.04-sagemaker-v1",
            "0.5.9-gpu-py312",
            "0.5-gpu-py312",
        ]

    def test_announcement(self) -> None:
        result = generate_announcement("sglang", "0.5.9", "sagemaker")
        assert result == "Introduced SGLang 0.5.9 containers for SageMaker"


class TestDisplayNames:
    """Verify display name mapping for known frameworks."""

    def test_vllm(self) -> None:
        assert get_display_name("vllm") == "vLLM"

    def test_sglang(self) -> None:
        assert get_display_name("sglang") == "SGLang"


class TestBranchNamesAndPrTitles:
    """Verify branch names and PR titles for known configs."""

    def test_vllm_ec2_branch(self) -> None:
        assert generate_branch_name("vllm", "0.17.1", "ec2") == "docs/auto-update-vllm-0.17.1-ec2"

    def test_vllm_sagemaker_branch(self) -> None:
        assert (
            generate_branch_name("vllm", "0.17.1", "sagemaker")
            == "docs/auto-update-vllm-0.17.1-sagemaker"
        )

    def test_sglang_sagemaker_branch(self) -> None:
        assert (
            generate_branch_name("sglang", "0.5.9", "sagemaker")
            == "docs/auto-update-sglang-0.5.9-sagemaker"
        )

    def test_vllm_ec2_pr_title(self) -> None:
        assert generate_pr_title("vllm", "0.17.1", "ec2") == "docs: Add vLLM 0.17.1 EC2 image data"

    def test_vllm_sagemaker_pr_title(self) -> None:
        assert (
            generate_pr_title("vllm", "0.17.1", "sagemaker")
            == "docs: Add vLLM 0.17.1 SAGEMAKER image data"
        )

    def test_sglang_sagemaker_pr_title(self) -> None:
        assert (
            generate_pr_title("sglang", "0.5.9", "sagemaker")
            == "docs: Add SGLang 0.5.9 SAGEMAKER image data"
        )


# ===================================================================
# 7.2 — Unit tests for edge cases and error handling
# ===================================================================


class TestParsePipVersionEdgeCases:
    """Edge cases for parse_pip_version."""

    def test_empty_output(self) -> None:
        assert parse_pip_version("") == ""

    def test_missing_version_line(self) -> None:
        assert parse_pip_version("Name: some-pkg\nSummary: A package") == ""


class TestDisplayNameFallback:
    """Unknown framework falls back to raw identifier."""

    def test_unknown_framework(self) -> None:
        assert get_display_name("unknown-framework") == "unknown-framework"


class TestBuildFailedPackagesReport:
    """Edge cases for build_failed_packages_report."""

    def test_empty_list(self) -> None:
        assert build_failed_packages_report([]) == ""

    def test_contains_both_packages(self) -> None:
        report = build_failed_packages_report(["pkg1", "pkg2"])
        assert "pkg1" in report
        assert "pkg2" in report


class TestParseCudaVersionEdgeCases:
    """Edge cases for parse_cuda_version."""

    def test_empty_output(self) -> None:
        assert parse_cuda_version("") == ""


class TestGenerateTagsUnknownPlatform:
    """Unknown platform returns empty list."""

    def test_unknown_platform(self) -> None:
        assert generate_tags("1.0.0", "gpu", "py312", "cu129", "ubuntu22.04", "unknown") == []
