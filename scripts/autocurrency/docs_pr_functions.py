"""Pure functions extracted from the step4-docs-pr workflow for testing.

These functions mirror the shell logic in the GitHub Actions workflow
so that correctness properties can be validated with hypothesis and pytest.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Display-name mapping
# ---------------------------------------------------------------------------
_DISPLAY_NAMES: dict[str, str] = {
    "vllm": "vLLM",
    "sglang": "SGLang",
}


def get_display_name(framework: str) -> str:
    """Map a framework identifier to its human-readable display name.

    Falls back to the raw *framework* string when no mapping exists.
    """
    return _DISPLAY_NAMES.get(framework, framework)


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------


def parse_major_minor(version: str) -> str:
    """Extract the ``X.Y`` prefix from a version string like ``"0.17.1"``."""
    parts = version.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return version


# ---------------------------------------------------------------------------
# Tag generation
# ---------------------------------------------------------------------------


def generate_tags(
    version: str,
    device: str,
    python: str,
    cuda: str,
    os_ver: str,
    platform: str,
) -> list[str]:
    """Generate the list of Docker image tags for a given platform.

    Returns 4 tags whose format depends on *platform*:

    **ec2** (all suffixed with ``-ec2``):
      1. ``{version}-{device}-{python}-{cuda}-{os}-ec2``
      2. ``{major.minor}-{device}-{python}-{cuda}-{os}-ec2-v1``
      3. ``{version}-{device}-{python}-ec2``
      4. ``{major.minor}-{device}-{python}-ec2``

    **sagemaker**:
      1. ``{version}-{device}-{python}-{cuda}-{os}-sagemaker``
      2. ``{major.minor}-{device}-{python}-{cuda}-{os}-sagemaker-v1``
      3. ``{version}-{device}-{python}``  (no platform suffix)
      4. ``{major.minor}-{device}-{python}``  (no platform suffix)
    """
    mm = parse_major_minor(version)

    if platform == "ec2":
        return [
            f"{version}-{device}-{python}-{cuda}-{os_ver}-ec2",
            f"{mm}-{device}-{python}-{cuda}-{os_ver}-ec2-v1",
            f"{version}-{device}-{python}-ec2",
            f"{mm}-{device}-{python}-ec2",
        ]
    elif platform == "sagemaker":
        return [
            f"{version}-{device}-{python}-{cuda}-{os_ver}-sagemaker",
            f"{mm}-{device}-{python}-{cuda}-{os_ver}-sagemaker-v1",
            f"{version}-{device}-{python}",
            f"{mm}-{device}-{python}",
        ]
    else:
        return []


# ---------------------------------------------------------------------------
# Announcement generation
# ---------------------------------------------------------------------------


def generate_announcement(framework: str, version: str, platform: str) -> str:
    """Return a platform-specific release announcement string."""
    display = get_display_name(framework)
    if platform == "ec2":
        return f"Introduced {display} {version} containers for EC2, ECS, EKS"
    elif platform == "sagemaker":
        return f"Introduced {display} {version} containers for SageMaker"
    return ""


# ---------------------------------------------------------------------------
# Branch / PR helpers
# ---------------------------------------------------------------------------


def generate_branch_name(framework: str, version: str, platform: str) -> str:
    """Return the git branch name for a docs-update PR."""
    return f"docs/auto-update-{framework}-{version}-{platform}"


def generate_pr_title(framework: str, version: str, platform: str) -> str:
    """Return the pull-request title with the platform uppercased."""
    display = get_display_name(framework)
    return f"docs: Add {display} {version} {platform.upper()} image data"


# ---------------------------------------------------------------------------
# Command-output parsers
# ---------------------------------------------------------------------------


def parse_pip_version(pip_show_output: str) -> str:
    """Extract the version from ``pip show`` output.

    Looks for a line matching ``Version: X.Y.Z`` and returns the version
    string.  Returns ``""`` when no match is found.
    """
    for line in pip_show_output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Version:"):
            parts = stripped.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return ""


def parse_cuda_version(nvcc_output: str) -> str:
    """Extract the CUDA version from ``nvcc --version`` output.

    Looks for the pattern ``release X.Y`` and returns ``X.Y``.
    Returns ``""`` when no match is found.
    """
    match = re.search(r"release\s+(\d+\.\d+)", nvcc_output)
    if match:
        return match.group(1)
    return ""


# ---------------------------------------------------------------------------
# Slack notification payloads
# ---------------------------------------------------------------------------


def build_slack_payload(
    display_name: str,
    version: str,
    platform: str,
    pr_url: str,
) -> dict:
    """Build the Slack JSON payload for a successful docs-PR notification."""
    return {
        "text": (f"\U0001f4c4 Docs PR created for {display_name} {version} ({platform}): {pr_url}"),
    }


def build_failure_slack_payload(
    display_name: str,
    version: str,
    platform: str,
    run_url: str,
) -> dict:
    """Build the Slack JSON payload for a failed docs-PR notification."""
    return {
        "text": (f"\u274c Docs PR job failed for {display_name} {version} ({platform}): {run_url}"),
    }


# ---------------------------------------------------------------------------
# PR body helpers
# ---------------------------------------------------------------------------


def build_failed_packages_report(failed_packages: list[str]) -> str:
    """Return a warning section listing *failed_packages* for the PR body.

    Returns an empty string when the list is empty.
    """
    if not failed_packages:
        return ""

    lines = [
        "⚠️ **Warning: Some package versions could not be extracted:**",
        "",
    ]
    for pkg in failed_packages:
        lines.append(f"- `{pkg}`")
    lines.append("")
    return "\n".join(lines)
