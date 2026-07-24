#!/usr/bin/env python3
"""Sanity check: an AL2023 vLLM/SGLang *server* config must not declare a
framework version that is *behind* the upstream commit it pins.

Version contract for these images:

    framework_version == "<upstream_release>[+dlc<n>]"

and the pinned source ref (``sglang_ref`` / ``vllm_ref``) must resolve to an
upstream state that is *at least* ``<upstream_release>``. When the ref is moved
ahead of the latest upstream release tag (e.g. onto ``main`` past ``v0.5.14``),
``framework_version`` must be bumped so its ``<upstream_release>`` equals that
latest tag.

This guards against the failure mode where the source ref is bumped but
``framework_version`` is left stale (e.g. ref advanced past ``v0.5.14`` while
``framework_version`` stayed ``0.5.13+dlc1``). That stale value gets baked into
the container as ``SETUPTOOLS_SCM_PRETEND_VERSION`` and into the CVE-scanner
metadata, mislabelling the shipped framework version.

Runs on the CI host (needs network + the GitHub API), one config at a time:

    python3 test/sanity/scripts/check_framework_version_currency.py \\
        --config-file .github/config/image/sglang/ec2-amzn2023.yml

Set ``GITHUB_TOKEN`` (or ``GH_TOKEN``) to raise the API rate limit.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

from packaging.version import InvalidVersion, Version

# metadata.framework -> (upstream repo, config field holding the source ref).
# Only the from-source *server* images follow the "<release>+dlc<n>" contract.
FRAMEWORK_SOURCES = {
    "sglang_server": ("sgl-project/sglang", "sglang_ref"),
    "vllm_server": ("vllm-project/vllm", "vllm_ref"),
}

# Strict release tags only (no rc/dev/post/a/b): v1.2.3 or 1.2.3
_RELEASE_TAG_RE = re.compile(r"^v?\d+(?:\.\d+)*$")


def load_config(path):
    """Return the parsed config dict. Uses PyYAML if available, else yq."""
    try:
        import yaml

        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        out = subprocess.check_output(["yq", "-o=json", ".", path], text=True)
        return json.loads(out)


def to_version(tag_or_version):
    """Parse a tag name or version string into a ``Version`` (drops a ``v``
    prefix, e.g. ``v0.5.14`` -> ``Version('0.5.14')``)."""
    return Version(str(tag_or_version).lstrip("v"))


def github_get(url):
    """GET a GitHub API URL, returning parsed JSON. Uses a token if present."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "dlc-framework-version-currency-check",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def release_tags(repo):
    """Return upstream strict-release tags as ``(Version, tag_name)`` pairs,
    sorted newest-first."""
    tags = []
    for page in range(1, 6):  # up to 500 tags; newest first
        batch = github_get(f"https://api.github.com/repos/{repo}/tags?per_page=100&page={page}")
        if not batch:
            break
        for entry in batch:
            name = entry["name"]
            if _RELEASE_TAG_RE.match(name):
                try:
                    tags.append((to_version(name), name))
                except InvalidVersion:
                    continue
        if len(batch) < 100:
            break
    return sorted(set(tags), key=lambda t: t[0], reverse=True)


def ref_at_or_ahead_of(repo, tag, ref):
    """True if ``ref`` is at or ahead of ``tag`` in ``repo`` history."""
    data = github_get(f"https://api.github.com/repos/{repo}/compare/{tag}...{ref}")
    status = data.get("status")
    if status in ("identical", "ahead"):
        return True
    if status == "behind":
        return False
    if status == "diverged":
        # ref is on a newer/parallel line (e.g. main past a release-branch tag).
        # Treat it as "at least" the tag when it carries at least as much unique
        # history as the tag does.
        return data.get("ahead_by", 0) >= data.get("behind_by", 0)
    return False


def resolve_ref_release(repo, ref):
    """The highest strict-release tag that ``ref`` is at or ahead of, as a
    ``(Version, tag_name)`` pair, or ``None`` if it precedes all tags."""
    for version, name in release_tags(repo):
        if ref_at_or_ahead_of(repo, name, ref):
            return version, name
    return None


def check_config(config_path):
    """Check one config file. Returns (ok: bool, message: str)."""
    config = load_config(config_path)
    metadata = config.get("metadata", {})
    build = config.get("build", {})

    framework = metadata.get("framework", "")
    if framework not in FRAMEWORK_SOURCES:
        return True, f"SKIP: framework {framework!r} is not a vLLM/SGLang server image"

    repo, ref_field = FRAMEWORK_SOURCES[framework]
    ref = build.get(ref_field)
    if not ref:
        return True, f"SKIP: no {ref_field} in {config_path} (not a from-source build)"

    declared = to_version(metadata.get("framework_version"))
    if declared.is_devrelease:
        # A ".devN" version is a legacy setuptools_scm snapshot of an unreleased
        # upstream commit (e.g. the out-of-support hyperpod image) and does not
        # use the "<release>+dlc<n>" contract, so the check does not apply. Such
        # an image would be checked again once it adopts the +dlc convention.
        return True, f"SKIP: framework_version {declared} is a legacy dev snapshot"

    resolved = resolve_ref_release(repo, ref)
    if resolved is None:
        return True, f"SKIP: {ref[:10]} precedes all release tags of {repo}"
    resolved_version, resolved_tag = resolved

    # Compare on the release portion only, so the "+dlc<n>" local segment does
    # not read as newer than the bare upstream tag.
    declared_release = Version(declared.base_version)
    resolved_release = Version(resolved_version.base_version)

    detail = (
        f"framework_version={metadata.get('framework_version')!r} vs "
        f"{ref_field}={ref[:10]} which is at upstream {resolved_tag} in {repo}"
    )
    if declared_release < resolved_release:
        return False, (
            f"framework_version is BEHIND the pinned ref: {detail}. "
            f"Bump framework_version's upstream to {resolved_release} "
            f"(keeping the +dlc<n> suffix)."
        )
    if declared_release > resolved_release:
        # The ref has not (fully) reached the declared release. Common for a ref
        # pinned just before a release tag; warn but do not fail.
        return True, (
            f"WARNING: framework_version is ahead of the pinned ref's latest "
            f"reached release: {detail}."
        )
    return True, f"OK: {detail}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-file",
        required=True,
        help="Path to the image config YAML to check.",
    )
    args = parser.parse_args()

    try:
        ok, message = check_config(args.config_file)
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"ERROR: GitHub API request failed: {exc}", file=sys.stderr)
        return 2

    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
