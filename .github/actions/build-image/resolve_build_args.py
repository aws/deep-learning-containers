#!/usr/bin/env python3
"""Resolve build-args from image config file.

Reads the build: block from a YAML config file, skips reserved keys, and
writes each remaining key as UPPER_CASE=value to $GITHUB_ENV. Also writes
EXTRA_BUILD_ARGS (space-separated list of key names) for build_image.sh.

Reserved keys (not forwarded as --build-arg):
  dockerfile, target

`efa_version` accepts the literal `latest`, resolved here to a real version
number so it still reaches the image label, the sanity test, and release notes.
AWS publishes no version index for the installer, so the number is looked up.

Keys in PRESERVE_CASE_KEYS are forwarded verbatim instead of upper-cased, for
Dockerfiles that declare a lower-case ARG. Docker build-args are case-sensitive,
so an upper-cased name silently fails to bind to a lower-case ARG.

Usage in GitHub Actions:
  - name: Resolve build args
    run: python3 scripts/ci/resolve_build_args.py --config-file ${{ env.CONFIG_FILE }}

Local usage:
  python3 scripts/ci/resolve_build_args.py --config-file .github/refactor/config/image/vllm/ec2-amzn2023.yml

Requires: pyyaml (pip install pyyaml) or yq on PATH as fallback.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request

RESERVED_KEYS = {"dockerfile", "target"}

# Keys whose Dockerfile ARG is lower-case, so the name must not be upper-cased.
# vLLM's upstream setup.py reads TORCH_CUDA_ARCH_LIST from the environment, which
# the Dockerfile sets from a lower-case `ARG torch_cuda_arch_list`.
PRESERVE_CASE_KEYS = {"torch_cuda_arch_list"}

# Same host install_efa*.sh downloads from, so a resolved version is fetchable.
EFA_INSTALLER_BASE_URL = "https://efa-installer.amazonaws.com"

# The install docs name the current version in their download command, making
# them a self-updating starting point rather than a pin we maintain.
EFA_DOCS_URL = "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html"

# Read only when the docs are unreachable, so keep it a version that exists.
EFA_VERSION_FALLBACK = "1.50.0"

# Bounds the forward walk so a misbehaving endpoint cannot spin.
EFA_MAX_PROBES = 25


def load_yaml(path):
    """Load YAML file, trying pyyaml first, falling back to yq."""
    try:
        import yaml

        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        result = subprocess.run(
            ["yq", "-o=json", ".", path], capture_output=True, text=True, check=True
        )
        return json.loads(result.stdout)


def _warn(message):
    """Warn so it lands on the workflow summary, not just in the log body."""
    print(f"::warning::{message}" if os.environ.get("GITHUB_ACTIONS") else f"  WARNING: {message}")


def _fail(message):
    """Fail the job, annotated so the reason shows on the workflow summary."""
    prefix = "::error::" if os.environ.get("GITHUB_ACTIONS") else "ERROR: "
    print(f"{prefix}{message}", file=sys.stderr)
    sys.exit(1)


def _efa_tarball_etag(version):
    """ETag for an EFA installer tarball, or None if that version is absent.

    HEAD, so the tarball is never downloaded. An unpublished version answers
    403 rather than 404 because listing is denied.
    """
    url = f"{EFA_INSTALLER_BASE_URL}/aws-efa-installer-{version}.tar.gz"
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.headers.get("ETag")
    except OSError:
        # An unpublished version and a network failure both land here; neither
        # should fail the build, so the caller falls back.
        return None


def _efa_version_from_docs():
    """Newest version named by the EFA install docs, or None if unreadable."""
    try:
        with urllib.request.urlopen(EFA_DOCS_URL, timeout=15) as response:
            page = response.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    found = re.findall(r"aws-efa-installer-(\d+\.\d+\.\d+)\.tar\.gz", page)
    if not found:
        return None
    return max(found, key=lambda v: tuple(int(part) for part in v.split(".")))


def _successors(version):
    """Next patch and next minor after `version`, in release order."""
    major, minor, patch = (int(part) for part in version.split("."))
    return (f"{major}.{minor}.{patch + 1}", f"{major}.{minor + 1}.0")


def resolve_efa_version(pinned):
    """Turn `latest` into a concrete EFA installer version.

    An explicit version is returned untouched, so one image can stay pinned
    while the rest float. `latest` is the same object as the newest numbered
    tarball, so a matching ETag proves which number it is.
    """
    if pinned and pinned != "latest":
        return pinned

    latest_etag = _efa_tarball_etag("latest")
    if latest_etag is None:
        _fail(f"Cannot reach {EFA_INSTALLER_BASE_URL} to resolve EFA 'latest'.")

    candidate = _efa_version_from_docs()
    if candidate is None:
        _warn(
            f"Could not read an EFA version from {EFA_DOCS_URL} (page moved?); "
            f"searching upward from {EFA_VERSION_FALLBACK} instead."
        )
    candidate_etag = _efa_tarball_etag(candidate) if candidate else None
    if candidate_etag is None:
        if candidate is not None:
            _warn(
                f"{EFA_DOCS_URL} names {candidate}, but no tarball is published for it yet; "
                f"searching upward from {EFA_VERSION_FALLBACK} instead."
            )
        candidate = EFA_VERSION_FALLBACK
        candidate_etag = _efa_tarball_etag(candidate)

    if candidate_etag == latest_etag:
        return candidate

    # Candidate is behind: step forward until one is byte-identical to `latest`.
    for _ in range(EFA_MAX_PROBES):
        for successor in _successors(candidate):
            etag = _efa_tarball_etag(successor)
            if etag is None:
                continue
            candidate = successor
            if etag == latest_etag:
                return candidate
            break
        else:
            break

    _fail(
        f"Could not determine which version EFA 'latest' is, searching up from {candidate}. "
        f"Raise EFA_VERSION_FALLBACK (currently {EFA_VERSION_FALLBACK}) or check {EFA_DOCS_URL}."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Resolve build-args from image config file.")
    parser.add_argument("--config-file", required=True, help="Path to the image config YAML file")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.config_file):
        print(f"ERROR: Config file not found: {args.config_file}", file=sys.stderr)
        sys.exit(1)

    config = load_yaml(args.config_file)
    build = config.get("build", {})

    github_env = os.environ.get("GITHUB_ENV")
    keys = []

    for key, value in build.items():
        if key in RESERVED_KEYS:
            continue
        if key == "efa_version":
            value = resolve_efa_version(str(value))
        env_key = key if key in PRESERVE_CASE_KEYS else key.upper()
        keys.append(env_key)

        if github_env:
            with open(github_env, "a") as f:
                f.write(f"{env_key}={value}\n")

        print(f"  {env_key}={value}")

    extra = " ".join(keys)

    if github_env:
        with open(github_env, "a") as f:
            f.write(f"EXTRA_BUILD_ARGS={extra}\n")

    print(f"\nEXTRA_BUILD_ARGS={extra}")
    print(f"Total: {len(keys)} build-args resolved")


if __name__ == "__main__":
    main()
