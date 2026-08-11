#!/usr/bin/env python3
"""Compute image_content_hash for a pushed image with no layer pull.

We read the config via `docker buildx imagetools inspect --format '{{json
.Image}}'`, which resolves the reference in the registry and returns the image
config JSON, then hash its rootfs.diff_ids. DLC builds single-platform x86-64
images, so we expect a single config and verify its os/architecture matches
--platform.

Usage:
    python3 hash_image_content.py --image-uri <ref> [--platform linux/amd64]
"""

import argparse
import hashlib
import json
import subprocess
import sys

DEFAULT_PLATFORM = "linux/amd64"


class PlatformNotFoundError(Exception):
    """Raised when the image's platform does not match the requested one."""


def hash_diff_ids(diff_ids):
    """sha256 of the ordered diff_ids list (``sha256:<hex>``)."""
    if not diff_ids:
        raise ValueError("empty diff_ids — refusing to emit a content hash")
    digest = hashlib.sha256()
    for diff_id in diff_ids:
        digest.update(diff_id.encode("utf-8"))
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def _diff_ids_from_config(config):
    """Pull rootfs.diff_ids out of one image config object."""
    rootfs = config.get("rootfs") if isinstance(config, dict) else None
    if not isinstance(rootfs, dict) or "diff_ids" not in rootfs:
        raise ValueError("image config has no rootfs.diff_ids")
    return rootfs["diff_ids"]


def _assert_platform_matches(config, platform):
    """Raise PlatformNotFoundError if a config's own os/architecture ≠ platform."""
    if not isinstance(config, dict):
        return
    want_os, want_arch = platform.split("/")[0], platform.split("/")[-1]
    got_os, got_arch = config.get("os"), config.get("architecture")
    if got_os is not None and got_os != want_os:
        raise PlatformNotFoundError(
            f"image os {got_os!r} does not match requested platform {platform!r}"
        )
    if got_arch is not None and got_arch != want_arch:
        raise PlatformNotFoundError(
            f"image architecture {got_arch!r} does not match requested platform {platform!r}"
        )


def extract_diff_ids(inspect_json, platform=DEFAULT_PLATFORM):
    """Extract diff_ids from a single-platform `imagetools inspect` config."""
    payload = json.loads(inspect_json)
    if not isinstance(payload, dict):
        raise ValueError("unexpected imagetools inspect output (not an object)")
    if not payload.keys() & {"rootfs", "config", "architecture", "os"}:
        raise ValueError("expected a single image config (got no config fields)")

    _assert_platform_matches(payload, platform)
    return _diff_ids_from_config(payload)


def _inspect_image_config(image_uri):
    """Return the raw `imagetools inspect --format '{{json .Image}}'` stdout.

    Reads only manifest + config from the registry — no layer pull. Requires a
    prior `docker login` to the registry (the CI runner already authenticates).
    """
    result = subprocess.run(
        [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            image_uri,
            "--format",
            "{{json .Image}}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def compute_image_content_hash(image_uri, platform=DEFAULT_PLATFORM):
    """Compute the image_content_hash for a pushed image reference."""
    inspect_json = _inspect_image_config(image_uri)
    diff_ids = extract_diff_ids(inspect_json, platform=platform)
    return hash_diff_ids(diff_ids)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute image_content_hash from a pushed image (no layer pull)."
    )
    parser.add_argument(
        "--image-uri", required=True, help="ECR image reference (prefer digest-pinned)."
    )
    parser.add_argument(
        "--platform",
        default=DEFAULT_PLATFORM,
        help=f"Platform to hash (default {DEFAULT_PLATFORM}).",
    )
    args = parser.parse_args(argv)
    print(compute_image_content_hash(args.image_uri, platform=args.platform))
    return 0


if __name__ == "__main__":
    sys.exit(main())
