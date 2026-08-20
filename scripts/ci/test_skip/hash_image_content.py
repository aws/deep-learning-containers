"""Compute image_content_hash for a pushed image with no layer pull.

Read the config, resolve the reference in the registry, and return the image
config JSON, then hash its rootfs.diff_ids together with the runtime config
(Env, Cmd, Entrypoint, Labels, ...). The build timestamps (top-level ``created``
and ``history`` timestamps) are excluded so the hash is stable across rebuilds of
identical content.
"""

import hashlib
import json
import subprocess

DEFAULT_PLATFORM = "linux/amd64"


class PlatformNotFoundError(Exception):
    """Raised when the image's platform does not match the requested one."""


def content_hash(diff_ids, config):
    """sha256 of the ordered (diff_ids, config) pair (``sha256:<hex>``).

    ``config`` is the OCI image config object (Env, Cmd, Entrypoint, WorkingDir,
    User, Labels, etc.). Build timestamps (``created`` and ``history``) are
    not part of this pair, so identical content hashes the same across rebuilds.
    """
    if not diff_ids:
        raise ValueError("empty diff_ids — refusing to emit a content hash")
    canonical = json.dumps(
        {"diff_ids": diff_ids, "config": config or {}},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


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


def _parse_config_payload(inspect_json, platform=DEFAULT_PLATFORM):
    """Parse `imagetools inspect` output into one validated single-platform config."""
    payload = json.loads(inspect_json)
    if not isinstance(payload, dict):
        raise ValueError("unexpected imagetools inspect output (not an object)")
    if not payload.keys() & {"rootfs", "config", "architecture", "os"}:
        raise ValueError("expected a single image config (got no config fields)")

    _assert_platform_matches(payload, platform)
    return payload


def extract_diff_ids(inspect_json, platform=DEFAULT_PLATFORM):
    """Extract diff_ids from a single-platform `imagetools inspect` config."""
    return _diff_ids_from_config(_parse_config_payload(inspect_json, platform))


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
    payload = _parse_config_payload(_inspect_image_config(image_uri), platform=platform)
    diff_ids = _diff_ids_from_config(payload)
    config = payload.get("config") or {}
    return content_hash(diff_ids, config)
