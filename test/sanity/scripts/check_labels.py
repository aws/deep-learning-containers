#!/usr/bin/env python3
"""Check that all URI labels in a Docker image use https:// (public)
and label names/values are within size limits.

Usage: python3 check_labels.py <image-uri>
"""

import argparse
import json
import logging
import subprocess
import sys

from test_utils.logger import ColoredFormatter

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(ColoredFormatter())
LOGGER.addHandler(_handler)


def main():
    parser = argparse.ArgumentParser(description="Validate Docker image labels")
    parser.add_argument("image_uri", help="Docker image URI to inspect")
    args = parser.parse_args()

    result = subprocess.run(
        ["docker", "inspect", "--format={{json .Config.Labels}}", args.image_uri],
        capture_output=True,
        text=True,
        check=True,
    )
    labels = json.loads(result.stdout.strip())

    if not labels:
        LOGGER.warning("No labels found on image")
        return 0

    failed = []
    for name, value in labels.items():
        if "uri" in name.lower() and not value.startswith("https://"):
            failed.append(f"{name}: {value}")
        if len(name) > 128:
            failed.append(f"Label name exceeds 128 chars: {name}")
        if len(value) > 256:
            failed.append(f"Label value exceeds 256 chars for {name}: {value[:50]}...")

    if failed:
        LOGGER.error("Label validation errors:")
        for f in failed:
            LOGGER.error(f"  {f}")
        return 1

    LOGGER.info(f"All {len(labels)} labels pass validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
