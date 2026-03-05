#!/usr/bin/env python3
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
"""Inspect Docker image labels: URI visibility and dlc_major_version presence.

Usage: python3 inspect_docker.py <image-uri>
"""

import argparse
import logging
import sys

from test_utils.docker_helper import get_docker_labels

import test  # noqa: F401 — triggers colored logging setup

# To enable debugging, change logging.INFO to logging.DEBUG
LOGGER = logging.getLogger("test").getChild("inspect_docker")
LOGGER.setLevel(logging.INFO)


def check_uri_visibility(labels):
    """Check URI labels use https://."""
    failed = []
    for name, value in labels.items():
        if "uri" in name.lower() and not value.startswith("https://"):
            failed.append(f"{name}: {value}")
    return failed


def check_dlc_major_version(labels):
    """Check that dlc_major_version label exists."""
    if "dlc_major_version" not in labels:
        return ["Missing required label: dlc_major_version"]
    LOGGER.info(f"dlc_major_version = {labels['dlc_major_version']}")
    return []


def main():
    parser = argparse.ArgumentParser(description="Inspect Docker image labels")
    parser.add_argument("image_uri", help="Docker image URI to inspect")
    args = parser.parse_args()

    labels = get_docker_labels(args.image_uri)
    if not labels:
        LOGGER.warning("No labels found on image")
        return 0

    failed = []
    failed.extend(check_uri_visibility(labels))
    failed.extend(check_dlc_major_version(labels))

    if failed:
        LOGGER.error("Label validation errors:")
        for error in failed:
            LOGGER.error(f"  {error}")
        return 1

    LOGGER.info(f"All {len(labels)} labels pass validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
