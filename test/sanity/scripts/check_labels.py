#!/usr/bin/env python3
"""Validate standard SageMaker DLC labels and max label count.

Usage:
    python3 check_labels.py <image-uri> \
        --framework sglang --framework-version 0.5.9 \
        --device-type gpu --cuda-version cu129 --arch-type x86 \
        --python-version py312 --os-version ubuntu24.04 \
        --container-type general --contributor None
"""

import argparse
import logging
import sys
from pprint import pformat

from test_utils.docker_helper import get_docker_labels

import test  # noqa: F401 — triggers colored logging setup

# To enable debugging, change logging.INFO to logging.DEBUG
LOGGER = logging.getLogger("test").getChild("check_labels")
LOGGER.setLevel(logging.INFO)

SAGEMAKER_LABEL_PREFIX = "com.amazonaws.ml.engines.sagemaker.dlc"
MAX_SAGEMAKER_LABELS = 10


def build_expected_labels(args):
    """Construct expected sagemaker label keys from CLI args, mirroring build_image.sh logic."""
    fw = args.framework.replace("_", "-")
    fw_ver = args.framework_version.replace(".", "-")
    device = args.device_type
    if device == "gpu" and args.cuda_version:
        device = f"{device}.{args.cuda_version}"

    expected = [
        f"{SAGEMAKER_LABEL_PREFIX}.framework.{fw}.{fw_ver}",
        f"{SAGEMAKER_LABEL_PREFIX}.device.{device}",
        f"{SAGEMAKER_LABEL_PREFIX}.job.{args.container_type}",
        f"{SAGEMAKER_LABEL_PREFIX}.arch.{args.arch_type}",
    ]
    if args.os_version:
        os_ver = args.os_version.replace(".", "-")
        expected.append(f"{SAGEMAKER_LABEL_PREFIX}.os.{os_ver}")
    if args.python_version:
        python_ver = args.python_version.replace(".", "-")
        expected.append(f"{SAGEMAKER_LABEL_PREFIX}.python.{python_ver}")
    if args.contributor:
        expected.append(f"{SAGEMAKER_LABEL_PREFIX}.contributor.{args.contributor}")
    if args.transformers_version:
        transformers_ver = args.transformers_version.replace(".", "-")
        expected.append(f"{SAGEMAKER_LABEL_PREFIX}.lib.transformers.{transformers_ver}")
    return expected


def check_standard_labels(labels, args):
    """Validate standard SageMaker labels are present."""
    expected = build_expected_labels(args)
    observed = [key for key in labels if key.startswith(SAGEMAKER_LABEL_PREFIX)]

    LOGGER.info(f"Expected labels ({len(expected)}):\n{pformat(expected)}")
    LOGGER.info(f"Observed labels on container ({len(observed)}):\n{pformat(observed)}")

    missing = [label for label in expected if label not in labels]
    if missing:
        return [f"Missing standard SageMaker label: {label}" for label in missing]
    LOGGER.info(f"All {len(expected)} standard SageMaker labels present")
    return []


def check_max_sagemaker_labels(labels):
    """Ensure no more than {MAX_SAGEMAKER_LABELS} SageMaker labels."""
    sm_labels = [key for key in labels if key.startswith(SAGEMAKER_LABEL_PREFIX)]
    if len(sm_labels) > MAX_SAGEMAKER_LABELS:
        return [
            f"Too many SageMaker labels: {len(sm_labels)} (max {MAX_SAGEMAKER_LABELS}): {sm_labels}"
        ]
    LOGGER.info(f"SageMaker label count: {len(sm_labels)} (max {MAX_SAGEMAKER_LABELS})")
    return []


def main():
    parser = argparse.ArgumentParser(description="Validate standard SageMaker DLC labels")
    parser.add_argument("image_uri", help="Docker image URI to inspect")
    parser.add_argument("--framework", required=True)
    parser.add_argument("--framework-version", required=True)
    parser.add_argument("--container-type", required=True)
    parser.add_argument("--device-type", default="gpu")
    parser.add_argument("--arch-type", default="x86")
    parser.add_argument("--contributor", default="None")
    parser.add_argument("--cuda-version", default="")
    parser.add_argument("--python-version", default="")
    parser.add_argument("--os-version", default="")
    parser.add_argument("--transformers-version", default="")
    args = parser.parse_args()

    # Treat empty strings as unset
    for attr in vars(args):
        if getattr(args, attr) == "":
            setattr(args, attr, None)

    required_args = ["framework", "framework_version", "container_type"]
    missing = [arg for arg in required_args if not getattr(args, arg)]
    if missing:
        LOGGER.error(
            f"Required arguments cannot be empty: {', '.join('--' + a.replace('_', '-') for a in missing)}"
        )
        return 1

    labels = get_docker_labels(args.image_uri)
    if not labels:
        LOGGER.warning("No labels found on image")
        return 1

    failed = []
    failed.extend(check_standard_labels(labels, args))
    failed.extend(check_max_sagemaker_labels(labels))

    if failed:
        LOGGER.error("Label validation errors:")
        for error in failed:
            LOGGER.error(f"  {error}")
        return 1

    LOGGER.info("SageMaker label validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
