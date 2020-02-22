from github import GitHubHandler
import os
from image_builder import image_builder
import argparse
import constants
import re

import utils
import constants

from context import Context
from metrics import Metrics
from image import DockerImage
from buildspec import Buildspec
from output import OutputFormatter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Program to build docker images")
    parser.add_argument("--buildspec", type=str)
    parser.add_argument("--framework", type=str)
    parser.add_argument("--device_types", type=str, default=constants.ALL)
    parser.add_argument("--image_types", type=str, default=constants.ALL)
    parser.add_argument("--python_versions", type=str, default=constants.ALL)

    args = parser.parse_args()

    device_types = (
        args.device_types.split(",")
        if not args.device_types == constants.ALL
        else args.device_types
    )
    image_types = (
        args.image_types.split(",")
        if not args.image_types == constants.ALL
        else args.image_types
    )
    python_versions = (
        args.python_versions.split(",")
        if not args.python_version == contstants.ALL
        else args.python_versions
    )

    utils.build_setup(
        args.framework,
        device_types=device_types,
        image_types=image_types,
        python_versions=python_versions,
    )
    image_builder(args.buildspec)
