import argparse

import utils
import constants

from image_builder import image_builder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Program to build docker images")
    parser.add_argument("--buildspec", type=str)
    parser.add_argument("--framework", type=str)
    parser.add_argument("--device_types", type=str, default=constants.ALL)
    parser.add_argument("--image_types", type=str, default=constants.ALL)
    parser.add_argument("--py_versions", type=str, default=constants.ALL)

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
    py_versions = (
        args.py_versions.split(",")
        if not args.py_versions == constants.ALL
        else args.py_versions
    )

    utils.build_setup(
        args.framework,
        device_types=device_types,
        image_types=image_types,
        py_versions=py_versions,
    )
    image_builder(args.buildspec)
