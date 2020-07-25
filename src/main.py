import argparse
import os

import utils
import constants

from image_builder import image_builder
from config import build_config

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
    # create the empty json file for images
    build_context = os.getenv("BUILD_CONTEXT")
    ei_dedicated = os.getenv("EIA_DEDICATED") == "True"
    utils.write_to_json_file(constants.TEST_TYPE_IMAGES_PATH, {})
    if (not ei_dedicated and not build_config.ENABLE_EI_MODE and args.framework not in build_config.DISABLE_FRAMEWORK_TESTS) or \
       (ei_dedicated and build_config.ENABLE_EI_MODE and args.framework not in build_config.DISABLE_FRAMEWORK_TESTS) or \
        build_context != "PR":
        utils.build_setup(
            args.framework,
            device_types=device_types,
            image_types=image_types,
            py_versions=py_versions,
        )
        image_builder(args.buildspec)
