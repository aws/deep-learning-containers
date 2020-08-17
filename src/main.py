import argparse
import os

import utils
import constants

from image_builder import image_builder
from config import build_config


def main():
    parser = argparse.ArgumentParser(description="Program to build docker images")
    parser.add_argument("--buildspec", type=str)
    parser.add_argument("--framework", type=str)
    parser.add_argument("--device_types", type=str, default=constants.ALL)
    parser.add_argument("--image_types", type=str, default=constants.ALL)
    parser.add_argument("--py_versions", type=str, default=constants.ALL)

    args = parser.parse_args()

    device_types = args.device_types.split(",") if not args.device_types == constants.ALL else args.device_types
    image_types = args.image_types.split(",") if not args.image_types == constants.ALL else args.image_types
    py_versions = args.py_versions.split(",") if not args.py_versions == constants.ALL else args.py_versions
    # create the empty json file for images
    build_context = os.getenv("BUILD_CONTEXT")
    ei_dedicated = os.getenv("EIA_DEDICATED") == "True"
    inf_dedicated = os.getenv("INF_DEDICATED") == "True"

    # A general will work if in non-EI and non-INF mode and its framework not been disabled
    general_builder_enabled = (
        not ei_dedicated
        and not inf_dedicated
        and not build_config.ENABLE_EI_MODE
        and not build_config.ENABLE_INF_MODE
        and args.framework not in build_config.DISABLE_FRAMEWORK_TESTS
    )
    # An EI dedicated builder will work if in EI mode and its framework not been disabled
    ei_builder_enabled = (
        ei_dedicated and build_config.ENABLE_EI_MODE and args.framework not in build_config.DISABLE_FRAMEWORK_TESTS
    )

    # An INF dedicated builder will work if in INF mode and its framework has not been disabled
    inf_builder_enabled = (
        inf_dedicated and build_config.ENABLE_INF_MODE and args.framework not in build_config.DISABLE_FRAMEWORK_TESTS
    )

    utils.write_to_json_file(constants.TEST_TYPE_IMAGES_PATH, {})
    # A builder will always work if it is in non-PR context
    if general_builder_enabled or ei_builder_enabled or inf_builder_enabled or build_context != "PR":
        utils.build_setup(
            args.framework, device_types=device_types, image_types=image_types, py_versions=py_versions,
        )
        image_builder(args.buildspec)


if __name__ == "__main__":
    main()
