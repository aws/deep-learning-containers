import argparse
import os

import utils
import constants

from image_builder import image_builder
from config import parse_dlc_developer_configs


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
    ei_dedicated = os.getenv("EIA_DEDICATED", "false").lower() == "true"
    neuron_dedicated = os.getenv("NEURON_DEDICATED", "false").lower() == "true"
    graviton_dedicated = os.getenv("GRAVITON_DEDICATED", "false").lower() == "true"
    habana_dedicated = os.getenv("HABANA_DEDICATED", "false").lower() == "true"
    hopper_dedicated = os.getenv("HOPPER_DEDICATED", "false").lower() == "true"

    # Get config value options
    frameworks_to_skip = parse_dlc_developer_configs("build", "skip_frameworks")
    ei_build_mode = parse_dlc_developer_configs("dev", "ei_mode")
    neuron_build_mode = parse_dlc_developer_configs("dev", "neuron_mode")
    graviton_build_mode = parse_dlc_developer_configs("dev", "graviton_mode")
    habana_build_mode = parse_dlc_developer_configs("dev", "habana_mode")
    hopper_build_mode = parse_dlc_developer_configs("dev", "hopper_mode")

    # A general will work if in non-EI, non-NEURON and non-GRAVITON mode and its framework not been disabled
    general_builder_enabled = (
        not ei_dedicated
        and not neuron_dedicated
        and not graviton_dedicated
        and not habana_dedicated
        and not hopper_dedicated
        and not ei_build_mode
        and not neuron_build_mode
        and not graviton_build_mode
        and not habana_build_mode
        and not hopper_build_mode
        and args.framework not in frameworks_to_skip
    )
    # An EI dedicated builder will work if in EI mode and its framework not been disabled
    ei_builder_enabled = (
        ei_dedicated and ei_build_mode and args.framework not in frameworks_to_skip
    )

    # A NEURON dedicated builder will work if in NEURON mode and its framework has not been disabled
    neuron_builder_enabled = (
        neuron_dedicated
        and neuron_build_mode
        and args.framework not in frameworks_to_skip
    )

    # A GRAVITON dedicated builder will work if in GRAVITON mode and its framework has not been disabled
    graviton_builder_enabled = (
        graviton_dedicated
        and graviton_build_mode
        and args.framework not in frameworks_to_skip
    )

    # A HABANA dedicated builder will work if in HABANA mode and its framework has not been disabled
    habana_builder_enabled = (
        habana_dedicated
        and habana_build_mode
        and args.framework not in frameworks_to_skip
    )

    # A HOPPER dedicated builder will work if in HOPPER mode and its framework has not been disabled.
    hopper_builder_enabled = (
        hopper_dedicated
        and hopper_build_mode
        and args.framework not in frameworks_to_skip
    )

    utils.write_to_json_file(constants.TEST_TYPE_IMAGES_PATH, {})
    # A builder will always work if it is in non-PR context
    if (
        general_builder_enabled
        or ei_builder_enabled
        or neuron_builder_enabled
        or habana_builder_enabled
        or hopper_builder_enabled
        or build_context != "PR"
    ):
        utils.build_setup(
            args.framework, device_types=device_types, image_types=image_types, py_versions=py_versions,
        )
        image_builder(args.buildspec)


if __name__ == "__main__":
    main()
