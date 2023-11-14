import argparse
import os
import re

import config
import constants
import utils

from codebuild_environment import get_codebuild_project_name
from image_builder import image_builder


def main():
    parser = argparse.ArgumentParser(description="Program to build docker images")
    parser.add_argument("--buildspec", type=str)
    parser.add_argument("--framework", type=str)
    parser.add_argument("--device_types", type=str, default=constants.ALL)
    parser.add_argument("--image_types", type=str, default=constants.ALL)
    parser.add_argument("--py_versions", type=str, default=constants.ALL)

    args = parser.parse_args()

    image_types = []
    py_versions = []
    device_types = []

    if args.device_types != constants.ALL:
        device_types = args.device_types.split(",")

    if args.image_types != constants.ALL:
        image_types = args.image_types.split(",")

    if args.py_versions != constants.ALL:
        py_versions = args.py_versions.split(",")

    # create the empty json file for images
    build_context = os.getenv("BUILD_CONTEXT")

    # Write empty dict to JSON file, so subsequent buildspec steps do not fail in case we skip
    # this build.
    utils.write_to_json_file(constants.TEST_TYPE_IMAGES_PATH, {})

    # Skip tensorflow-1 PR jobs, as there are no longer patch releases being added for TF1
    # Purposefully not including this in developer config to make this difficult to enable
    # TODO: Remove when we remove these jobs completely
    build_name = get_codebuild_project_name()
    if build_context == "PR" and build_name == "dlc-pr-tensorflow-1":
        return

    # A general build will work if build job and build mode are in non-EI, non-NEURON
    # and non-GRAVITON mode, and its framework and image-type has not been disabled.
    general_builder_enabled = (
        config.is_general_builder_enabled_for_this_pr_build(args.framework)
        and not config.is_deep_canary_mode_enabled()
    )

    # An EI dedicated builder will work if in EI mode and its framework not been disabled
    ei_builder_enabled = (
        config.is_ei_builder_enabled_for_this_pr_build(args.framework)
        and not config.is_deep_canary_mode_enabled()
    )

    # A NEURON dedicated builder will work if in NEURON mode and its framework has not been disabled
    neuron_builder_enabled = (
        config.is_neuron_builder_enabled_for_this_pr_build(args.framework)
        and not config.is_deep_canary_mode_enabled()
    )

    # A NEURONX dedicated builder will work if in NEURONX mode and its framework has not
    # been disabled.
    neuronx_builder_enabled = (
        config.is_neuronx_builder_enabled_for_this_pr_build(args.framework)
        and not config.is_deep_canary_mode_enabled()
    )

    # A GRAVITON dedicated builder will work if in GRAVITON mode and its framework has not
    # been disabled.
    graviton_builder_enabled = (
        config.is_graviton_builder_enabled_for_this_pr_build(args.framework)
        and not config.is_deep_canary_mode_enabled()
    )

    # A HABANA dedicated builder will work if in HABANA mode and its framework has not been disabled
    habana_builder_enabled = (
        config.is_habana_builder_enabled_for_this_pr_build(args.framework)
        and not config.is_deep_canary_mode_enabled()
    )

    # A HUGGINGFACE TRCOMP dedicated builder will work if in HUGGINGFACE_TRCOMP mode and its
    # framework has not been disabled.
    hf_trcomp_builder_enabled = (
        config.is_hf_trcomp_builder_enabled_for_this_pr_build(args.framework)
        and not config.is_deep_canary_mode_enabled()
    )

    # A TRCOMP dedicated builder will work if in TRCOMP mode and its framework has not been disabled
    trcomp_builder_enabled = (
        config.is_trcomp_builder_enabled_for_this_pr_build(args.framework)
        and not config.is_deep_canary_mode_enabled()
    )

    buildspec_file = config.get_buildspec_override() or args.buildspec

    # Ensure that buildspec_file starts with buildspec and ends with yml
    buildspec_pattern = re.compile(r"buildspec\S*\.yml")
    assert buildspec_pattern.match(
        os.path.basename(buildspec_file)
    ), f"{buildspec_file} must match {buildspec_pattern.pattern}. Please rename file."

    # A builder will always work if it is in non-PR context
    if (
        general_builder_enabled
        or ei_builder_enabled
        or neuron_builder_enabled
        or neuronx_builder_enabled
        or graviton_builder_enabled
        or habana_builder_enabled
        or hf_trcomp_builder_enabled
        or trcomp_builder_enabled
        or build_context != "PR"
    ):
        utils.build_setup(
            args.framework,
            device_types=device_types,
            image_types=image_types,
            py_versions=py_versions,
        )
        image_builder(buildspec_file, image_types, device_types)


if __name__ == "__main__":
    main()
