import argparse
import os
import re

import utils
import constants

from codebuild_environment import get_codebuild_project_name
from config import parse_dlc_developer_configs, get_buildspec_override, is_deep_canary_mode_enabled
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
    ei_dedicated = os.getenv("EIA_DEDICATED", "false").lower() == "true"
    neuron_dedicated = os.getenv("NEURON_DEDICATED", "false").lower() == "true"
    neuronx_dedicated = os.getenv("NEURONX_DEDICATED", "false").lower() == "true"
    graviton_dedicated = os.getenv("GRAVITON_DEDICATED", "false").lower() == "true"
    habana_dedicated = os.getenv("HABANA_DEDICATED", "false").lower() == "true"
    hf_trcomp_dedicated = os.getenv("HUGGINFACE_TRCOMP_DEDICATED", "false").lower() == "true"
    trcomp_dedicated = os.getenv("TRCOMP_DEDICATED", "false").lower() == "true"
    image_type = os.getenv("IMAGE_TYPE", "").lower()
    training_dedicated = image_type == "training"
    inference_dedicated = image_type == "inference"

    # Get config value options
    frameworks_to_build = parse_dlc_developer_configs("build", "build_frameworks")
    training_enabled = parse_dlc_developer_configs("build", "build_training")
    inference_enabled = parse_dlc_developer_configs("build", "build_inference")
    ei_build_mode = parse_dlc_developer_configs("dev", "ei_mode")
    neuron_build_mode = parse_dlc_developer_configs("dev", "neuron_mode")
    neuronx_build_mode = parse_dlc_developer_configs("dev", "neuronx_mode")
    graviton_build_mode = parse_dlc_developer_configs("dev", "graviton_mode")
    habana_build_mode = parse_dlc_developer_configs("dev", "habana_mode")
    trcomp_build_mode = parse_dlc_developer_configs("dev", "trcomp_mode")
    hf_trcomp_build_mode = parse_dlc_developer_configs("dev", "huggingface_trcomp_mode")
    deep_canary_mode = is_deep_canary_mode_enabled()

    # Condition to check whether training or inference dedicated/enabled
    # If image_type is empty, assume this is not a training or inference specific job, and allow 'True' state
    train_or_inf_enabled = (
        (training_dedicated and training_enabled)
        or (inference_dedicated and inference_enabled)
        or (image_type == "")
    )

    # Write empty dict to JSON file, so subsequent buildspec steps do not fail in case we skip this build
    utils.write_to_json_file(constants.TEST_TYPE_IMAGES_PATH, {})

    # Skip tensorflow-1 PR jobs, as there are no longer patch releases being added for TF1
    # Purposefully not including this in developer config to make this difficult to enable
    # TODO: Remove when we remove these jobs completely
    build_name = get_codebuild_project_name()
    if build_context == "PR" and build_name == "dlc-pr-tensorflow-1":
        return

    # A general will work if in non-EI, non-NEURON and non-GRAVITON mode and its framework not been disabled
    general_builder_enabled = (
        not ei_dedicated
        and not neuron_dedicated
        and not neuronx_dedicated
        and not graviton_dedicated
        and not habana_dedicated
        and not hf_trcomp_dedicated
        and not trcomp_dedicated
        and not ei_build_mode
        and not neuron_build_mode
        and not neuronx_build_mode
        and not graviton_build_mode
        and not habana_build_mode
        and not hf_trcomp_build_mode
        and not trcomp_build_mode
        and not deep_canary_mode
        and args.framework in frameworks_to_build
        and train_or_inf_enabled
    )
    # An EI dedicated builder will work if in EI mode and its framework not been disabled
    ei_builder_enabled = (
        ei_dedicated
        and ei_build_mode
        and args.framework in frameworks_to_build
        and train_or_inf_enabled
    )

    # A NEURON dedicated builder will work if in NEURON mode and its framework has not been disabled
    neuron_builder_enabled = (
        neuron_dedicated
        and neuron_build_mode
        and args.framework in frameworks_to_build
        and train_or_inf_enabled
    )

    neuronx_builder_enabled = (
        neuronx_dedicated
        and neuronx_build_mode
        and args.framework in frameworks_to_build
        and train_or_inf_enabled
    )

    # A GRAVITON dedicated builder will work if in GRAVITON mode and its framework has not been disabled
    graviton_builder_enabled = (
        graviton_dedicated
        and graviton_build_mode
        and args.framework in frameworks_to_build
        and not deep_canary_mode
        and train_or_inf_enabled
    )

    # A HABANA dedicated builder will work if in HABANA mode and its framework has not been disabled
    habana_builder_enabled = (
        habana_dedicated
        and habana_build_mode
        and args.framework in frameworks_to_build
        and train_or_inf_enabled
    )

    # A HUGGINGFACE TRCOMP dedicated builder will work if in HUGGINGFACE TRCOMP mode and its framework has not been disabled.
    hf_trcomp_builder_enabled = (
        hf_trcomp_dedicated
        and hf_trcomp_build_mode
        and args.framework in frameworks_to_build
        and train_or_inf_enabled
    )

    # A TRCOMP dedicated builder will work if in TRCOMP mode and its framework has not been disabled.
    trcomp_builder_enabled = (
        trcomp_dedicated
        and trcomp_build_mode
        and args.framework in frameworks_to_build
        and train_or_inf_enabled
    )

    buildspec_file = get_buildspec_override() or args.buildspec

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
