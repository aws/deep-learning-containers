import json
import logging
import os
import sys

from packaging.specifiers import SpecifierSet
from packaging.version import Version

import pytest
import requests

from invoke import run

from test.test_utils import (
    CONTAINER_TESTS_PREFIX,
    is_dlc_cicd_context,
    is_canary_context,
    is_mainline_context,
    is_time_for_canary_safety_scan,
    is_safety_test_context,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))

# List of safety check vulnerability IDs to ignore. To get the ID, run safety check on the container,
# and copy paste the ID given by the safety check for a package.
# Note:- 1. This ONLY needs to be done if a package version exists that resolves this safety issue, but that version
#           cannot be used because of incompatibilities.
#        2. Ensure that IGNORE_SAFETY_IDS is always as small/empty as possible.
IGNORE_SAFETY_IDS = {
    "tensorflow": {
        "training": {
            "py2": [
                # for shipping pillow<=6.2.2 - the last available version for py2
                "38449",
                "38450",
                "38451",
                "38452",
                # for shipping pycrypto<=2.6.1 - the last available version for py2
                "35015",
            ],
            "py3": [],
        },
        "inference": {
            "py2": [
                # for shipping pillow<=6.2.2 - the last available version for py2
                "38449",
                "38450",
                "38451",
                "38452",
            ],
            "py3": [],
        },
        "inference-eia": {
            "py2": [
                # for shipping pillow<=6.2.2 - the last available version for py2
                "38449",
                "38450",
                "38451",
                "38452",
            ],
            "py3": [],
        },
        "inference-neuron": {
            "py3": [
                # TF 1.15.5 is on par with TF 2.0.4, 2.1.3, 2.2.2, 2.3.2 in security patches
                "39409",
                "39408",
                "39407",
                "39406",
            ],
        },
    },
    "mxnet": {
        "inference-eia": {
            "py2": [
                # numpy<=1.16.0 -- This has to only be here while we publish MXNet 1.4.1 EI DLC v1.0
                "36810",
                # for shipping pillow<=6.2.2 - the last available version for py2
                "38449",
                "38450",
                "38451",
                "38452",
            ],
            "py3": [],
        },
        "inference": {
            "py2": [
                # for shipping pillow<=6.2.2 - the last available version for py2
                "38449",
                "38450",
                "38451",
                "38452",
            ],
            "py3": [],
        },
        "training": {
            "py2": [
                # for shipping pillow<=6.2.2 - the last available version for py2
                "38449",
                "38450",
                "38451",
                "38452",
            ],
            "py3": [],
        },
        "inference-neuron": {
            "py3": [
                # for shipping tensorflow 1.15.5
                "40673",
                "40675",
                "40676",
                "40794",
                "40795",
                "40796",
            ]
        },
    },
    "pytorch": {
        "training": {
            "py2": [
                # astropy<3.0.1
                "35810",
                # for shipping pillow<=6.2.2 - the last available version for py2
                "38449",
                "38450",
                "38451",
                "38452",
            ],
            "py3": [],
        },
        "inference": {"py3": []},
        "inference-eia": {"py3": []},
        "inference-neuron": {
            "py3": [
                # 39409, 39408, 39407, 39406: TF 1.15.5 is on par with TF 2.0.4, 2.1.3, 2.2.2, 2.3.2 in security patches
                "39409",
                "39408",
                "39407",
                "39406",
                # 40794, 40795: TF 1.15.5 is the last available version of TF 1
                "40794",
                "40795",
            ]
        },
    },
}


def _get_safety_ignore_list(image_uri):
    """
    Get a list of known safety check issue IDs to ignore, if specified in IGNORE_LISTS.
    :param image_uri:
    :return: <list> list of safety check IDs to ignore
    """
    framework = "mxnet" if "mxnet" in image_uri else "pytorch" if "pytorch" in image_uri else "tensorflow"
    job_type = (
        "training"
        if "training" in image_uri
        else "inference-eia"
        if "eia" in image_uri
        else "inference-neuron"
        if "neuron" in image_uri
        else "inference"
    )
    python_version = "py2" if "py2" in image_uri else "py3"

    return IGNORE_SAFETY_IDS.get(framework, {}).get(job_type, {}).get(python_version, [])


def _get_latest_package_version(package):
    """
    Get the latest package version available on pypi for a package.
    It is retried multiple times in case there are transient failures in executing the command.

    :param package: str Name of the package whose latest version must be retrieved
    :return: tuple(command_success: bool, latest_version_value: str)
    """
    pypi_package_info = requests.get(f"https://pypi.org/pypi/{package}/json")
    data = json.loads(pypi_package_info.text)
    versions = data["releases"].keys()
    return str(max(Version(v) for v in versions))


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("N/A")
@pytest.mark.canary("Run safety tests regularly on production images")
@pytest.mark.skipif(not is_dlc_cicd_context(), reason="Skipping test because it is not running in dlc cicd infra")
@pytest.mark.skipif(
    not (is_mainline_context() or (is_canary_context() and is_time_for_canary_safety_scan())),
    reason=(
        "Skipping the test to decrease the number of calls to the Safety Check DB. "
        "Test will be executed in the 'mainline' pipeline and canaries pipeline."
    )
)
def test_safety(image):
    """
    Runs safety check on a container with the capability to ignore safety issues that cannot be fixed, and only raise
    error if an issue is fixable.
    """
    from dlc.safety_check import SafetyCheck

    safety_check = SafetyCheck()

    repo_name, image_tag = image.split("/")[-1].split(":")
    ignore_ids_list = _get_safety_ignore_list(image)
    sep = " -i "
    ignore_str = "" if not ignore_ids_list else f"{sep}{sep.join(ignore_ids_list)}"

    container_name = f"{repo_name}-{image_tag}-safety"
    docker_exec_cmd = f"docker exec -i {container_name}"
    test_file_path = os.path.join(CONTAINER_TESTS_PREFIX, "testSafety")
    # Add null entrypoint to ensure command exits immediately
    run(
        f"docker run -id "
        f"--name {container_name} "
        f"--mount type=bind,src=$(pwd)/container_tests,target=/test "
        f"--entrypoint='/bin/bash' "
        f"{image}",
        hide=True,
    )
    try:
        run(f"{docker_exec_cmd} pip install safety yolk3k ", hide=True)
        json_str_safety_result = safety_check.run_safety_check_on_container(docker_exec_cmd)
        safety_result = json.loads(json_str_safety_result)
        for vulnerability in safety_result:
            package, affected_versions, curr_version, _, vulnerability_id = vulnerability[:5]
            # Get the latest version of the package with vulnerability
            latest_version = _get_latest_package_version(package)
            # If the latest version of the package is also affected, ignore this vulnerability
            if Version(latest_version) in SpecifierSet(affected_versions):
                # Version(x) gives an object that can be easily compared with another version, or with a SpecifierSet.
                # Comparing two versions as a string has some edge cases which require us to write more code.
                # SpecifierSet(x) takes a version constraint, such as "<=4.5.6", ">1.2.3", or ">=1.2,<3.4.5", and
                # gives an object that can be easily compared against a Version object.
                # https://packaging.pypa.io/en/latest/specifiers/
                ignore_str += f" -i {vulnerability_id}"
        assert (
            safety_check.run_safety_check_with_ignore_list(docker_exec_cmd, ignore_str) == 0
        ), f"Safety test failed for {image}"
    finally:
        run(f"docker rm -f {container_name}", hide=True)
