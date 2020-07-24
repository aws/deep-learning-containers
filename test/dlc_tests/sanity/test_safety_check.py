import json
import logging
import os
import sys

from pkg_resources._vendor.packaging.specifiers import SpecifierSet
from pkg_resources._vendor.packaging.version import Version

import pytest

from invoke import run

from test.test_utils import CONTAINER_TESTS_PREFIX, is_dlc_cicd_context

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
            # for shipping pillow<=6.2.2 - the last available version for py2
            "py2": ['38449', '38450', '38451', '38452']
        },
        "inference": {
            # for shipping pillow<=6.2.2 - the last available version for py2
            "py2": ['38449', '38450', '38451', '38452']
        },
        "inference-eia": {
            # for shipping pillow<=6.2.2 - the last available version for py2
            "py2": ['38449', '38450', '38451', '38452']
        }
    },
    "mxnet": {
        "inference-eia": {
            # numpy<=1.16.0 -- This has to only be here while we publish MXNet 1.4.1 EI DLC v1.0
            "py2": ['36810',
                    # for shipping pillow<=6.2.2 - the last available version for py2
                    '38449', '38450', '38451', '38452'],
            "py3": ['36810']
        },
        "inference": {
            # for shipping pillow<=6.2.2 - the last available version for py2
            "py2": ['38449', '38450', '38451', '38452']
        },
        "training": {
            # for shipping pillow<=6.2.2 - the last available version for py2
            "py2": ['38449', '38450', '38451', '38452']
        }
    },
    "pytorch": {
        "training": {
            # astropy<3.0.1
            "py2": ['35810',
                    # for shipping pillow<=6.2.2 - the last available version for py2
                    '38449', '38450', '38451', '38452'],
            "py3": []
        }
    }
}


def _get_safety_ignore_list(image_uri):
    """
    Get a list of known safety check issue IDs to ignore, if specified in IGNORE_LISTS.
    :param image_uri:
    :return: <list> list of safety check IDs to ignore
    """
    framework = ("mxnet" if "mxnet" in image_uri else
                 "pytorch" if "pytorch" in image_uri else
                 "tensorflow")
    job_type = "training" if "training" in image_uri else "inference-eia" if "eia" in image_uri else "inference"
    python_version = "py2" if "py2" in image_uri else "py3"

    return IGNORE_SAFETY_IDS.get(framework, {}).get(job_type, {}).get(python_version)


@pytest.mark.canary("Run safety tests regularly on production images")
@pytest.mark.skipif(not is_dlc_cicd_context(), reason="Skipping test because it is not running in dlc cicd infra")
def test_safety(image):
    """
    Runs safety check on a container with the capability to ignore safety issues that cannot be fixed, and only raise
    error if an issue is fixable.
    """
    from dlc.safety_check import SafetyCheck
    safety_check = SafetyCheck()

    repo_name, image_tag = image.split('/')[-1].split(':')
    ignore_ids_list = _get_safety_ignore_list(image)
    sep = " -i "
    ignore_str = "" if not ignore_ids_list else f"{sep}{sep.join(ignore_ids_list)}"

    container_name = f"{repo_name}-{image_tag}-safety"
    docker_exec_cmd = f"docker exec -i {container_name}"
    test_file_path = os.path.join(CONTAINER_TESTS_PREFIX, "testSafety")
    # Add null entrypoint to ensure command exits immediately
    run(f"docker run -id "
        f"--name {container_name} "
        f"--mount type=bind,src=$(pwd)/container_tests,target=/test "
        f"--entrypoint='/bin/bash' "
        f"{image}", hide=True)
    try:
        run(f"{docker_exec_cmd} pip install safety yolk3k ", hide=True)
        json_str_safety_result = safety_check.run_safety_check_on_container(docker_exec_cmd)
        safety_result = json.loads(json_str_safety_result)
        for package, affected_versions, curr_version, _, vulnerability_id in safety_result:
            run_out = run(f"{docker_exec_cmd} yolk -M {package} -f version ", warn=True, hide=True)
            if run_out.return_code != 0:
                continue
            latest_version = run_out.stdout
            if Version(latest_version) in SpecifierSet(affected_versions):
                # Version(x) gives an object that can be easily compared with another version, or with a SpecifierSet.
                # Comparing two versions as a string has some edge cases which require us to write more code.
                # SpecifierSet(x) takes a version constraint, such as "<=4.5.6", ">1.2.3", or ">=1.2,<3.4.5", and
                # gives an object that can be easily compared against a Version object.
                # https://packaging.pypa.io/en/latest/specifiers/
                ignore_str += f" -i {vulnerability_id}"
        assert (safety_check.run_safety_check_with_ignore_list(docker_exec_cmd, ignore_str) == 0), \
            f"Safety test failed for {image}"
    finally:
        run(f"docker rm -f {container_name}", hide=True)
