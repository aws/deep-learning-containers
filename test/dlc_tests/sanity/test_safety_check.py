import json
import logging
import os
from pkg_resources._vendor.packaging.specifiers import SpecifierSet
from pkg_resources._vendor.packaging.version import Version
import sys

from invoke import run

from test.test_utils import CONTAINER_TESTS_PREFIX


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))


# List of safety check vulnerability IDs to ignore. To get the ID, run safety check on the container,
# and copy paste the ID given by the safety check for a package.
# Note:- 1. This ONLY needs to be done if a package version exists that resolves this safety issue, but that version
#           cannot be used because of incompatibilities.
#        2. Ensure that IGNORE_SAFETY_IDS is always as small/empty as possible.
IGNORE_SAFETY_IDS = {
    "mxnet": {
        "inference-eia": {
            # numpy<=1.16.0 -- This has to only be here while we publish MXNet 1.4.1 EI DLC v1.0
            "py2": ['36810'],
            "py3": ['36810']
        }
    },
    "pytorch": {
        "training": {
            # astropy<3.0.1
            "py2": ['35810'],
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


def test_safety(image):
    """
    Runs safety check on a container with the capability to ignore safety issues that cannot be fixed, and only raise
    error if an issue is fixable.
    """
    repo_name, image_tag = image.split('/')[-1].split(':')
    ignore_ids_list = _get_safety_ignore_list(image)
    ignore_str = "" if not ignore_ids_list else " ".join(ignore_ids_list)

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
        run_out = run(f"{docker_exec_cmd} safety check --json ", warn=True, hide=True)
        json_str_safety_result = run_out.stdout
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
                ignore_str += f" {vulnerability_id}"

        run(f"{docker_exec_cmd} chmod +x {test_file_path}", hide=True)
        output = run(f"{docker_exec_cmd} {test_file_path} {ignore_str} ", warn=True)
        LOGGER.info(f"{test_file_path} log for {image}\n{output.stdout}")
        assert output.return_code == 0, f"Safety test failed for {image}\n{output.stdout}"
    finally:
        run(f"docker rm -f {container_name}", hide=True)
