from pkg_resources._vendor.packaging.specifiers import SpecifierSet
from pkg_resources._vendor.packaging.version import Version

from invoke import run
import json
import pytest


# List of safety check vulnerability IDs to ignore. To get the ID, run safety check on the container,
# and copy paste the ID given by the safety check for a package.
# Note:- 1. This ONLY needs to be done if a package version exists that resolves this safety issue, but that version
#           cannot be used because of incompatibilities.
#        2. Ensure that IGNORE_SAFETY_IDS is always as small/empty as possible.
IGNORE_SAFETY_IDS = {
    "mxnet-inference-eia": {
        # numpy<=1.16.0 -- This has to only be here while we publish MXNet 1.4.1 EI DLC v1.0
        "py2": ['36810'],
        "py3": ['36810']
    },
    "pytorch-training": {
        # astropy<3.0.1
        "py2": ['35810'],
        "py3": []
    }
}


def _get_safety_ignore_list(repo_name, python_version):
    """
    Get a list of known safety check issue IDs to ignore, if specified in IGNORE_LISTS.
    :param repo_name:
    :param python_version:
    :return: <list> list of safety check IDs to ignore
    """
    return IGNORE_SAFETY_IDS.get(repo_name, {}).get(python_version)


def test_safety(image):
    """
    Runs safety check on a container with the capability to ignore safety issues that cannot be fixed, and only raise
    error if an issue is fixable.
    """
    repo_name, image_tag = image.split('/')[-1].split(':')
    python_version = "py2" if "py2" in image_tag else "py3"
    ignore_ids_list = _get_safety_ignore_list(repo_name, python_version)
    ignore_str = "" if not ignore_ids_list else " ".join(ignore_ids_list)

    container_name = f"{repo_name}-{image_tag}-safety"
    docker_exec_cmd = f"docker exec -i {container_name}"
    # Add null entrypoint to ensure command exits immediately
    run(f"docker run -id "
        f"--name {container_name} "
        f"--mount type=bind,src=$(pwd)/container_tests,target=/test "
        f"--entrypoint='/bin/bash' "
        f"{image}", echo=True)
    try:
        run(f"{docker_exec_cmd} pip install safety yolk3k ", hide='out')
        run_out = run(f"{docker_exec_cmd} safety check --json ", warn=True, hide='out')
        json_str_safety_result = run_out.stdout
        safety_result = json.loads(json_str_safety_result)
        for package, affected_versions, curr_version, _, vulnerability_id in safety_result:
            run_out = run(f"{docker_exec_cmd} yolk -M {package} -f version ", warn=True, hide='out')
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

        run(f"{docker_exec_cmd} chmod +x /test/bin/testSafety", echo=True)
        run(f"{docker_exec_cmd} /test/bin/testSafety {ignore_str} ", echo=True)
    finally:
        run(f"docker rm -f {container_name}", echo=True)
