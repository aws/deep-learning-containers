from pkg_resources._vendor.packaging.specifiers import SpecifierSet
from pkg_resources._vendor.packaging.version import Version

from invoke import run
import json
import pytest

from test.dlc_tests.sanity import get_safety_ignore_list


@pytest.mark.usefixtures("pull_images")
def test_safety(image):
    """
    Runs safety check on a container with the capability to ignore safety issues that cannot be fixed, and only raise
    error if an issue is fixable.
    """
    repo_name, image_tag = image.split('/')[-1].split(':')
    python_version = "py2" if "py2" in image_tag else "py3"
    ignore_ids_list = get_safety_ignore_list(repo_name, python_version)
    ignore_str = "" if ignore_ids_list is None else " ".join(ignore_ids_list)

    container_name = f"{repo_name}-safety"
    docker_exec_cmd = f"docker exec -i {container_name}"
    # Add null entrypoint to ensure command exits immediately
    run(f"docker run -id "
        f"--name {container_name} "
        f"--mount type=bind,src=$(pwd)/container_tests,target=/test "
        f"--entrypoint='/bin/bash' "
        f"{image}")
    try:
        run(f"{docker_exec_cmd} pip install -qq safety yolk3k ")
        run_out = run(f"{docker_exec_cmd} safety check --json 2>&1 ", warn=True)
        json_str_safety_result = run_out.stdout
        safety_result = json.loads(json_str_safety_result)
        for package, affected_versions, curr_version, _, vulnerability_id in safety_result:
            run_out = run(f"{docker_exec_cmd} yolk -M {package} -f version ", warn=True)
            if run_out.return_code != 0:
                continue
            latest_version = run_out.stdout
            if Version(latest_version) in SpecifierSet(affected_versions):
                # Version(x) gives an object that can be easily compared with another version, or with a SpecifierSet.
                # Comparing two versions as a string has some edge cases which require us to write more code.
                # SpecifierSet(x) takes a version constraint, such as "<=4.5.6", ">1.2.3", or ">=1.2,<3.4.5", and
                # gives an object that can be easily compared against a Version object.
                # https://packaging.pypa.io/en/latest/specifiers/
                print(f"Latest version {package}=={latest_version} is within affected versions {affected_versions}")
                print(f"Ignoring safety vulnerability for package {package}=={curr_version} "
                      f"because both current and latest versions are vulnerable.")
                ignore_str += f" {vulnerability_id}"
            else:
                print('Package {}=={} can be updated to version {}'.format(package, curr_version, latest_version))
                if vulnerability_id in ignore_str:
                    print('Ignoring vulnerability ID {} as it is a known issue.'.format(vulnerability_id))

        run(f"{docker_exec_cmd} chmod +x /test/bin/testSafety")
        run(f"{docker_exec_cmd} /test/bin/testSafety {ignore_str} ")
    finally:
        run(f"docker rm -f {container_name}")
