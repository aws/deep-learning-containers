import os
import boto3
import concurrent.futures
import json

import constants

from output import OutputFormatter
from invoke import run
from image import DockerImage

from utils import (
    get_dummy_boto_client,
    upload_data_to_pr_creation_s3_bucket,
    get_core_packages_path,
    get_unique_s3_path_for_uploading_data_to_pr_creation_bucket,
    get_git_path_for_overall_history,
    get_overall_history_path,
    remove_repo_root_folder_path_from_the_given_path,
)
from codebuild_environment import get_cloned_folder_path
from context import Context

FORMATTER = OutputFormatter(constants.PADDING)


def trigger_language_patching(image_uri, s3_downloaded_path, python_version=None):
    """
    This method initiates the processing for language packages. It creates a patch dump specific for each container that has the
    patched package details.

    :param image_uri: str, image_uri
    :param s3_downloaded_path: str, Path where the relevant data is downloaded
    :param python_version: str, python_version
    :return: str, Returns constants.SUCCESS to allow the multi-threaded caller to know that the method has succeeded.
    """
    patch_dlc_folder_mount = os.path.join(os.sep, s3_downloaded_path)
    dlc_repo_folder_mount = os.path.join(os.sep, get_cloned_folder_path())
    docker_run_cmd = f"docker run -v {patch_dlc_folder_mount}:/patch-dlc -v {dlc_repo_folder_mount}:/deep-learning-containers -id --entrypoint='/bin/bash' {image_uri} "
    FORMATTER.print(f"[trigger_language] docker_run_cmd : {docker_run_cmd}")
    container_id = run(f"{docker_run_cmd}", hide=True).stdout.strip()
    docker_exec_cmd = f"docker exec -i {container_id}"

    try:
        absolute_core_package_path = get_core_packages_path(image_uri, python_version)
        core_package_path_within_dlc_repo = ""
        if os.path.exists(absolute_core_package_path):
            core_package_path_within_dlc_repo = absolute_core_package_path.replace(
                dlc_repo_folder_mount, os.path.join(os.sep, "deep-learning-containers")
            )
        script_run_cmd = f"bash /patch-dlc/script.sh {image_uri}"
        if core_package_path_within_dlc_repo:
            script_run_cmd = f"{script_run_cmd} {core_package_path_within_dlc_repo}"
        FORMATTER.print(f"[trigger_language] script_run_cmd : {script_run_cmd}")
        result = run(f"{docker_exec_cmd} {script_run_cmd}", hide=True)
        new_cmd = result.stdout.strip().split("\n")[-1]
        print(f"For {image_uri} => {new_cmd}")
    finally:
        run(f"docker rm -f {container_id}", hide=True, warn=True)

    return constants.SUCCESS


def get_impacted_os_packages(image_uri, python_version=None):
    """
    Runs Enhanced Scan on the image and returns the impacted OS packages.

    :param image_uri: str, image_uri
    :param python_version: str, python_version
    :return: set, impacted OS packages
    """
    # Lazy import is done over here to prevent circular dependencies. In general, it is a good practice to have lazy imports.
    from test.dlc_tests.sanity.test_ecr_scan import (
        helper_function_for_leftover_vulnerabilities_from_enhanced_scanning,
    )

    (
        remaining_vulnerabilities,
        _,
    ) = helper_function_for_leftover_vulnerabilities_from_enhanced_scanning(
        image_uri, python_version=python_version
    )
    impacted_packages = set()
    if remaining_vulnerabilities:
        for package_name, package_cve_list in remaining_vulnerabilities.vulnerability_list.items():
            for cve in package_cve_list:
                if cve.package_details.package_manager == "OS":
                    impacted_packages.add(package_name)
    return impacted_packages


def trigger_enhanced_scan_patching(image_uri, patch_details_path, python_version=None):
    """
    This method initiates the processing for enhanced scan patching of the images. It triggers the enhanced scanning for the
    image and then gets the result to find the impacted packages. These impacted packages are then sent to the extract_apt_patch_data.py
    script that executes in the GENERATE mode to get the list of all the impacted packages that can be upgraded and their version in the
    released image. This data is then used to create the apt upgrade command and is dumped in the form of install_script_os.sh.

    Note: We need to do a targeted package upgrade to upgrade the impacted packages to esnure that the image does not inflate.

    :param image_uri: str, image_uri
    :param s3_downloaded_path: str, Path where the relevant data is downloaded
    :param python_version: str, python_version
    :return: str, Returns constants.SUCCESS to allow the multi-threaded caller to know that the method has succeeded.
    """
    impacted_packages = get_impacted_os_packages(image_uri=image_uri, python_version=python_version)
    dlc_repo_folder_mount = os.path.join(os.sep, get_cloned_folder_path())
    image_specific_patch_folder = os.path.join(
        os.sep, patch_details_path
    )  # image_specific_patch_folder
    docker_run_cmd = f"docker run -v {dlc_repo_folder_mount}:/deep-learning-containers -v {image_specific_patch_folder}:/image-specific-patch-folder  -id --entrypoint='/bin/bash' {image_uri} "
    container_id = run(f"{docker_run_cmd}", hide=True).stdout.strip()
    try:
        docker_exec_cmd = f"docker exec -i {container_id}"
        container_setup_cmd = "apt-get update"
        run(f"{docker_exec_cmd} {container_setup_cmd}", hide=True)
        save_file_name = "os_summary.json"
        script_run_cmd = f"""python /deep-learning-containers/miscellaneous_scripts/extract_apt_patch_data.py --save-result-path /image-specific-patch-folder/{save_file_name} --mode_type generate"""
        if impacted_packages:
            script_run_cmd = (
                f"""{script_run_cmd} --impacted-packages {",".join(impacted_packages)}"""
            )
        run(f"{docker_exec_cmd} {script_run_cmd}", hide=True)
        with open(os.path.join(os.sep, patch_details_path, save_file_name), "r") as readfile:
            saved_json_data = json.load(readfile)
        print(f"For {image_uri} => {saved_json_data}")
        patch_package_dict = saved_json_data["patch_package_dict"]
        patch_package_list = list(patch_package_dict.keys())
        echo_cmd = """ echo "echo N/A" """
        file_concat_cmd = f"tee {patch_details_path}/install_script_os.sh"
        if patch_package_list:
            echo_cmd = f"""echo  "apt-get update && apt-get install -y --only-upgrade {" ".join(patch_package_list)}" """
        if os.getenv("IS_CODEBUILD_IMAGE") is None:
            file_concat_cmd = f"sudo {file_concat_cmd}"
        complete_command = f"{echo_cmd} | {file_concat_cmd}"
        print(f"For {image_uri} => {complete_command}")
        run(complete_command, hide=True)
    finally:
        run(f"docker rm -f {container_id}", hide=True, warn=True)
    return constants.SUCCESS


def conduct_autopatch_build_setup(pre_push_image_object: DockerImage, download_path: str):
    """
    This method conducts the setup for the AutoPatch builds. It pulls the already released image and then triggers the autopatching
    procedures on the image to get the packages that need to be modified. Thereafter, it modifies pre_push_image_object to make changes
    to the original build process such that it starts to utilize miscellaneous_dockerfiles/Dockerfile.autopatch Dockerfile for building the image.

    :param pre_push_image_object: Object of type DockerImage, The original DockerImage object that gets modified by this method.
    :param download_path: str, Path of the file where the relevant scripts have alread been downloaded.
    :return: str, Returns constants.SUCCESS to allow the multi-threaded caller to know that the method has succeeded.
    """
    info = pre_push_image_object.info
    image_name = info.get("name")
    released_image_uri = info.get("release_image_uri")

    from test.test_utils import get_sha_of_an_image_from_ecr

    patch_details_path = os.path.join(
        os.sep, download_path, released_image_uri.replace("/", "_").replace(":", "_")
    )
    if not os.path.exists(patch_details_path):
        run(f"mkdir {patch_details_path}", hide=True)

    run(f"docker pull {released_image_uri}", hide=True)

    THREADS = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        get_dummy_boto_client()
        THREADS[f"trigger_language_patching-{released_image_uri}"] = executor.submit(
            trigger_language_patching,
            image_uri=released_image_uri,
            s3_downloaded_path=download_path,
            python_version=info.get("python_version"),
        )
        THREADS[f"trigger_enhanced_scan_patching-{released_image_uri}"] = executor.submit(
            trigger_enhanced_scan_patching,
            image_uri=released_image_uri,
            patch_details_path=patch_details_path,
            python_version=info.get("python_version"),
        )
    FORMATTER.progress(THREADS)

    pre_push_image_object.dockerfile = os.path.join(
        os.sep, get_cloned_folder_path(), "miscellaneous_dockerfiles", "Dockerfile.autopatch"
    )

    miscellaneous_scripts_path = os.path.join(
        os.sep, get_cloned_folder_path(), "miscellaneous_scripts"
    )

    pre_push_image_object.target = None
    ecr_client = boto3.client("ecr", region_name=os.getenv("REGION"))
    released_image_sha = get_sha_of_an_image_from_ecr(
        ecr_client=ecr_client, image_uri=released_image_uri
    )

    info["extra_build_args"].update({"RELEASED_IMAGE": released_image_uri})
    info["extra_build_args"].update({"RELEASED_IMAGE_SHA": released_image_sha})

    autopatch_artifacts = {
        "miscellaneous_scripts": {
            "source": miscellaneous_scripts_path,
            "target": "miscellaneous_scripts",
        },
        "dockerfile": {
            "source": pre_push_image_object.dockerfile,
            "target": "Dockerfile",
        },
        "patch-details": {
            "source": patch_details_path,
            "target": "patch-details",
        },
    }
    context = Context(
        autopatch_artifacts,
        f"build/{image_name}.tar.gz",
        os.path.join(os.sep, get_cloned_folder_path(), "src"),
    )
    pre_push_image_object.info = info
    pre_push_image_object.context = context
    return constants.SUCCESS


def initiate_multithreaded_autopatch_prep(PRE_PUSH_STAGE_IMAGES, make_dummy_boto_client=False):
    """
    This method executes a few pre-requisites before initiating multi-threading for conduct_autopatch_build_setup

    :param PRE_PUSH_STAGE_IMAGES: List[DockerImage], Consists the list of all the pre_push_image_objects that would be eventually
                                  modified by the conduct_autopatch_build_setup method.
    :param make_dummy_boto_client: bool, specifies if a dummy client should be declared or not.
    """
    run(
        f"""pip install -r {os.path.join(os.sep, get_cloned_folder_path(), "test", "requirements.txt")}""",
        hide=True,
    )
    folder_path_outside_clone = os.path.join(os.sep, *get_cloned_folder_path().split(os.sep)[:-1])
    download_path = os.path.join(os.sep, folder_path_outside_clone, "patch-dlc")
    if not os.path.exists(download_path):
        run(f"aws s3 cp s3://patch-dlc {download_path} --recursive", hide=True)
    run(f"bash {download_path}/preprocessing_script.sh {download_path}", hide=True)

    THREADS = {}
    # In the context of the ThreadPoolExecutor each instance of image.build submitted
    # to it is executed concurrently in a separate thread.
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        #### TODO: Remove this entire if block when get_dummy_boto_client is removed ####
        if make_dummy_boto_client:
            get_dummy_boto_client()
        for pre_push_image_object in PRE_PUSH_STAGE_IMAGES:
            THREADS[pre_push_image_object.name] = executor.submit(
                conduct_autopatch_build_setup, pre_push_image_object, download_path
            )
    # the FORMATTER.progress(THREADS) function call also waits until all threads have completed
    FORMATTER.progress(THREADS)


def retrive_autopatched_image_history_and_upload_to_s3(image_uri):
    """
    In this method , we extract the overall_history.txt file from the image and then upload it to the s3 bucket.
    The extracted data is also returned by this function.

    :param image_uri: str, Image URI
    :return: str, overall_history.txt data
    """
    docker_run_cmd = f"docker run -id --entrypoint='/bin/bash' {image_uri} "
    container_id = run(f"{docker_run_cmd}", hide=True).stdout.strip()
    try:
        docker_exec_cmd = f"docker exec -i {container_id}"
        history_retrieval_command = f"cat /opt/aws/dlc/patch-details/overall_history.txt"
        data = run(f"{docker_exec_cmd} {history_retrieval_command}", hide=True)
        upload_path = get_unique_s3_path_for_uploading_data_to_pr_creation_bucket(
            image_uri=image_uri.replace("-multistage-common", ""), file_name="overall_history.txt"
        )
        tag_set = [
            {
                "Key": "upload_path",
                "Value": remove_repo_root_folder_path_from_the_given_path(
                    given_path=get_overall_history_path(
                        image_uri=image_uri.replace("-multistage-common", "")
                    )
                ),
            },
            {"Key": "image_uri", "Value": image_uri.replace("-multistage-common", "")},
        ]
        upload_data_to_pr_creation_s3_bucket(
            upload_data=data.stdout, s3_filepath=upload_path, tag_set=tag_set
        )
    finally:
        run(f"docker rm -f {container_id}", hide=True, warn=True)
    return data.stdout
