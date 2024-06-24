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
    get_overall_history_path,
    get_folder_size_in_bytes,
    check_if_folder_contents_are_valid,
    verify_if_child_image_is_built_on_top_of_base_image,
    remove_repo_root_folder_path_from_the_given_path,
)
from codebuild_environment import get_cloned_folder_path
from context import Context

from src.constants import PATCHING_INFO_PATH_WITHIN_DLC

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
        image_uri,
        python_version=python_version,
        minimum_sev_threshold="UNDEFINED",
        allowlist_removal_enabled=False,
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
        ## Update key in case nginx exists
        container_setup_cmd = """bash -c 'VARIABLE=$(apt-key list 2>&1  |  { grep -c nginx || true; }) && if [ "$VARIABLE" != 0 ]; then echo "Nginx exists, thus upgrade" && curl https://nginx.org/keys/nginx_signing.key | gpg --dearmor | tee /usr/share/keyrings/nginx-archive-keyring.gpg >/dev/null && apt-key add /usr/share/keyrings/nginx-archive-keyring.gpg; fi && apt-get update'"""
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
    from test.test_utils import get_sha_of_an_image_from_ecr

    info = pre_push_image_object.info
    image_name = info.get("name")
    latest_released_image_uri = info.get("release_image_uri")

    run(f"docker pull {latest_released_image_uri}", hide=True)

    first_image_sha = extract_first_image_sha_using_patching_info_contents_of_given_image(
        image_uri=latest_released_image_uri
    )
    base_image_uri_for_patch_builds = latest_released_image_uri
    if first_image_sha:
        # In case the latest released image is an autopatched image first_image_sha will not be None
        # In those cases, pull the first image using the SHA and use that as base
        base_image_uri_for_patch_builds = pull_base_image_uri_for_patch_builds_and_get_the_tag(
            latest_released_image_uri=latest_released_image_uri, first_image_sha=first_image_sha
        )

    assert verify_if_child_image_is_built_on_top_of_base_image(
        base_image_uri=base_image_uri_for_patch_builds, child_image_uri=latest_released_image_uri
    ), f"Child image {latest_released_image_uri} is not built on {base_image_uri_for_patch_builds}"

    ecr_client = boto3.client("ecr", region_name=os.getenv("REGION"))
    latest_released_image_sha = get_sha_of_an_image_from_ecr(
        ecr_client=ecr_client, image_uri=latest_released_image_uri
    )

    current_patch_details_path = os.path.join(
        os.sep, download_path, base_image_uri_for_patch_builds.replace("/", "_").replace(":", "_")
    )
    if not os.path.exists(current_patch_details_path):
        run(f"mkdir {current_patch_details_path}", hide=True)

    complete_patching_info_dump_location = os.path.join(
        os.sep,
        get_cloned_folder_path(),
        f"""{base_image_uri_for_patch_builds.replace("/", "_").replace(":", "_")}_patch-dump""",
    )
    if not os.path.exists(complete_patching_info_dump_location):
        run(f"mkdir {complete_patching_info_dump_location}", hide=True)

    extract_patching_relevant_data_from_latest_released_image(
        image_uri=latest_released_image_uri,
        extraction_location=complete_patching_info_dump_location,
    )

    THREADS = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        get_dummy_boto_client()
        THREADS[f"trigger_language_patching-{base_image_uri_for_patch_builds}"] = executor.submit(
            trigger_language_patching,
            image_uri=base_image_uri_for_patch_builds,
            s3_downloaded_path=download_path,
            python_version=info.get("python_version"),
        )
        THREADS[
            f"trigger_enhanced_scan_patching-{base_image_uri_for_patch_builds}"
        ] = executor.submit(
            trigger_enhanced_scan_patching,
            image_uri=base_image_uri_for_patch_builds,
            patch_details_path=current_patch_details_path,
            python_version=info.get("python_version"),
        )
    FORMATTER.progress(THREADS)

    run(
        f"cp -r {current_patch_details_path}/. {complete_patching_info_dump_location}/patch-details-current"
    )

    pre_push_image_object.dockerfile = os.path.join(
        os.sep, get_cloned_folder_path(), "miscellaneous_dockerfiles", "Dockerfile.autopatch"
    )

    miscellaneous_scripts_path = os.path.join(
        os.sep, get_cloned_folder_path(), "miscellaneous_scripts"
    )

    verify_artifact_contents_for_patch_builds(
        patching_info_folder_path=complete_patching_info_dump_location,
        miscellaneous_scripts_path=miscellaneous_scripts_path,
    )

    pre_push_image_object.target = None

    info["extra_build_args"].update({"BASE_IMAGE_FOR_PATCH_BUILD": base_image_uri_for_patch_builds})
    info["extra_build_args"].update({"LATEST_RELEASED_IMAGE_SHA": latest_released_image_sha})
    info["extra_build_args"].update({"LATEST_RELEASED_IMAGE_URI": latest_released_image_uri})

    autopatch_artifacts = {
        "miscellaneous_scripts": {
            "source": miscellaneous_scripts_path,
            "target": "miscellaneous_scripts",
        },
        "dockerfile": {
            "source": pre_push_image_object.dockerfile,
            "target": "Dockerfile",
        },
        "patching-info": {
            "source": complete_patching_info_dump_location,
            "target": "patching-info",
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
        history_retrieval_command = (
            f"cat {PATCHING_INFO_PATH_WITHIN_DLC}/patch-details/overall_history.txt"
        )
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


def extract_first_image_sha_using_patching_info_contents_of_given_image(image_uri):
    """
    This method takes an image_uri and looks into the patching-info/patch-details-archive/first_image_sha.txt file of the
    image to get the SHA of the first manually released image. In case the SHA is not found, it returns None.

    :param image_uri: str, URI of the image
    :return: str, SHA of the first image or None in case the SHA is not found.
    """
    docker_run_cmd = f"docker run -id --entrypoint='/bin/bash' {image_uri} "
    container_id = run(f"{docker_run_cmd}").stdout.strip()
    docker_exec_cmd = f"docker exec -i {container_id}"
    sha_file_path = f"{PATCHING_INFO_PATH_WITHIN_DLC}/patch-details-archive/first_image_sha.txt"
    image_sha_extraction_cmd = (
        f"""bash -c "if [ -f {sha_file_path} ]; then cat {sha_file_path}; else echo ''; fi" """
    )
    docker_extraction_cmd = f"{docker_exec_cmd} {image_sha_extraction_cmd}"
    FORMATTER.print(f"[extract_sha_cmd] {docker_extraction_cmd}")
    result = run(docker_extraction_cmd, hide=True)
    first_image_sha = result.stdout.strip()
    return first_image_sha


def extract_patching_relevant_data_from_latest_released_image(image_uri, extraction_location):
    """
    Extracts the patching-info data from the given image-uri and dumps it in a folder that is then put into the new DLC.

    param image_uri: str, URI of the image
    """
    docker_run_cmd = f"docker run -v {extraction_location}:/dlc-extraction-folder -id --entrypoint='/bin/bash' {image_uri} "
    FORMATTER.print(f"[extract_relevant_data] docker_run_cmd : {docker_run_cmd}")
    container_id = run(f"{docker_run_cmd}", hide=True).stdout.strip()
    docker_exec_cmd = f"docker exec -i {container_id}"
    extraction_cmd = f"""bash -c "if [ -d {PATCHING_INFO_PATH_WITHIN_DLC} ] ; then cp -r {PATCHING_INFO_PATH_WITHIN_DLC}/. /dlc-extraction-folder ; fi" """
    FORMATTER.print(f"Extraction Command: {docker_exec_cmd} {extraction_cmd}")
    run(f"{docker_exec_cmd} {extraction_cmd}")


def verify_artifact_contents_for_patch_builds(
    patching_info_folder_path, miscellaneous_scripts_path
):
    """
    This method ensures that the folder being copied into the new DLC meets size requirements and that the contents within the
    folder are of specific type.

    :param patching_info_folder_path: str, Local path of the patching-info dump that will be onboarded to the new DLC.
    :param miscellaneous_scripts_path: str, Path of the miscellaneous_scripts folder that is present on Github.
    :return: boolean, Returns True in case the size and content conditions are met. Otherwise, returns False.
    """
    folder_size_in_bytes = get_folder_size_in_bytes(folder_path=patching_info_folder_path)
    folder_size_in_megabytes = folder_size_in_bytes / (1024.0 * 1024.0)
    assert (
        folder_size_in_megabytes <= 0.5
    ), f"Folder size for {patching_info_folder_path} is {folder_size_in_megabytes} MB which is more that 0.5 MB."

    assert check_if_folder_contents_are_valid(
        folder_path=patching_info_folder_path,
        hidden_files_allowed=False,
        subdirs_allowed=True,
        only_acceptable_file_types=[".sh", ".txt", ".json"],
    ), f"Root folder {patching_info_folder_path} contents are invalid"

    patch_details_current_folder_path = os.path.join(
        os.sep, patching_info_folder_path, "patch-details-current"
    )
    if os.path.exists(patch_details_current_folder_path):
        assert check_if_folder_contents_are_valid(
            folder_path=patch_details_current_folder_path,
            hidden_files_allowed=False,
            subdirs_allowed=False,
            only_acceptable_file_types=[".sh", ".txt", ".json"],
        ), f"{patch_details_current_folder_path} contents are invalid"

    patch_details_folder_path = os.path.join(os.sep, patching_info_folder_path, "patch-details")
    if os.path.exists(patch_details_folder_path):
        assert check_if_folder_contents_are_valid(
            folder_path=patch_details_folder_path,
            hidden_files_allowed=False,
            subdirs_allowed=False,
            only_acceptable_file_types=[".sh", ".txt", ".json"],
        ), f"{patch_details_folder_path} contents are invalid"

    folder_size_in_bytes = get_folder_size_in_bytes(folder_path=miscellaneous_scripts_path)
    folder_size_in_megabytes = folder_size_in_bytes / (1024.0 * 1024.0)
    assert (
        folder_size_in_megabytes <= 0.5
    ), f"Folder size for {miscellaneous_scripts_path} is {folder_size_in_megabytes} MB which is more that 0.5 MB."


def pull_base_image_uri_for_patch_builds_and_get_the_tag(
    latest_released_image_uri, first_image_sha
):
    """
    Pulls Base image from ECR using the SHA and LOCALLY tags it using the tag of the latest released image uri appended with -FIMG.

    :param latest_released_image_uri: str, Image URI of the latest released image.
    :param first_image_sha: str, SHA of the first non-autopatched image that would be used as base.
    :return: str, Base Image URI that would be used for building the new image.
    """
    FORMATTER.print(
        f"Latest released image is different from the first image that has sha: {first_image_sha}"
    )
    prod_repo = latest_released_image_uri.split(":")[0]
    first_image_uri_with_sha = f"{prod_repo}@{first_image_sha}"
    first_image_uri = f"{latest_released_image_uri}-FIMG"
    run(f"docker pull {first_image_uri_with_sha}")
    run(f"docker tag {first_image_uri_with_sha} {first_image_uri}")
    FORMATTER.print(f"First Image URI tagged as: {first_image_uri}")
    return first_image_uri
