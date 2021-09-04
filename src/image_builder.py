"""
Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"). You
may not use this file except in compliance with the License. A copy of
the License is located at

    http://aws.amazon.com/apache2.0/

or in the "license" file accompanying this file. This file is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
"""

import concurrent.futures
import datetime
import os

from copy import deepcopy

import constants
import utils
import boto3
import itertools

from context import Context
from metrics import Metrics
from image import DockerImage
from common_stage_image import CommonStageImage
from buildspec import Buildspec
from output import OutputFormatter
from config import parse_dlc_developer_configs

FORMATTER = OutputFormatter(constants.PADDING)
build_context = os.getenv("BUILD_CONTEXT")

def _find_image_object(images_list, image_name):
    """
    Find and return an image object from images_list with a name that matches image_name
    :param images_list: <list> List of <DockerImage> objects
    :param image_name: <str> Name of image as per buildspec
    :return: <DockerImage> Object with image_name as "name" attribute
    """
    ret_image_object = None
    for image in images_list:
        if image.name == image_name:
            ret_image_object = image
            break
    return ret_image_object


# TODO: Abstract away to ImageBuilder class
def image_builder(buildspec):

    BUILDSPEC = Buildspec()
    BUILDSPEC.load(buildspec)
    PRE_PUSH_STAGE_IMAGES = []
    COMMON_STAGE_IMAGES = []

    if "huggingface" in str(BUILDSPEC["framework"]):
        os.system("echo login into public ECR")
        os.system("aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com")

    for image_name, image_config in BUILDSPEC["images"].items():
        ARTIFACTS = deepcopy(BUILDSPEC["context"]) if BUILDSPEC.get("context") else {}

        extra_build_args = {}
        labels = {}
        enable_datetime_tag = parse_dlc_developer_configs("build", "datetime_tag")

        if image_config.get("version") is not None:
            if BUILDSPEC["version"] != image_config.get("version"):
                continue

        if image_config.get("context") is not None:
            ARTIFACTS.update(image_config["context"])

        image_tag = (
            tag_image_with_pr_number(image_config["tag"])
            if build_context == "PR"
            else image_config["tag"]
        )
        if enable_datetime_tag or build_context != "PR":
            image_tag = tag_image_with_datetime(image_tag)
        image_repo_uri = (
            image_config["repository"]
            if build_context == "PR"
            else modify_repository_name_for_context(str(image_config["repository"]), build_context)
        )
        base_image_uri = None
        if image_config.get("base_image_name") is not None:
            base_image_object = _find_image_object(PRE_PUSH_STAGE_IMAGES, image_config["base_image_name"])
            base_image_uri = base_image_object.ecr_url

        if image_config.get("download_artifacts") is not None:
            for artifact_name, artifact in image_config.get("download_artifacts").items():
                type = artifact["type"]
                uri = artifact["URI"]
                var = artifact["VAR_IN_DOCKERFILE"]

                try:
                    file_name = utils.download_file(uri, type).strip()
                except ValueError:
                    FORMATTER.print(f"Artifact download failed: {uri} of type {type}.")

                ARTIFACTS.update({
                    f"{artifact_name}": {
                        "source": f"{os.path.join(os.sep, os.path.abspath(os.getcwd()), file_name)}",
                        "target": file_name
                    }
                })

                extra_build_args[var] = file_name
                labels[var] = file_name
                labels[f"{var}_URI"] = uri

        if str(BUILDSPEC["framework"]).startswith("huggingface"):
            if "transformers_version" in image_config:
                extra_build_args["TRANSFORMERS_VERSION"] = image_config.get("transformers_version")
            else:
                raise KeyError(f"HuggingFace buildspec.yml must contain 'transformers_version' field for each image")
            if "datasets_version" in image_config:
                extra_build_args["DATASETS_VERSION"] = image_config.get("datasets_version")
            elif str(image_config["image_type"]) == "training":
                raise KeyError(f"HuggingFace buildspec.yml must contain 'datasets_version' field for each image")

        ARTIFACTS.update(
            {
                "dockerfile": {
                    "source": image_config["docker_file"],
                    "target": "Dockerfile",
                }
            }
        )

        context = Context(ARTIFACTS, f"build/{image_name}.tar.gz", image_config["root"])

        if "labels" in image_config:
            labels.update(image_config.get("labels"))

        """
        Override parameters from parent in child.
        """

        info = {
            "account_id": str(BUILDSPEC["account_id"]),
            "region": str(BUILDSPEC["region"]),
            "framework": str(BUILDSPEC["framework"]),
            "version": str(BUILDSPEC["version"]),
            "root": str(image_config["root"]),
            "name": str(image_name),
            "device_type": str(image_config["device_type"]),
            "python_version": str(image_config["python_version"]),
            "image_type": str(image_config["image_type"]),
            "image_size_baseline": int(image_config["image_size_baseline"]),
            "base_image_uri": base_image_uri,
            "labels": labels,
            "extra_build_args": extra_build_args
        }
        
        # Create pre_push stage docker object
        pre_push_stage_image_object = DockerImage(
            info=info,
            dockerfile=image_config["docker_file"],
            repository=image_repo_uri,
            tag=image_tag,
            to_build=image_config["build"],
            stage=constants.PRE_PUSH_STAGE,
            context=context,
        )

        ##### Create Common stage docker object #####
        # If for a pre_push stage image we create a common stage image, then we do not push the pre_push stage image
        # to the repository. Instead, we just push its common stage image to the repository. Therefore,
        # inside function get_common_stage_image_object we make pre_push_stage_image_object non pushable.
        common_stage_image_object = get_common_stage_image_object(pre_push_stage_image_object)

        PRE_PUSH_STAGE_IMAGES.append(pre_push_stage_image_object)
        COMMON_STAGE_IMAGES.append(common_stage_image_object)
        FORMATTER.separator()

    FORMATTER.banner("DLC")

    # Standard images must be built before example images
    # Example images will use standard images as base
    # Common images must be built at the end as they will consume respective standard and example images
    standard_images = [image for image in PRE_PUSH_STAGE_IMAGES if "example" not in image.name.lower()]
    example_images = [image for image in PRE_PUSH_STAGE_IMAGES if "example" in image.name.lower()]
    common_stage_images = [image for image in COMMON_STAGE_IMAGES]
    ALL_IMAGES = PRE_PUSH_STAGE_IMAGES + COMMON_STAGE_IMAGES
    IMAGES_TO_PUSH = [image for image in ALL_IMAGES if image.to_push and image.to_build]

    # pre_push stage standard images build
    FORMATTER.banner("Standard Build")
    build_images(standard_images)

    # pre_push stage example images build
    FORMATTER.banner("Example Build")
    build_images(example_images)

    # Common stage build
    FORMATTER.banner("Common Build")
    build_images(common_stage_images, make_dummy_boto_client=True)

    FORMATTER.banner("Push Started")
    push_images(IMAGES_TO_PUSH)

    # After the build, display logs/summary for all the images.
    FORMATTER.banner("Build Logs")
    show_build_logs(ALL_IMAGES)

    FORMATTER.banner("Summary")
    show_build_summary(ALL_IMAGES)

    FORMATTER.banner("Errors")
    is_any_build_failed, is_any_build_failed_size_limit = show_build_errors(ALL_IMAGES)

    # From all images, filter the images that were supposed to be built and upload their metrics
    BUILT_IMAGES = [image for image in ALL_IMAGES if image.to_build]

    FORMATTER.banner("Upload Metrics")
    upload_metrics(BUILT_IMAGES, BUILDSPEC, is_any_build_failed, is_any_build_failed_size_limit)

    FORMATTER.banner("Test Env")
    # Set environment variables to be consumed by test jobs
    test_trigger_job = utils.get_codebuild_project_name()
    # Tests should only run on images that were pushed to the repository
    utils.set_test_env(
        IMAGES_TO_PUSH, BUILD_CONTEXT=os.getenv("BUILD_CONTEXT"), TEST_TRIGGER=test_trigger_job,
    )

def get_common_stage_image_object(pre_push_stage_image_object):
    """
    Creates a common stage image object for a pre_push stage image. If for a pre_push stage image we create a common 
    stage image, then we do not push the pre_push stage image to the repository. Instead, we just push its common stage 
    image to the repository. Therefore, inside the function pre_push_stage_image_object is made NON-PUSHABLE.

    :param pre_push_stage_image_object: DockerImage, an object of class DockerImage
    :return: CommonStageImage, an object of class CommonStageImage. CommonStageImage inherits DockerImage.
    """
    common_stage_image_object = CommonStageImage(
        info=pre_push_stage_image_object.info,
        dockerfile=os.path.join(os.sep, utils.get_root_folder_path(), "miscellaneous_dockerfiles", "Dockerfile.common"),
        repository=pre_push_stage_image_object.repository,
        tag=pre_push_stage_image_object.tag,
        to_build=pre_push_stage_image_object.to_build,
        stage=constants.COMMON_STAGE,
    )
    pre_push_stage_image_object.to_push = False
    return common_stage_image_object

def show_build_logs(images):
    """
    Display and save the build logs for a list of input images.

    :param images: list[DockerImage]
    """

    if not os.path.isdir("logs"):
        os.makedirs("logs")

    for image in images:
        image_description = f"{image.name}-{image.stage}"
        FORMATTER.title(image_description)
        FORMATTER.table(image.info.items())
        FORMATTER.title(f'Ending Logs for {image_description}')
        FORMATTER.print_lines(image.log[-1][-2:])
        flattened_logs = list(itertools.chain(*image.log))
        with open(f"logs/{image_description}", "w") as fp:
            fp.write("/n".join(flattened_logs))
            image.summary["log"] = f"logs/{image_description}"

def show_build_summary(images):
    """
    Display the summary for a list of input images.

    :param images: list[DockerImage]
    """

    for image in images:
        FORMATTER.title(image.name)
        FORMATTER.table(image.summary.items())

def show_build_errors(images):
    """
    Iterates through each image to check if there is any image that has a failed status. In case
    an image with a failed status is found, it raises an exception.

    :param images: list[DockerImage]
    """
    is_any_build_failed = False
    is_any_build_failed_size_limit = False

    for image in images:
        if image.build_status == constants.FAIL:
            FORMATTER.title(image.name)
            FORMATTER.print_lines(image.log[-1][-10:])
            is_any_build_failed = True
        else:
            if image.build_status == constants.FAIL_IMAGE_SIZE_LIMIT:
                is_any_build_failed_size_limit = True
    if is_any_build_failed:
        raise Exception("Build failed")
    else:
        if is_any_build_failed_size_limit:
            FORMATTER.print("Build failed. Image size limit breached.")
        else:
            FORMATTER.print("No errors")
    return is_any_build_failed, is_any_build_failed_size_limit

def upload_metrics(images, BUILDSPEC, is_any_build_failed, is_any_build_failed_size_limit):
    """
    Uploads Metrics for a list of images.

    :param images: list[DockerImage]
    :param BUILDSPEC: Buildspec
    :param is_any_build_failed: bool
    :param is_any_build_failed_size_limit: bool
    """
    metrics = Metrics(
        context=constants.BUILD_CONTEXT,
        region=BUILDSPEC["region"],
        namespace=constants.METRICS_NAMESPACE,
    )
    for image in images:
        try:
            metrics.push_image_metrics(image)
        except Exception as e:
            if is_any_build_failed or is_any_build_failed_size_limit:
                raise Exception(f"Build failed.{e}")
            else:
                raise Exception(f"Build passed. {e}")

    if is_any_build_failed_size_limit:
        raise Exception("Build failed because of file limit")

    FORMATTER.print("Metrics Uploaded")

def build_images(images, make_dummy_boto_client=False):
    """
    Takes a list of images and executes their build process concurrently. 

    :param images: list[DockerImage]
    :param make_dummy_boto_client: bool, specifies if a dummy client should be declared or not.

    TODO: The parameter make_dummy_boto_client should be removed when get_dummy_boto_client method is removed.
    """
    THREADS = {}
    # In the context of the ThreadPoolExecutor each instance of image.build submitted
    # to it is executed concurrently in a separate thread.
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        #### TODO: Remove this entire if block when get_dummy_boto_client is removed ####
        if make_dummy_boto_client:
            get_dummy_boto_client()
        for image in images:
            THREADS[image.name] = executor.submit(image.build)
    # the FORMATTER.progress(THREADS) function call also waits until all threads have completed
    FORMATTER.progress(THREADS)

#### TODO: Remove this entire method when https://github.com/boto/boto3/issues/1592 is resolved ####
def get_dummy_boto_client():
    """
    Makes a dummy boto3 client to ensure that boto3 clients behave in a thread safe manner.
    In absence of this method, the behaviour documented in https://github.com/boto/boto3/issues/1592 is observed.
    Once https://github.com/boto/boto3/issues/1592 is resolved, this method can be removed.

    :return: BotocoreClientSTS
    """
    return boto3.client("sts", region_name=os.getenv("REGION"))

def push_images(images):
    """
    Takes a list of images and PUSHES them to ECR concurrently. 

    :param images: list[DockerImage]
    """
    THREADS = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=constants.MAX_WORKER_COUNT_FOR_PUSHING_IMAGES) as executor:
        for image in images:
            THREADS[image.name] = executor.submit(image.push_image)
    FORMATTER.progress(THREADS)



def tag_image_with_pr_number(image_tag):
    pr_number = os.getenv("CODEBUILD_SOURCE_VERSION").replace("/", "-")
    return f"{image_tag}-{pr_number}"


def tag_image_with_datetime(image_tag):
    datetime_suffix = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return f"{image_tag}-{datetime_suffix}"


def modify_repository_name_for_context(image_repo_uri, build_context):
    repo_uri_values = image_repo_uri.split("/")
    repo_name = repo_uri_values[-1]
    if build_context == "MAINLINE":
        repo_uri_values[-1] = repo_name.replace(
            constants.PR_REPO_PREFIX, constants.MAINLINE_REPO_PREFIX
        )
    elif build_context == "NIGHTLY":
        repo_uri_values[-1] = repo_name.replace(
            constants.PR_REPO_PREFIX, constants.NIGHTLY_REPO_PREFIX
        )
    return "/".join(repo_uri_values)
