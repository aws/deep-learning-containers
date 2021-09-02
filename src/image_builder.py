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

from context import Context
from metrics import Metrics
from image import DockerImage
from conclusion_stage_image import ConclusionStageImage
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
    INITIAL_STAGE_IMAGES = []
    CONCLUSION_STAGE_IMAGES = []

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
            base_image_object = _find_image_object(INITIAL_STAGE_IMAGES, image_config["base_image_name"])
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
        
        #Create initial stage docker object
        initial_stage_image_object = DockerImage(
            info=info,
            dockerfile=image_config["docker_file"],
            repository=image_repo_uri,
            tag=image_tag,
            to_build=image_config["build"],
            stage=constants.INITIAL_STAGE,
            context=context,
        )

        ##### Create Conclusion stage docker object #####
        # If for an initial stage image we create a conclusion stage image, then we do not push the initial stage image
        # to the repository. Instead, we just push its conclusion stage image to the repository. Therefore,
        # inside function get_conclusion_stage_image_object we make initial_stage_image_object non pushable.
        conclusion_stage_image_object = get_conclusion_stage_image_object(initial_stage_image_object)

        FORMATTER.separator()

        INITIAL_STAGE_IMAGES.append(initial_stage_image_object)
        if conclusion_stage_image_object is not None:
            CONCLUSION_STAGE_IMAGES.append(conclusion_stage_image_object)

    FORMATTER.banner("DLC")
    FORMATTER.title("Status")
    
    # Standard images must be built before example images
    # Example images will use standard images as base
    # Conclusion images must be built at the end as they will consume respective standard and example images
    standard_images = [image for image in INITIAL_STAGE_IMAGES if "example" not in image.name.lower()]
    example_images = [image for image in INITIAL_STAGE_IMAGES if "example" in image.name.lower()]
    conclusion_stage_images = [image for image in CONCLUSION_STAGE_IMAGES]
    ALL_IMAGES = INITIAL_STAGE_IMAGES + CONCLUSION_STAGE_IMAGES
    IMAGES_TO_PUSH = [image for image in ALL_IMAGES if image.to_push and image.to_build]

    #initial stage standard images build
    FORMATTER.banner("Standard Build")
    build_images(standard_images)

    #initial stage example images build
    FORMATTER.banner("Example Build")
    build_images(example_images)
       
    #Conclusion stage build
    if len(conclusion_stage_images) > 0:
        FORMATTER.banner("Conclusion Build")
        build_images(conclusion_stage_images, make_dummy_boto_client=True)
    
    FORMATTER.banner("Push Started")
    push_images(IMAGES_TO_PUSH)

    FORMATTER.title("Log Display")
    #After the build, display logs/summary for all the images.
    show_build_logs(ALL_IMAGES)
    show_build_summary(ALL_IMAGES)
    is_any_build_failed, is_any_build_failed_size_limit = show_build_errors(ALL_IMAGES)

    #From all images, filter the images that were supposed to be built and upload their metrics
    BUILT_IMAGES = [image for image in ALL_IMAGES if image.to_build]

    FORMATTER.title("Upload Metrics")
    # change logic here. upload metrics only for the Conclusion stage image
    upload_metrics(BUILT_IMAGES, BUILDSPEC, is_any_build_failed, is_any_build_failed_size_limit)

    FORMATTER.title("Setting Test Env")
    # Set environment variables to be consumed by test jobs
    test_trigger_job = utils.get_codebuild_project_name()
    # Tests should only run on images that were pushed to the repository
    utils.set_test_env(
        IMAGES_TO_PUSH,
        BUILD_CONTEXT=os.getenv("BUILD_CONTEXT"),
        TEST_TRIGGER=test_trigger_job,
    )

def get_conclusion_stage_image_object(initial_stage_image_object):
    # Check if this is only required for mainline
    # if build_context == "MAINLINE":
    conclusion_stage_image_object = None
    conclusion_stage_image_object = ConclusionStageImage(
        info=initial_stage_image_object.info,
        dockerfile=os.path.join(os.sep, os.getenv("ROOT_FOLDER_PATH"), "src", "Dockerfile.multipart"),
        repository=initial_stage_image_object.repository,
        tag=initial_stage_image_object.tag,
        to_build=initial_stage_image_object.to_build,
        stage=constants.CONCLUSION_STAGE,
    )
    initial_stage_image_object.to_push = False
    return conclusion_stage_image_object

def show_build_logs(images):

    FORMATTER.title("Build Logs")

    if not os.path.isdir("logs"):
        os.makedirs("logs")

    for image in images:
        image_description = f"{image.name}-{image.stage}"
        FORMATTER.title(image_description)
        FORMATTER.table(image.info.items())
        FORMATTER.separator()
        FORMATTER.print_lines(image.log)
        with open(f"logs/{image_description}", "w") as fp:
            fp.write("/n".join(image.log))
            image.summary["log"] = f"logs/{image_description}"

def show_build_summary(images):

    FORMATTER.title("Summary")

    for image in images:
        FORMATTER.title(image.name)
        FORMATTER.table(image.summary.items())

def show_build_errors(images):
    FORMATTER.title("Errors")
    is_any_build_failed = False
    is_any_build_failed_size_limit = False

    for image in images:
        if image.build_status == constants.FAIL:
            FORMATTER.title(image.name)
            FORMATTER.print_lines(image.log[-10:])
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

    FORMATTER.title("Uploading Metrics")
    is_any_build_failed = False
    is_any_build_failed_size_limit = False
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

    FORMATTER.separator()

def build_images(images, make_dummy_boto_client=False):
    THREADS = {}
    # In the context of the ThreadPoolExecutor each instance of image.build submitted
    # to it is executed concurrently in a separate thread.
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        #### TODO: Remove this entire if block when https://github.com/boto/boto3/issues/1592 is resolved ####
        if make_dummy_boto_client: 
            get_dummy_boto_client()
        for image in images:
            FORMATTER.print(f"image_object.context {image.context}")
            THREADS[image.name] = executor.submit(image.build)
    # the FORMATTER.progress(THREADS) function call also waits until all threads have completed
    FORMATTER.progress(THREADS)

#### TODO: Remove this entire method when https://github.com/boto/boto3/issues/1592 is resolved ####
def get_dummy_boto_client():
    # In absence of this method, the behaviour documented in https://github.com/boto/boto3/issues/1592 is observed.
    # If this function is not added, boto3 fails because boto3 sessions are not thread safe.
    # However, once a dummy client is created, it is ensured that the calls are thread safe.
    import boto3
    return boto3.client("sts", region_name=os.getenv('REGION'))

def push_images(images):
    THREADS = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=constants.MAX_WORKER_COUNT_FOR_PUSHING_IMAGES
    ) as executor:
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
