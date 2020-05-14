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
from buildspec import Buildspec
from output import OutputFormatter
from config import build_config


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
    FORMATTER = OutputFormatter(constants.PADDING)

    BUILDSPEC = Buildspec()
    BUILDSPEC.load(buildspec)
    IMAGES = []

    for image in BUILDSPEC["images"].items():
        ARTIFACTS = deepcopy(BUILDSPEC["context"])

        image_name = image[0]
        image_config = image[1]

        if image_config.get("version") is not None:
            if BUILDSPEC["version"] != image_config.get("version"):
                continue

        if image_config.get("context") is not None:
            ARTIFACTS.update(image_config["context"])

        build_context = os.getenv("BUILD_CONTEXT")
        image_tag = (
            tag_image_with_pr_number(image_config["tag"])
            if build_context == "PR"
            else image_config["tag"]
        )
        if not build_config.DISABLE_DATETIME_TAG:
            image_tag = tag_image_with_datetime(image_tag)
        image_repo_uri = (
            image_config["repository"]
            if build_context == "PR"
            else modify_repository_name_for_context(str(image_config["repository"]), build_context)
        )
        base_image_uri = None
        if image_config.get("base_image_name") is not None:
            base_image_object = _find_image_object(IMAGES, image_config["base_image_name"])
            base_image_uri = base_image_object.ecr_url

        ARTIFACTS.update(
            {
                "dockerfile": {
                    "source": image_config["docker_file"],
                    "target": "Dockerfile",
                }
            }
        )

        context = Context(ARTIFACTS, f"build/{image_name}.tar.gz", image_config["root"])

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
            "base_image_uri": base_image_uri
        }

        image_object = DockerImage(
            info=info,
            dockerfile=image_config["docker_file"],
            repository=image_repo_uri,
            tag=image_tag,
            to_build=image_config["build"],
            context=context,
        )

        IMAGES.append(image_object)

    FORMATTER.banner("DLC")
    FORMATTER.title("Status")

    THREADS = {}

    # In the context of the ThreadPoolExecutor each instance of image.build submitted
    # to it is executed concurrently in a separate thread.
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Standard images must be built before example images
        # Example images will use standard images as base
        standard_images = [image for image in IMAGES if "example" not in image.name.lower()]
        example_images = [image for image in IMAGES if "example" in image.name.lower()]

        for image in standard_images:
            THREADS[image.name] = executor.submit(image.build)

        # the FORMATTER.progress(THREADS) function call also waits until all threads have completed
        FORMATTER.progress(THREADS)

        for image in example_images:
            THREADS[image.name] = executor.submit(image.build)

        # the FORMATTER.progress(THREADS) function call also waits until all threads have completed
        FORMATTER.progress(THREADS)

        FORMATTER.title("Build Logs")

        if not os.path.isdir("logs"):
            os.makedirs("logs")

        for image in IMAGES:
            FORMATTER.title(image.name)
            FORMATTER.table(image.info.items())
            FORMATTER.separator()
            FORMATTER.print_lines(image.log)
            with open(f"logs/{image.name}", "w") as fp:
                fp.write("/n".join(image.log))
                image.summary["log"] = f"logs/{image.name}"

        FORMATTER.title("Summary")

        for image in IMAGES:
            FORMATTER.title(image.name)
            FORMATTER.table(image.summary.items())

        FORMATTER.title("Errors")
        ANY_FAIL = False
        for image in IMAGES:
            if image.build_status == constants.FAIL:
                FORMATTER.title(image.name)
                FORMATTER.print_lines(image.log[-10:])
                ANY_FAIL = True
        if ANY_FAIL:
            raise Exception("Build failed")
        else:
            FORMATTER.print("No errors")

        FORMATTER.title("Uploading Metrics")
        metrics = Metrics(
            context=constants.BUILD_CONTEXT,
            region=BUILDSPEC["region"],
            namespace=constants.METRICS_NAMESPACE,
        )
        for image in IMAGES:
            try:
                metrics.push_image_metrics(image)
            except Exception as e:
                if ANY_FAIL:
                    raise Exception(f"Build failed.{e}")
                else:
                    raise Exception(f"Build passed. {e}")

        FORMATTER.separator()

        # Set environment variables to be consumed by test jobs
        test_trigger_job = utils.get_codebuild_project_name()
        utils.set_test_env(
            IMAGES,
            BUILD_CONTEXT=os.getenv("BUILD_CONTEXT"),
            TEST_TRIGGER=test_trigger_job,
        )


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
