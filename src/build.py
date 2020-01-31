"""
This script takes a buildspec as input and builds the docker images
"""

import os
import concurrent.futures
from collections import defaultdict

import utils
import constants

from image import DockerImage
from context import Context
from output import OutputFormatter

if __name__ == "__main__":
    ARGS = utils.parse_args()
    BUILDSPEC = utils.parse_buildspec(ARGS.buildspec)

    IMAGES = []

    ARTIFACTS = list(BUILDSPEC["context"].items())

    for image in BUILDSPEC["images"].items():
        if image[1].get("context") is not None:
            ARTIFACTS += list(image[1]["context"])

        ARTIFACTS.append([image[1]["docker_file"], "Dockerfile"])

        context = Context(ARTIFACTS, f"build/{image[0]}.tar.gz", BUILDSPEC["root"])

        image_object = DockerImage(
            account_id=BUILDSPEC["account_id"],
            repository=BUILDSPEC["repository"],
            region=BUILDSPEC["region"],
            framework=BUILDSPEC["framework"],
            version=BUILDSPEC["version"],
            root=BUILDSPEC["root"],
            name=image[0],
            device_type=image[1]["device_type"],
            python_version=image[1]["python_version"],
            image_type=image[1]["image_type"],
            image_size_baseline=image[1]["image_size_baseline"],
            dockerfile=image[1]["docker_file"],
            tag=image[1]["tag"],
            example=image[1]["example"],
            build=image[1]["build"],
            context=context,
        )

        IMAGES.append(image_object)

    FORMATTER = OutputFormatter(constants.PADDING)

    FORMATTER.banner("DLC")
    FORMATTER.title("Status")

    THREADS = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for image in IMAGES:
            THREADS[image.name] = executor.submit(image.build)

        FORMATTER.progress(THREADS)

        FORMATTER.title("Build Logs")

        if not os.path.isdir("logs"):
            os.makedirs("logs")

        for image in IMAGES:
            FORMATTER.title(image.name)
            FORMATTER.table(BUILDSPEC["images"][image.name].items())
            FORMATTER.separator()
            FORMATTER.print_lines(THREADS[image.name].result()["response"])
            with open(f"logs/{image.name}", "w") as fp:
                fp.write("/n".join(THREADS[image.name].result()["response"]))

        FORMATTER.title("Summary")

        SUMMARY = defaultdict(dict)

        STATUS_CODE = {constants.SUCCESS: "Success", constants.FAIL: "Failure"}

        for image in IMAGES:
            SUMMARY[image.name]["status"] = STATUS_CODE[
                THREADS[image.name].result()["status"]
            ]
            SUMMARY[image.name]["buildtime"] = (
                str((image.endtime - image.starttime).seconds) + "s"
            )
            if THREADS[image.name].result()["status"] != constants.FAIL:
                SUMMARY[image.name]["ECR"] = image.ecr_url

        for image in IMAGES:
            FORMATTER.title(image.name)
            FORMATTER.table(SUMMARY[image.name].items())

        ANY_FAIL = False
        for image in IMAGES:
            if THREADS[image.name].result()["status"] == constants.FAIL:
                FORMATTER.title(image.name)
                FORMATTER.print_lines(THREADS[image.name].result()["response"][-10:])
                ANY_FAIL = True

        if ANY_FAIL:
            raise Exception("Build failed")
