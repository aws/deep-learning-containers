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

import os
import argparse
from copy import deepcopy
import concurrent.futures

# import utils
import constants

from context import Context
from metrics import Metrics
from image import DockerImage
from buildspec import Buildspec
from output import OutputFormatter


#Test
from github import GitHubHandler

# TODO: Abstract away to ImageBuilder class
if __name__ == "__main__":
    # TEST
    handler = GitHubHandler()
    handler.set_status('success', context="HEYIMATEST", description="Sup nikhil")

    parser = argparse.ArgumentParser(description="Program to build docker images")
    parser.add_argument("--buildspec", required=True, type=str)
    ARGS = parser.parse_args()

    BUILDSPEC = Buildspec()
    BUILDSPEC.load(ARGS.buildspec)
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

        ARTIFACTS.update({"dockerfile": {"source": image_config["docker_file"], "target": "Dockerfile"}})

        context = Context(
            ARTIFACTS, f"build/{image_name}.tar.gz", image_config["root"]
        )

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
        }

        image_object = DockerImage(
            info=info,
            dockerfile=image_config["docker_file"],
            repository=image_config["repository"],
            tag=image_config["tag"],
            to_build=image_config["build"],
            context=context,
        )

        IMAGES.append(image_object)

    FORMATTER = OutputFormatter(constants.PADDING)

    FORMATTER.banner("DLC")
    FORMATTER.title("Status")

    THREADS = {}

    # In the context of the ThreadPoolExecutor each instance of image.build submitted 
    # to it is executed concurrently in a separate thread. 
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for image in IMAGES:
            THREADS[image.name] = executor.submit(image.build)

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
