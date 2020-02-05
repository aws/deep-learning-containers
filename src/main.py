'''
Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"). You
may not use this file except in compliance with the License. A copy of
the License is located at

    http://aws.amazon.com/apache2.0/

or in the "license" file accompanying this file. This file is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
'''

import os
import concurrent.futures

import utils
import constants

from image import DockerImage
from context import Context
from output import OutputFormatter
from buildspec import Buildspec
from metrics import Metrics

# TODO: Abstract away to ImageBuilder class
if __name__ == "__main__":
    ARGS = utils.parse_args()

    BUILDSPEC = Buildspec()
    BUILDSPEC.load(ARGS.buildspec)

    IMAGES = []
    ARTIFACTS = list(BUILDSPEC["context"].items())

    for image in BUILDSPEC["images"].items():
        IMAGE_ARTIFACTS = []
        IMAGE_ARTIFACTS += ARTIFACTS

        if image[1].get("version") is not None:
            if BUILDSPEC["version"] != image[1].get("version"):
                continue

        if image[1].get("context") is not None:
            IMAGE_ARTIFACTS += list(image[1]["context"].items())

        IMAGE_ARTIFACTS.append([image[1]["docker_file"], "Dockerfile"])
        try:
            context = Context(IMAGE_ARTIFACTS, f"build/{image[0]}.tar.gz", image[1]["root"])
        except Exception as e:
            print(e)
            import pdb
            pdb.set_trace()

        """
        Override parameters from parent in child.
        """

        info = {
            "account_id": str(BUILDSPEC["account_id"]),
            "region": str(BUILDSPEC["region"]),
            "framework": str(BUILDSPEC["framework"]),
            "version": str(BUILDSPEC["version"]),
            "root": str(image[1]["root"]),
            "name": str(image[0]),
            "device_type": str(image[1]["device_type"]),
            "python_version": str(image[1]["python_version"]),
            "image_type": str(image[1]["image_type"]),
            "image_size_baseline": int(image[1]["image_size_baseline"])
        }

        image_object = DockerImage(
            info=info,
            dockerfile=image[1]["docker_file"],
            repository=image[1]["repository"],
            tag=image[1]["tag"],
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
        metrics = Metrics(context=constants.BUILD_CONTEXT, region=BUILDSPEC['region'],namespace=constants.METRICS_NAMESPACE)
        for image in IMAGES:
            if image.build_status == constants.FAIL:
                FORMATTER.title(image.name)
                FORMATTER.print_lines(image.log[-10:])
                ANY_FAIL = True

        FORMATTER.title("Uploading Metrics")
        for image in IMAGES:
            try:
                metrics.push_image_metrics(image)
            except Exception as e:
                if ANY_FAIL:
                    raise Exception(f"Build failed.{e}")
                else:
                    raise Exception(f"Build passed. {e}")

        if ANY_FAIL:
            raise Exception("Build failed")
        else:
            FORMATTER.print("No errors")
        FORMATTER.separator()
