"""
This script takes a buildspec as input and builds the docker images
"""

import os
import concurrent.futures

import utils
import constants

from image import DockerImage
from context import Context
from output import OutputFormatter
from buildspec import Buildspec

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
            import pdb
            pdb.set_trace()

        """
        Override parameters from parent in child.
        """

        info = {
            "account_id": BUILDSPEC["account_id"],
            "region": BUILDSPEC["region"],
            "framework": BUILDSPEC["framework"],
            "version": BUILDSPEC["version"],
            "root": image[1]["root"],
            "name": image[0],
            "device_type": image[1]["device_type"],
            "python_version": image[1]["python_version"],
            "image_type": image[1]["image_type"],
            "image_size_baseline": image[1]["image_size_baseline"],
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
        for image in IMAGES:
            if image.build_status == constants.FAIL:
                FORMATTER.title(image.name)
                FORMATTER.print_lines(image.log[-10:])
                ANY_FAIL = True

        if ANY_FAIL:
            raise Exception("Build failed")
