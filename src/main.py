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
from buildspec import Buildspec

if __name__ == "__main__":
    ARGS = utils.parse_args()

    buildspec = Buildspec()
    buildspec.load(ARGS.buildspec) 

    IMAGES = []
    ARTIFACTS = list(buildspec["context"].items())

    for image in buildspec["images"].items():
        
        if image[1].get('version') is not None:
            if buildspec['version'] != image[1].get('version'):
                continue     
    
        if image[1].get("context") is not None:
            ARTIFACTS += list(image[1]["context"])

        ARTIFACTS.append([image[1]["docker_file"], "Dockerfile"])

        context = Context(ARTIFACTS, f"build/{image[0]}.tar.gz", buildspec["root"])

        '''
        Override parameters from parent in child.
        '''

        info = { 'account_id': buildspec['account_id'],
                 'region': buildspec['region'],
                 'framework': buildspec['framework'],
                 'version': buildspec['version'],
                 'root': buildspec['root'],
                 'name': image[0],
                 'device_type': image[1]['device_type'],
                 'python_version': image[1]['python_version'] ,
                 'image_type': buildspec['image_type'],
                 'image_size_baseline': image[1]['image_size_baseline'],
                  }

        image_object = DockerImage(
            info=info,
            dockerfile=image[1]["docker_file"],
            repository=buildspec["repository"],
            tag=image[1]["tag"],
            build=image[1]["build"],
            context=context
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
            FORMATTER.table(buildspec["images"][image.name].items())
            FORMATTER.separator()
            FORMATTER.print_lines(image.log)
            with open(f"logs/{image.name}", "w") as fp:
                fp.write("/n".join(image.log))

        FORMATTER.title("Summary")

        SUMMARY = defaultdict(dict)

        STATUS_CODE = {constants.SUCCESS: "Success", constants.FAIL: "Failure"}

        #for image in IMAGES:
        #    SUMMARY[image.name]["status"] = STATUS_CODE[
        #        image.summary["status"]
        #    ]
        #    SUMMARY[image.name]["buildtime"] = (
        #        str((image.endtime - image.starttime).seconds) + "s"
        #    )
        #    if image.summary["status"] != constants.FAIL:
        #        SUMMARY[image.name]["ECR"] = image.ecr_url

        for image in IMAGES:
            FORMATTER.title(image.name)
            FORMATTER.table(image.summary.items())

        ANY_FAIL = False
        for image in IMAGES:
            if image.summary["status"] == constants.FAIL:
                FORMATTER.title(image.name)
                FORMATTER.print_lines(image.log[-10:])
                ANY_FAIL = True

        if ANY_FAIL:
            raise Exception("Build failed")
