from github import GitHubHandler
import os
from image_builder import image_builder
import argparse
import constants 
import re

# import utils
import constants

from context import Context
from metrics import Metrics
from image import DockerImage
from buildspec import Buildspec
from output import OutputFormatter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Program to build docker images")
    parser.add_argument("--device_types", type=str, default=constants.ALL)
    parser.add_argument("--image_types", type=str, default=constants.ALL)
    parser.add_argument("--python_versions", type=str, default=constants.ALL)

    args = parser.parse_args()

    # Set necessary environment variables
    to_build = {'device_types': constants.DEVICE_TYPES,
                'image_types': constants.IMAGE_TYPES,
                'python_versions': constants.PYTHON_VERSIONS}

    framework = args.framework
    device_types = []
    image_types = []
    python_versions = []

    if os.environ.get('BUILD_CONTEXT') == 'PR':
        g = GitHubHandler("aws", "deep-learning-containers")
        PR_NUMBER = os.getenv("CODEBUILD_SOURCE_VERSION")
        files = '\n'.join(g.get_pr_files_changed(PR_NUMBER))

        dockerfile_match = re.findall("\S+Dockerfile\S+", files) 
        buildspec_match = re.findall("\S+\/buildspec.yml", files)
        src_match = re.findall("src\/\S+", files)

        for dockerfile in dockerfile_match:
            _, image_type, _, version, python_version, _ = dockerfile.split('/')
            device_type = dockerfile.split('.')[-1]
            device_types.append(device_type)
            image_types.append(image_type)
            python_versions.append(python_version)

        for buildspec in buildspec_match:
            buildspec_framework, _ = buildspec.split('/')
            if buildspec_framework == framework:
                device_types = constants.ALL
                image_types = constants.ALL
                python_versions = constants.ALL

        if len(src_match) != 0:
            device_types = constants.ALL
            image_types = constants.ALL
            python_versions = constants.ALL

    else:        
        device_types = args.device_types.split(',')
        image_types = args.image_types.split(',')
        python_versions = args.python_versions.split(',')
    
    if device_types != constants.ALL:
        to_build['device_types'] = constants.DEVICE_TYPES.intersection(set(device_types))
    if image_types != constants.ALL:
        to_build['image_types'] = constants.IMAGE_TYPES.intersection(set(image_types))
    if python_versions != constants.ALL:
        to_build['python_versions'] = constants.PYTHON_VERSIONS.intersection(set(python_versions))
    
    for device_type in to_build['device_types']:
        for image_type in to_build['image_types']:
            for python_version in to_build['python_versions']:
                env_variable = f"{framework.upper()}_{device_type.upper()}_{image_type.upper()}_{python_version.upper()}"
                os.environ[env_variable] = 'true'

    image_builder(args.buildspec)
