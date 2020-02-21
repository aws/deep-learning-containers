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
import json

import constants


def set_test_env(images, images_env="DLC_IMAGES", **kwargs):
    """
    Util function to write a file to be consumed by test env with necessary environment variables

    ENV variables set by os do not persist, as a new shell is instantiated for post_build steps

    :param images: List of image objects
    :param images_env: Name for the images environment variable
    :param env_file: File to write environment variables to
    :param kwargs: other environment variables to set
    """
    test_envs = []
    ecr_urls = []
    for docker_image in images:
        ecr_urls.append(docker_image.ecr_url)

    images_arg = " ".join(ecr_urls)
    test_envs.append({"name": images_env, "value": images_arg, "type": "PLAINTEXT"})

    if kwargs:
        for key, value in kwargs:
            test_envs.append({"name": key, "value": value, "type": "PLAINTEXT"})

    with open(constants.TEST_ENV, "w") as ef:
        json.dump(test_envs, ef)
