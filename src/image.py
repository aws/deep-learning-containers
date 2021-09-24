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
from datetime import datetime

from docker import APIClient
from docker import DockerClient

import constants
import logging
import json

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


class DockerImage:
    """
    The DockerImage class has the functions and attributes for building the dockerimage
    """

    def __init__(self, info, dockerfile, repository, tag, to_build, stage, context=None, to_push=True, additional_tags=[]):

        # Meta-data about the image should go to info.
        # All keys in info are accessible as attributes
        # of this class
        self.info = info
        self.summary = {}
        self.build_args = {}
        self.labels = {}
        self.stage = stage

        self.dockerfile = dockerfile
        self.context = context
        self.to_push = to_push

        # TODO: Add ability to tag image with multiple tags
        self.repository = repository
        self.tag = tag
        self.additional_tags = additional_tags
        self.ecr_url = f"{self.repository}:{self.tag}"

        if not isinstance(to_build, bool):
            to_build = True if to_build == "true" else False

        self.to_build = to_build
        self.build_status = None
        self.client = APIClient(base_url=constants.DOCKER_URL, timeout=constants.API_CLIENT_TIMEOUT)
        self.log = []
        self._corresponding_common_stage_image = None

    def __getattr__(self, name):
        return self.info[name]

    @property
    def is_child_image(self):
        """
        If we require a base image URI, the image is a child image (where the base image is the parent)
        """
        return bool(self.info.get('base_image_uri'))

    @property
    def is_test_promotion_enabled(self):
        return bool(self.info.get('enable_test_promotion'))

    @property
    def corresponding_common_stage_image(self):
        """
        Retrieve the corresponding common stage image for a given image.
        """
        return self._corresponding_common_stage_image

    @corresponding_common_stage_image.setter
    def corresponding_common_stage_image(self, docker_image_object):
        """
        For a pre-push stage image, it sets the value for the corresponding_common_stage_image variable.
        """
        if self.to_push:
            raise ValueError(
                "For any pre-push stage image, corresponding common stage image should only exist if the pre-push stage image is non-pushable."
            )
        self._corresponding_common_stage_image = docker_image_object

    def collect_installed_packages_information(self):
        """
        Returns an array with outcomes of the commands listed in the 'commands' array
        """
        docker_client = DockerClient(base_url=constants.DOCKER_URL)
        command_responses = []
        commands = ["pip list", "dpkg-query -Wf '${Installed-Size}\\t${Package}\\n'", "apt list --installed"]
        for command in commands:
            command_responses.append(f"\n{command}")
            command_responses.append(bytes.decode(docker_client.containers.run(self.ecr_url, command)))
        docker_client.containers.prune()
        return command_responses

    def get_tail_logs_in_pretty_format(self, number_of_lines=10):
        """
        Displays the tail of the logs.

        :param number_of_lines: int, number of ending lines to be printed
        :return: str, last number_of_lines of the logs concatenated with a new line
        """
        return "\n".join(self.log[-1][-number_of_lines:])

    def update_pre_build_configuration(self):
        """
        Updates image configuration before the docker client starts building the image.
        """
        if self.info.get("base_image_uri"):
            self.build_args["BASE_IMAGE"] = self.info["base_image_uri"]

        if self.info.get("extra_build_args"):
            self.build_args.update(self.info.get("extra_build_args"))

        if self.info.get("labels"):
            self.labels.update(self.info.get("labels"))

    def build(self):
        """
        The build function sets the stage for starting the docker build process for a given image. 

        :return: int, Build Status 
        """
        self.summary["start_time"] = datetime.now()

        # Confirm if building the image is required or not
        if not self.to_build:
            self.log.append(["Not built"])
            self.build_status = constants.NOT_BUILT
            self.summary["status"] = constants.STATUS_MESSAGE[self.build_status]
            return self.build_status

        # Conduct some preprocessing before building the image
        self.update_pre_build_configuration()

        # Start building the image
        with open(self.context.context_path, "rb") as context_file:
            self.docker_build(fileobj=context_file, custom_context=True)
            self.context.remove()

        if self.build_status != constants.SUCCESS:
            return self.build_status

        if not self.to_push:
            # If this image is not supposed to be pushed, in that case, we are already done
            # with building the image and do not need to conduct any further processing.
            self.summary["end_time"] = datetime.now()

        # check the size after image is built.
        self.image_size_check()

        # This return is necessary. Otherwise FORMATTER fails while displaying the status.
        return self.build_status

    def docker_build(self, fileobj=None, custom_context=False):
        """
        Uses low level Docker API Client to actually start the process of building the image.

        :param fileobj: FileObject, a readable file-like object pointing to the context tarfile.
        :param custom_context: bool
        :return: int, Build Status
        """
        response = [f"Starting the Build Process for {self.repository}:{self.tag}"]
        for line in self.client.build(
            fileobj=fileobj,
            path=self.dockerfile,
            custom_context=custom_context,
            rm=True,
            decode=True,
            tag=self.ecr_url,
            buildargs=self.build_args,
            labels=self.labels,
        ):
            if line.get("error") is not None:
                response.append(line["error"])
                self.log.append(response)
                self.build_status = constants.FAIL
                self.summary["status"] = constants.STATUS_MESSAGE[self.build_status]
                self.summary["end_time"] = datetime.now()

                LOGGER.info(f"Docker Build Logs: \n {self.get_tail_logs_in_pretty_format(100)}")
                LOGGER.error("ERROR during Docker BUILD")
                LOGGER.error(f"Error message received for {self.dockerfile} while docker build: {line}")

                return self.build_status

            if line.get("stream") is not None:
                response.append(line["stream"])
            elif line.get("status") is not None:
                response.append(line["status"])
            else:
                response.append(str(line))

        self.log.append(response)

        LOGGER.info(f"DOCKER BUILD LOGS: \n{self.get_tail_logs_in_pretty_format()}")
        LOGGER.info(f"Completed Build for {self.repository}:{self.tag}")

        self.build_status = constants.SUCCESS
        return self.build_status

    def image_size_check(self):
        """
        Checks if the size of the image is not greater than the baseline.

        :return: int, Build Status
        """
        response = [f"Starting image size check for {self.repository}:{self.tag}"]
        self.summary["image_size"] = int(self.client.inspect_image(self.ecr_url)["Size"]) / (1024 * 1024)
        if self.summary["image_size"] > self.info["image_size_baseline"] * 1.20:
            response.append("Image size baseline exceeded")
            response.append(f"{self.summary['image_size']} > 1.2 * {self.info['image_size_baseline']}")
            response += self.collect_installed_packages_information()
            self.build_status = constants.FAIL_IMAGE_SIZE_LIMIT
        else:
            response.append(f"Image Size Check Succeeded for {self.repository}:{self.tag}")
            self.build_status = constants.SUCCESS
        self.log.append(response)

        LOGGER.info(f"{self.get_tail_logs_in_pretty_format()}")

        return self.build_status

    def push_image(self, tag_value=None):
        """
        Pushes the Docker image to ECR using Docker low-level API client for docker.
        
        :param tag_value: str, an optional variable to provide a different tag
        :return: int, states if the Push was successful or not
        """
        tag = tag_value
        if tag_value is None:
            tag = self.tag

        response = [f"Starting image Push for {self.repository}:{tag}"]
        for line in self.client.push(self.repository, tag, stream=True, decode=True):
            if line.get("error") is not None:
                response.append(line["error"])
                self.log.append(response)
                self.build_status = constants.FAIL
                self.summary["status"] = constants.STATUS_MESSAGE[self.build_status]
                self.summary["end_time"] = datetime.now()

                LOGGER.info(f"Docker Build Logs: \n {self.get_tail_logs_in_pretty_format(100)}")
                LOGGER.error("ERROR during Docker PUSH")
                LOGGER.error(f"Error message received for {self.repository}:{tag} while docker push: {line}")

                return self.build_status
            if line.get("stream") is not None:
                response.append(line["stream"])
            else:
                response.append(str(line))

        self.summary["status"] = constants.STATUS_MESSAGE[self.build_status]
        self.summary["end_time"] = datetime.now()
        self.summary["ecr_url"] = self.ecr_url
        if "pushed_uris" not in self.summary:
            self.summary["pushed_uris"] = []
        self.summary["pushed_uris"].append(f"{self.repository}:{tag}")
        response.append(f"Completed Push for {self.repository}:{tag}")
        self.log.append(response)

        LOGGER.info(f"DOCKER PUSH LOGS: \n {self.get_tail_logs_in_pretty_format(2)}")
        return self.build_status

    def push_image_with_additional_tags(self):
        """
        Pushes an already built Docker image by applying additional tags to it.
        
        :return: int, states if the Push was successful or not
        """
        self.log.append([f"Started Tagging for {self.ecr_url}"])
        for additional_tag in self.additional_tags:
            response = [f"Tagging {self.ecr_url} as {self.repository}:{additional_tag}"]
            tagging_successful = self.client.tag(self.ecr_url, self.repository, additional_tag)
            if not tagging_successful:
                response.append(f"Tagging {self.ecr_url} with {additional_tag} unsuccessful.")
                self.log.append(response)
                LOGGER.error("ERROR during Tagging")
                LOGGER.error(f"Tagging {self.ecr_url} with {additional_tag} unsuccessful.")
                self.build_status = constants.FAIL
                self.summary["status"] = constants.STATUS_MESSAGE[self.build_status]
                return self.build_status
            response.append(f"Tagged {self.ecr_url} succussefully as {self.repository}:{additional_tag}")
            self.log.append(response)

            self.build_status = self.push_image(tag_value=additional_tag)
            if self.build_status != constants.SUCCESS:
                return self.build_status

        self.summary["status"] = constants.STATUS_MESSAGE[self.build_status]
        self.summary["end_time"] = datetime.now()
        self.log.append([f"Completed Tagging for {self.ecr_url}"])

        LOGGER.info(f"DOCKER TAG and PUSH LOGS: \n {self.get_tail_logs_in_pretty_format(5)}")
        return self.build_status
