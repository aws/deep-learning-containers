"""
This file defines the DockerImage class which contains the attributes and functions for
a docker image
"""
from datetime import datetime

from docker import APIClient

import constants


class DockerImage:
    """
    The DockerImage class has the functions and attributes for building the dockerimage
    """

    def __init__(
        self,
        account_id,
        repository,
        region,
        framework,
        version,
        root,
        name,
        device_type,
        python_version,
        image_type,
        image_size_baseline,
        dockerfile,
        tag,
        example,
        build,
        context=None,
        docker_url="unix://var/run/docker.sock",
    ):
        """
        The constructor for DockerImage class converts the buildspec parameters as class attributes
        """

        self.account_id = account_id
        self.repository = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repository}"
        self.region = region
        self.framework = framework
        self.version = version
        self.root = root

        self.name = name
        self.device_type = device_type
        self.python_version = python_version
        self.image_type = image_type
        self.image_size_baseline = image_size_baseline
        self.dockerfile = dockerfile

        # TODO: Add ability to tag image with multiple tags
        self.tag = tag
        self.example = example
        self.to_build = build

        self.ecr_url = f"{self.repository}:{self.tag}"

        self.context = context

        self.client = APIClient(base_url=docker_url)

        self.starttime = None
        self.endtime = None

    def build(self):
        """
        The build function builds the specified docker image
        """
        if not self.to_build:
            return {"status": constants.SUCCESS, "response": "Not built"}

        self.starttime = datetime.now()
        with open(self.context.context_path, "rb") as fp:
            response = []

            # TODO: Move build status to a class
            for line in self.client.build(
                fileobj=fp,
                path=self.dockerfile,
                custom_context=True,
                rm=True,
                decode=True,
                tag=self.ecr_url,
            ):
                if line.get("error") is not None:
                    self.context.remove()
                    response.append(line["error"])
                    self.endtime = datetime.now()
                    return {"status": constants.FAIL, "response": response}
                elif line.get("stream") is not None:
                    response.append(line["stream"])
                elif line.get("status") is not None:
                    response.append(line["status"])
                else:
                    response.append(str(line))

            self.context.remove()

            for line in self.client.push(
                self.repository, self.tag, stream=True, decode=True
            ):
                if line.get("error") is not None:
                    response.append(line["error"])
                    self.endtime = datetime.now()
                    return {"status": constants.FAIL, "response": response}
                elif line.get("stream") is not None:
                    response.append(line["stream"])
                else:
                    response.append(str(line))

            self.endtime = datetime.now()

            return {"status": constants.SUCCESS, "response": response}
