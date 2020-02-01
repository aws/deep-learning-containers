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
        info,
        dockerfile,
        repository,
        tag,
        build,
        context=None,
    ):
        """
        The constructor for DockerImage class converts the buildspec parameters as class attributes
        """

        # Meta-data about the image should go to info.
        # All keys in info are accessible as attributes
        # of this class
        self.info = info
        self.summary = {}

        self.dockerfile = dockerfile
        self.context = context

        # TODO: Add ability to tag image with multiple tags
        self.repository = repository
        self.tag = tag
        self.ecr_url = f"{self.repository}:{self.tag}"

        self.to_build = build

        self.client = APIClient(base_url=constants.DOCKER_URL)

    def __getattr__(self,name):
        return self.info[name]

    def build(self):
        """
        The build function builds the specified docker image
        """
        if not self.to_build:
            self.summary["status"] = constants.SUCCESS
            self.summary["response"] = "Not built"
            return self.summary['status']

        self.summary['starttime'] = datetime.now()
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
                    self.summary['status'] = constants.FAIL
                    self.summary['response'] = response
                    return self.summary["status"] 
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
                    self.summary['status'] = constants.FAIL
                    self.summary['response'] = response
                    self.summary['endtime'] = datetime.now()
                    return self.summary['status'] 
                elif line.get("stream") is not None:
                    response.append(line["stream"])
                else:
                    response.append(str(line))

            self.summary['status'] = constants.FAIL
            self.summary['response'] = response
            self.summary['endtime'] = datetime.now()

            return self.summary['status']
