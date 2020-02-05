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
from datetime import datetime

from docker import APIClient

import constants


class DockerImage:
    """
    The DockerImage class has the functions and attributes for building the dockerimage
    """

    def __init__(
            self, info, dockerfile, repository, tag, build, context=None,
    ):

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
        self.build_status = None
        self.client = APIClient(base_url=constants.DOCKER_URL)
        self.log = []

    def __getattr__(self, name):
        return self.info[name]

    def build(self):
        """
        The build function builds the specified docker image
        """
        self.summary["starttime"] = datetime.now()

        if not self.to_build:
            self.log = ["Not built"]
            self.build_status = constants.NOT_BUILT
            self.summary["status"] = constants.STATUS_MESSAGE[self.build_status]
            return self.build_status

        with open(self.context.context_path, "rb") as context_file:
            response = []

            for line in self.client.build(
                    fileobj=context_file,
                    path=self.dockerfile,
                    custom_context=True,
                    rm=True,
                    decode=True,
                    tag=self.ecr_url,
            ):
                if line.get("error") is not None:
                    self.context.remove()
                    response.append(line["error"])

                    self.log = response
                    self.build_status = constants.FAIL
                    self.summary["status"] = constants.STATUS_MESSAGE[self.build_status]
                    self.summary["endtime"] = datetime.now()

                    return self.build_status

                elif line.get("stream") is not None:
                    response.append(line["stream"])
                elif line.get("status") is not None:
                    response.append(line["status"])
                else:
                    response.append(str(line))

            self.context.remove()

            self.summary['image_size'] = int(self.client.inspect_image(self.ecr_url)['Size']) /(1024 * 1024)
            if self.summary['image_size'] > self.info['image_size_baseline'] * 1.20:
                response.append("Image size baseline exceeded")
                self.log = response
                self.build_status = constants.FAIL
                self.summary["status"] = constants.STATUS_MESSAGE[self.build_status]
                self.summary["endtime"] = datetime.now()
                return self.build_status

            for line in self.client.push(
                    self.repository, self.tag, stream=True, decode=True
            ):
                if line.get("error") is not None:
                    response.append(line["error"])

                    self.log = response
                    self.build_status = constants.FAIL
                    self.summary["status"] = constants.STATUS_MESSAGE[self.build_status]
                    self.summary["endtime"] = datetime.now()

                    return self.build_status
                elif line.get("stream") is not None:
                    response.append(line["stream"])
                else:
                    response.append(str(line))

            self.build_status = constants.SUCCESS
            self.summary["status"] = constants.STATUS_MESSAGE[self.build_status]
            self.summary["endtime"] = datetime.now()
            self.summary["ecr_url"] = self.ecr_url
            self.log = response

            return self.build_status
