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

from utils import generate_safety_report_for_image
from context import Context

import constants
import os


class DockerImage:
    """
    The DockerImage class has the functions and attributes for building the dockerimage
    """

    def __init__(
        self, info, dockerfile, repository, tag, to_build, stage, context=None,
    ):

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

        # TODO: Add ability to tag image with multiple tags
        self.repository = repository
        self.tag = tag
        self.ecr_url = f"{self.repository}:{self.tag}"

        if not isinstance(to_build, bool):
            to_build = True if to_build == "true" else False

        self.to_build = to_build
        self.build_status = None
        self.client = APIClient(base_url=constants.DOCKER_URL)
        self.log = []

    def __getattr__(self, name):
        return self.info[name]

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

    def generate_conclude_stage_context(self, safety_report_path, tarfile_name='conclusion-stage-file'):
        ARTIFACTS = {}
        ARTIFACTS.update(
                    {
                        "safety_report": {
                            "source": safety_report_path,
                            "target": "safety_report.json"
                        }
                    })
        ARTIFACTS.update(
                    {
                        "dockerfile": {
                            "source": f"Dockerfile.multipart",
                            "target": "Dockerfile",
                        }
                    }
                )
        
        artifact_root = os.path.join(os.sep, os.getenv("PYTHONPATH"), "src") + "/"
        return Context(ARTIFACTS, context_path=f'build/{tarfile_name}.tar.gz',artifact_root=artifact_root)

    
    def pre_build_configuration_for_conclsion_stage(self):
        ## Generate safety scan report for the first stage image and add the file to artifacts
        first_stage_image_uri = self.build_args['FIRST_STAGE_IMAGE']
        processed_image_uri = first_stage_image_uri.replace('.','-').replace('/','-').replace(':','-')
        storage_file_path = f"{os.getenv('PYTHONPATH')}/src/{processed_image_uri}_safety_report.json"
        generate_safety_report_for_image(first_stage_image_uri, storage_file_path=storage_file_path)
        self.context = self.generate_conclude_stage_context(storage_file_path, tarfile_name=processed_image_uri)

    def pre_build_configuration(self):

        if not self.to_build:
            self.log = ["Not built"]
            self.build_status = constants.NOT_BUILT
            self.summary["status"] = constants.STATUS_MESSAGE[self.build_status]
            return self.build_status

        if self.info.get("base_image_uri"):
            self.build_args["BASE_IMAGE"] = self.info["base_image_uri"]

        if self.ecr_url:
            self.build_args["FIRST_STAGE_IMAGE"] = self.ecr_url

        if self.info.get("extra_build_args"):
            self.build_args.update(self.info.get("extra_build_args"))
        
        if self.info.get("labels"):
            self.labels.update(self.info.get("labels"))

        if self.stage == constants.CONCLUSION_STAGE:
            self.pre_build_configuration_for_conclsion_stage()
        
        print(f"self.build_args {self.build_args}")
        print(f"self.labels {self.labels}")

    def build(self):
        """
        The build function builds the specified docker image
        """
        self.summary["start_time"] = datetime.now()
        self.pre_build_configuration()
        print(f"self.context {self.context}")
        if self.context:
            with open(self.context.context_path, "rb") as context_file:
                print("within context")
                self.docker_build(fileobj=context_file, custom_context=True)
                self.context.remove()  
        else:
            print("out of context")
            self.docker_build()
        #check the size after image is built.
        self.image_size_check()
        ## This return is necessary. Otherwise formatter fails while displaying the status.
        return self.build_status
    
    def docker_build(self, fileobj=None, custom_context=False):
        response = []
        for line in self.client.build(
                fileobj=fileobj,
                path=self.dockerfile,
                custom_context=custom_context,
                rm=True,
                decode=True,
                tag=self.ecr_url,
                buildargs=self.build_args,
                labels=self.labels
            ):
                if line.get("error") is not None:
                    self.context.remove()
                    response.append(line["error"])

                    self.log = response
                    self.build_status = constants.FAIL
                    self.summary["status"] = constants.STATUS_MESSAGE[self.build_status]
                    self.summary["end_time"] = datetime.now()

                    return self.build_status

                if line.get("stream") is not None:
                    response.append(line["stream"])
                elif line.get("status") is not None:
                    response.append(line["status"])
                else:
                    response.append(str(line))

        self.log = response
        print(f"self.log {self.log}")
        self.build_status = constants.SUCCESS
        #TODO: return required?
        return self.build_status


    def image_size_check(self):
        response = []
        self.summary["image_size"] = int(
                self.client.inspect_image(self.ecr_url)["Size"]
            ) / (1024 * 1024)
        if self.summary["image_size"] > self.info["image_size_baseline"] * 1.20:
            response.append("Image size baseline exceeded")
            response.append(f"{self.summary['image_size']} > 1.2 * {self.info['image_size_baseline']}")
            response += self.collect_installed_packages_information()
            self.build_status = constants.FAIL_IMAGE_SIZE_LIMIT
        else:
            self.build_status = constants.SUCCESS
        self.log = response
        print(f"self.log {self.log}")
        #TODO: return required?
        return self.build_status

    def push_image(self):

        for line in self.client.push(self.repository, self.tag, stream=True, decode=True):
            response = []
            if line.get("error") is not None:
                response.append(line["error"])

                self.log = response
                self.build_status = constants.FAIL
                self.summary["status"] = constants.STATUS_MESSAGE[self.build_status]
                self.summary["end_time"] = datetime.now()

                return self.build_status
            if line.get("stream") is not None:
                response.append(line["stream"])
            else:
                response.append(str(line))

        self.summary["status"] = constants.STATUS_MESSAGE[self.build_status]
        self.summary["end_time"] = datetime.now()
        self.summary["ecr_url"] = self.ecr_url
        self.log = response
        #TODO: return required?
        return self.build_status
