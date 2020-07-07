import os
import json
import sys

import boto3
import logging
import requests

from botocore.exceptions import ClientError
from invoke import run

from src.buildspec import Buildspec

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))
LOGGER.setLevel(logging.INFO)


class DLCReleaseInformation:
    def __init__(self, dlc_account_id, dlc_region, dlc_repository, dlc_tag):
        self.dlc_account_id = dlc_account_id
        self.dlc_region = dlc_region
        self.dlc_repository = dlc_repository
        self.dlc_tag = dlc_tag

        self.container_name = self.run_container()

        imp_package_list_path = os.path.join(
            os.sep, os.path.dirname(__file__), "resources", "important_dlc_packages.yml"
        )
        self.imp_packages_to_record = Buildspec()
        self.imp_packages_to_record.load(imp_package_list_path)

        self._image_details = self.get_image_details_from_ecr()

    def get_boto3_ecr_client(self):
        return boto3.Session(region_name=self.dlc_region).client('ecr')

    def run_container(self):
        """
        Quickly run a container and assign it a name, so that different commands may be run on it
        :return:
        """
        repo_name, image_tag = self.image.split("/")[-1].split(":")

        container_name = f"{repo_name}-{image_tag}-release-information"

        run(f"docker rm -f {container_name}", warn=True, hide=True)

        run(f"docker run -id " f"--name {container_name} " f"--entrypoint='/bin/bash' " f"{self.image}", hide=True)

        return container_name

    def get_container_command_output(self, command):
        """
        Get stdout of the command executed.
        Note: does not handle stderr as it's not important in this context.
        :param command:
        :return:
        """
        docker_exec_cmd = f"docker exec -i {self.container_name}"
        run_stdout = run(f"{docker_exec_cmd} {command}", hide=True).stdout.strip()

        return run_stdout.strip()

    def get_image_details_from_ecr(self):
        _ecr = self.get_boto3_ecr_client()

        try:
            response = _ecr.describe_images(
                registryId=self.dlc_account_id,
                repositoryName=self.dlc_repository,
                imageIds=[{'imageTag': self.dlc_tag}]
            )
        except _ecr.exceptions.ImageNotFoundException:
            LOGGER.error(f"Image {self.image} does not exist in ECR repo.")
        except ClientError as e:
            LOGGER.error("ClientError when performing ECR operation. Exception: {}".format(e))

        return response["imageDetails"][0]

    @property
    def image(self):
        return f"{self.dlc_account_id}.dkr.ecr.{self.dlc_region}.amazonaws.com/{self.dlc_repository}:{self.dlc_tag}"

    @property
    def image_tags(self):
        return self._image_details['imageTags']

    @property
    def image_digest(self):
        return self._image_details['imageDigest']

    @property
    def bom_pip_packages(self):
        return self.get_container_command_output("pip freeze")

    @property
    def bom_apt_packages(self):
        return self.get_container_command_output("apt list --installed")

    @property
    def bom_pipdeptree(self):
        self.get_container_command_output("pip install pipdeptree")
        return self.get_container_command_output("pipdeptree")

    @property
    def imp_pip_packages(self):
        imp_pip_packages = {}
        container_pip_packages = json.loads(self.get_container_command_output("pip list --format=json"))

        for pip_package in self.imp_packages_to_record["pip_packages"]:
            for package_entry in container_pip_packages:
                if package_entry["name"] == pip_package:
                    imp_pip_packages[package_entry["name"]] = package_entry["version"]

        return imp_pip_packages

    @property
    def imp_apt_packages(self):
        imp_apt_packages = []

        for apt_package in self.imp_packages_to_record["apt_packages"]:
            apt_package_name = self.get_container_command_output(
                f"dpkg --get-selections | grep -i {apt_package} | awk '{{print $1}}'"
            )
            if apt_package_name:
                imp_apt_packages.append(apt_package_name.replace("\n", " & "))

        return imp_apt_packages
