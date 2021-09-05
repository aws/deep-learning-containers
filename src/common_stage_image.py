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

from image import DockerImage
from context import Context
from utils import generate_safety_report_for_image, get_root_folder_path

import os


class CommonStageImage(DockerImage):
    """
    This class is especially designed to handle the build process for CommonStageImages.
    All the functionality - either safety scan report, ecr scan report, etc. - that is especially 
    required to run the miscellaneous_dockerfiles/Dockerfile.common should go into this file. As of now,
    this class takes care of generating a safety report from a pre_push_image and then uses this 
    safety report for creating a context for Dockerfile.common 
    """

    def update_pre_build_configuration(self):
        """
        Conducts all the pre-build configurations from the parent class and then conducts
        Safety Scan on the images generated in previous stage builds. The safety scan generates
        the safety_report which is then copied into the image. 
        """
        # Call the update_pre_build_configuration steps from the parent class
        super(CommonStageImage, self).update_pre_build_configuration()
        # Generate safety scan report for the first stage image and add the file to artifacts
        first_stage_image_uri = self.build_args["PRE_PUSH_IMAGE"]
        processed_image_uri = first_stage_image_uri.replace(".", "-").replace("/", "-").replace(":", "-")
        image_name = self.name
        tarfile_name_for_context = f"{processed_image_uri}-{image_name}"
        storage_file_path = os.path.join(
            os.sep, get_root_folder_path(), "src", f"{tarfile_name_for_context}_safety_report.json",
        )
        generate_safety_report_for_image(first_stage_image_uri, storage_file_path=storage_file_path)
        self.context = self.generate_common_stage_context(storage_file_path, tarfile_name=tarfile_name_for_context)

    def generate_common_stage_context(self, safety_report_path, tarfile_name="common-stage-file"):
        """
        For CommonStageImage, build context is built once the safety report is generated. This is because
        the Dockerfile.common uses this safety report to COPY the report into the image.
        """
        artifacts = {
            "safety_report": {"source": safety_report_path, "target": "safety_report.json",},
            "dockerfile": {
                "source": os.path.join(
                    os.sep, get_root_folder_path(), "miscellaneous_dockerfiles", "Dockerfile.common",
                ),
                "target": "Dockerfile",
            },
        }

        artifact_root = os.path.join(os.sep, get_root_folder_path(), "src")
        return Context(artifacts, context_path=f"build/{tarfile_name}.tar.gz", artifact_root=artifact_root,)
