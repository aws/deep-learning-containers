# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import sys
import sagemaker
import logging

from sagemaker.mxnet.estimator import MXNet as MXNet

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


class MXNetWrapper(MXNet):
    def __init__(self, image_uri, sagemaker_regions, **kwargs):
        super().__init__(image_uri=image_uri, **kwargs)
        from ... import get_ecr_image_region, get_ecr_image, get_account_id_from_image_uri

        self.account_id = get_account_id_from_image_uri(image_uri)
        self.sagemaker_regions = sagemaker_regions
        LOGGER.info(f"sagemaker_regions - {sagemaker_regions}, \n image_uri - {image_uri},\n ")
        if self.sagemaker_regions[0] != get_ecr_image_region(image_uri):
            self.image_uri = get_ecr_image(image_uri, self.sagemaker_regions[0])
        else:
            self.image_uri = image_uri
        self.sagemaker_session = self.create_sagemaker_session(self.sagemaker_regions[0])
        LOGGER.info(f"sagemaker_regions - {sagemaker_regions}, \n image_uri - {self.image_uri},\n sagemaker_session.default_bucket() - {self.sagemaker_session.default_bucket()}")

    def fit(self, inputs=None, wait=True, logs="All", job_name=None, experiment_config=None, **kwargs):
        from ... import get_ecr_image_region, get_ecr_image

        for region in self.sagemaker_regions:
            self.sagemaker_session = self.create_sagemaker_session(region)
            # Upload the image to test region if needed
            if region != get_ecr_image_region(self.image_uri):
                self.image_uri = get_ecr_image(self.image_uri, region)
            try:
                super().fit(
                    inputs=inputs,
                    wait=wait,
                    logs=logs,
                    job_name=job_name,
                    experiment_config=experiment_config,
                    **kwargs
                )
                return
            except sagemaker.exceptions.UnexpectedStatusException as e:
                if "CapacityError" in str(e):
                    continue
                else:
                    raise e

    def create_sagemaker_session(self, region):
        from ... import get_sagemaker_session
        bucket_name = "sagemaker-{}-{}".format(region, self.account_id)
        return get_sagemaker_session(region, bucket_name)
