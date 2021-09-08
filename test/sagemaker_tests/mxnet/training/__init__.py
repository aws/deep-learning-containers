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

import sagemaker
from sagemaker.mxnet.estimator import MXNet as MXNet_origin


class MXNet(MXNet_origin):
    def __init__(self, image_uri, sagemaker_regions, **kwargs):
        from ... import get_sagemaker_session
        super().__init__(image_uri=image_uri, **kwargs)
        self.image_uri = image_uri
        self.sagemaker_regions = sagemaker_regions
        self.sagemaker_session = get_sagemaker_session(self.sagemaker_regions[0])

    def fit(self, inputs=None, wait=True, logs="All", job_name=None, experiment_config=None, **kwargs):
        from ... import get_ecr_image_region, get_sagemaker_session, get_ecr_image

        for region in self.sagemaker_regions:
            self.sagemaker_session = get_sagemaker_session(region)
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
