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
from .... import get_ecr_image, get_sagemaker_session
import sagemaker
from sagemaker.pytorch import PyTorch
import time

def upload_s3_data(estimator, path, key_prefix):

    estimator.sagemaker_session.default_bucket()
    inputs = estimator.sagemaker_session.upload_data(
        path=path,
        key_prefix=key_prefix)
    return inputs

def invoke_pytorch_estimator(pdx_ecr_image, sagemaker_region, estimator_parameter, inputs=None, disable_sm_profiler=False, upload_s3_data_args=None, job_name=None):

    RETRY = 2
    DELAY = 300
    for _ in range(RETRY):
        for region in sagemaker_region:
            sagemaker_session = get_sagemaker_session(region)
            ecr_image = get_ecr_image(pdx_ecr_image, region) if region != "us-west-2" else pdx_ecr_image
            try:
                pytorch = PyTorch(
                    image_uri=ecr_image,
                    sagemaker_session=sagemaker_session,
                    **estimator_parameter
                    )

                if sagemaker_session.boto_region_name in ('cn-north-1', 'cn-northwest-1'):
                    pytorch.disable_profiler = True

                if upload_s3_data_args:
                    training_input = upload_s3_data(pytorch, **upload_s3_data_args)
                    inputs = {'training': training_input}

                pytorch.fit(inputs=inputs, job_name=job_name)
                return pytorch, sagemaker_session

            except Exception as e:
                if type(e) == sagemaker.exceptions.UnexpectedStatusException and "CapacityError" in str(e):
                    continue
                else:
                    raise e

        time.sleep(DELAY)
