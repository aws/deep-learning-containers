#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.
from __future__ import absolute_import
import sagemaker
from sagemaker.tensorflow import TensorFlow

def invoke_tensorflow_estimator(ecr_image, n_virginia_ecr_image, sagemaker_session, n_virginia_sagemaker_session, estimator_parameter, multi_region_support):
    try:
        tensorflow = TensorFlow(
            image_uri=ecr_image,
            sagemaker_session=sagemaker_session,
            **estimator_parameter
            )
    except Exception as e:
        if multi_region_support and type(e) == sagemaker.exceptions.UnexpectedStatusException and "Capacity Error" in str(e):
            tensorflow = TensorFlow(
                image_uri=n_virginia_ecr_image,
                sagemaker_session=n_virginia_sagemaker_session,
                **estimator_parameter
            )
        else:
            raise e
    return tensorflow
