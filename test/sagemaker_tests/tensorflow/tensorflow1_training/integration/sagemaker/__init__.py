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
from sagemaker.tuner import HyperparameterTuner
from sagemaker import TrainingInput

def initate_tensorflow_estimator(ecr_image, sagemaker_session, estimator_parameter):
    
    tensorflow = TensorFlow(
        image_uri=ecr_image,
        sagemaker_session=sagemaker_session,
        **estimator_parameter
        )
        
    return tensorflow

def upload_s3_data(estimator, path, key_prefix):
    inputs = estimator.sagemaker_session.upload_data(
        path=path,
        key_prefix=key_prefix)
    return inputs

def disable_sm_profiler(estimator, region):
    """Disable SMProfiler feature for China regions
    """

    if region in ('cn-north-1', 'cn-northwest-1'):
        estimator.disable_profiler = True
    return estimator

def hyper_parameter_tuner(estimator, objective_metric_name, hyperparameter_ranges, metric_definitions, max_jobs, max_parallel_jobs):
    tuner = HyperparameterTuner(estimator,
                            objective_metric_name,
                            hyperparameter_ranges,
                            metric_definitions,
                            max_jobs=max_jobs,
                            max_parallel_jobs=max_parallel_jobs)
    return tuner

def invoke_tensorflow_estimator(ecr_image, n_virginia_ecr_image, sagemaker_session, n_virginia_sagemaker_session, estimator_parameter, multi_region_support, job_name=None, inputs=None, disable_sm_profiler=False, upload_s3_data_args=None, training_input_args=None, hyperparameter_args=None):
    try:
        estimator = initate_tensorflow_estimator(ecr_image, sagemaker_session, estimator_parameter)
        #TODO: Add it in a function??
        if disable_sm_profiler:
            disable_sm_profiler_args = {
                'region': sagemaker_session.boto_region_name
            }
            estimator = disable_sm_profiler(estimator, **disable_sm_profiler_args)

        if upload_s3_data_args:
            inputs = upload_s3_data(estimator, **upload_s3_data_args)
        
        if hyperparameter_args:
            estimator = hyper_parameter_tuner(estimator, **hyperparameter_args)
        
        estimator.fit(inputs=inputs, job_name=job_name)
        return estimator, sagemaker_session
    except Exception as e:
        if multi_region_support and type(e) == sagemaker.exceptions.UnexpectedStatusException and "CapacityError" in str(e):
            n_virginia_estimator = initate_tensorflow_estimator(n_virginia_ecr_image, n_virginia_sagemaker_session, estimator_parameter)
            #TODO: Add it in a function??
            if disable_sm_profiler:
                disable_sm_profiler_args = {
                    'region': sagemaker_session.boto_region_name
                }
                estimator = disable_sm_profiler(estimator, **disable_sm_profiler_args)

            if upload_s3_data_args:
                inputs = upload_s3_data(estimator, **upload_s3_data_args)
            
            if hyperparameter_args:
                estimator = hyper_parameter_tuner(estimator, **hyperparameter_args)

            n_virginia_estimator.fit(inputs=inputs, job_name=job_name)
            return n_virginia_estimator, n_virginia_sagemaker_session
        else:
            raise e
