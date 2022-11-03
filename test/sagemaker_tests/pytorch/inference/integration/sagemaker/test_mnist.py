# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import os

import numpy as np
import pytest
import sagemaker
from sagemaker.pytorch import PyTorchModel

#boto3 imports
import boto3
from datetime import datetime, timedelta
import time
import json

from ...integration import model_cpu_dir, mnist_cpu_script, mnist_gpu_script, model_eia_dir, mnist_eia_script
from ...integration.sagemaker.timeout import timeout_and_delete_endpoint
from .... import invoke_pytorch_helper_function


@pytest.mark.model("mnist")
@pytest.mark.processor("cpu")
@pytest.mark.cpu_test
def test_mnist_distributed_cpu(framework_version, ecr_image, instance_type, sagemaker_regions):
    instance_type = instance_type or 'ml.c4.xlarge'
    model_dir = os.path.join(model_cpu_dir, 'model_mnist.tar.gz')
    function_args = {
            'framework_version': framework_version,
            'instance_type': instance_type,
            'model_dir': model_dir,
            'mnist_script': mnist_cpu_script,
            'verify_logs': True

        }
    invoke_pytorch_helper_function(ecr_image, sagemaker_regions, _test_mnist_distributed, function_args)



@pytest.mark.model("mnist")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
def test_mnist_distributed_gpu(framework_version, ecr_image, instance_type, sagemaker_regions):
    instance_type = instance_type or 'ml.p2.xlarge'
    model_dir = os.path.join(model_cpu_dir, 'model_mnist.tar.gz')
    function_args = {
            'framework_version': framework_version,
            'instance_type': instance_type,
            'model_dir': model_dir,
            'mnist_script': mnist_gpu_script

        }
    invoke_pytorch_helper_function(ecr_image, sagemaker_regions, _test_mnist_distributed, function_args)


@pytest.mark.model("mnist")
@pytest.mark.integration("elastic_inference")
@pytest.mark.processor("eia")
@pytest.mark.eia_test
def test_mnist_eia(framework_version, ecr_image, instance_type, accelerator_type, sagemaker_regions):
    instance_type = instance_type or 'ml.c4.xlarge'
    # Scripted model is serialized with torch.jit.save().
    # Inference test for EIA doesn't need to instantiate model definition then load state_dict
    model_dir = os.path.join(model_eia_dir, 'model_mnist.tar.gz')
    function_args = {
            'framework_version': framework_version,
            'instance_type': instance_type,
            'model_dir': model_dir,
            'mnist_script': mnist_eia_script,
            'accelerator_type': accelerator_type,

        }
    invoke_pytorch_helper_function(ecr_image, sagemaker_regions, _test_mnist_distributed, function_args)



def _test_mnist_distributed(
        ecr_image, sagemaker_session, framework_version, instance_type, model_dir, mnist_script, accelerator_type=None, verify_logs= False
):
    endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-pytorch-serving")

    model_data = sagemaker_session.upload_data(
        path=model_dir,
        key_prefix="sagemaker-pytorch-serving/models",
    )

    pytorch = PyTorchModel(
        model_data=model_data,
        role='SageMakerRole',
        entry_point=mnist_script,
        framework_version=framework_version,
        image_uri=ecr_image,
        sagemaker_session=sagemaker_session,
    )

    with timeout_and_delete_endpoint(endpoint_name, sagemaker_session, minutes=30):
        # Use accelerator type to differentiate EI vs. CPU and GPU. Don't use processor value
        if accelerator_type is not None:
            predictor = pytorch.deploy(
                initial_instance_count=1,
                instance_type=instance_type,
                accelerator_type=accelerator_type,
                endpoint_name=endpoint_name,
            )
        else:
            predictor = pytorch.deploy(
                initial_instance_count=1,
                instance_type=instance_type,
                endpoint_name=endpoint_name,
            )

        batch_size = 100
        data = np.random.rand(batch_size, 1, 28, 28).astype(np.float32)
        output = predictor.predict(data)

        assert output.shape == (batch_size, 10)

        #  Check for Cloudwatch logs if true
        if verify_logs:
            _check_for_cloudwatch_logs(endpoint_name)

def _check_for_cloudwatch_logs(endpoint_name):
    print('##############################################################################')
    print('##############################################################################')
    print('##############################################################################')
    print('INFO: Checking logs for the endpoint: /aws/sagemaker/Endpoints/'+endpoint_name)
    print('##############################################################################')
    print('##############################################################################')
    print('##############################################################################')
    print('##############################################################################')
    client=boto3.client('logs')
    query = "fields @timestamp | sort @timestamp desc | limit 2";
    recordsAvailable=False;
    response = None;
    
    if not recordsAvailable:
        for i in range(3):        
            start_query_response = client.start_query(
            logGroupName='/aws/sagemaker/Endpoints/'+endpoint_name,
            startTime=int((datetime.today() - timedelta(minutes=30)).timestamp()),
            endTime=int(datetime.now().timestamp()),
            queryString=query,
        )
        query_id = start_query_response['queryId']
        print(f'Query ID: {query_id}')    
        while response == None or response['status'] == 'Running':
            print('Waiting for query to complete ...')
            time.sleep(1)
            response = client.get_query_results(
                queryId=query_id
            )
        recordsAvailable=bool(response['results'])    
        print(response)
        time.sleep(60)



    # start_query_response = client.start_query(
    #     logGroupName='/aws/sagemaker/Endpoints/'+endpoint_name,
    #     startTime=int((datetime.today() - timedelta(minutes=30)).timestamp()),
    #     endTime=int(datetime.now().timestamp()),
    #     queryString=query,
    # )

    # recordsAvailable=False;
    # response = None;
    # query_id = start_query_response['queryId']
    # response = None
    # print('INFO: Querying Cloudwatch for log events...')
    # while response == None or response['status'] == 'Running':
    #     print('Waiting for query to complete ...')
    #     time.sleep(1)
    #     response = client.get_query_results(
    #         queryId=query_id
    #     )        

    #recordsAvailable=bool(response['results'])   

    if not recordsAvailable:
        print("Exception... No cloudwatch log results!!")
        raise Exception('Exception: No cloudwatch events getting logged for the group /aws/sagemaker/Endpoints/'+endpoint_name)
    else:    
        print('INFO: Most recently logged event was found at @timestamp -- '+response['results'][0][0]['value'])

    print('##############################################################################')
    print('##############################################################################')
    print('##############################################################################')    


