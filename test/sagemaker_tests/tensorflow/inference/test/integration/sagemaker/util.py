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
import contextlib
import json
import logging
import os

import boto3
import botocore
import random
import time

logger = logging.getLogger(__name__)
BATCH_CSV = os.path.join('data', 'batch.csv')


def _botocore_resolver():
    """
    Get the DNS suffix for the given region.
    :return: endpoint object
    """
    loader = botocore.loaders.create_loader()
    return botocore.regions.EndpointResolver(loader.load_data('endpoints'))


def get_ecr_registry(account, region):
    """
    Get prefix of ECR image URI
    :param account: Account ID
    :param region: region where ECR repo exists
    :return: AWS ECR registry
    """
    endpoint_data = _botocore_resolver().construct_endpoint('ecr', region)
    return '{}.dkr.{}'.format(account, endpoint_data['hostname'])


def image_uri(registry, region, repo, tag):
    ecr_registry = get_ecr_registry(registry, region)
    return '{}/{}:{}'.format(ecr_registry, repo, tag)


def _execution_role(boto_session):
    return boto_session.resource('iam').Role('SageMakerRole').arn


@contextlib.contextmanager
def sagemaker_model(boto_session, sagemaker_client, image_uri, model_name, model_data):
    model = sagemaker_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=_execution_role(boto_session),
        PrimaryContainer={
            'Image': image_uri,
            'ModelDataUrl': model_data
        })

    try:
        yield model
    finally:
        logger.info('deleting model %s', model_name)
        sagemaker_client.delete_model(ModelName=model_name)


def _production_variants(model_name, instance_type, accelerator_type):
    production_variants = [{
        'VariantName': 'AllTraffic',
        'ModelName': model_name,
        'InitialInstanceCount': 1,
        'InstanceType': instance_type
    }]

    if accelerator_type:
        production_variants[0]['AcceleratorType'] = accelerator_type

    return production_variants


def _test_bucket(region, boto_session):
    domain_suffix = '.cn' if region in ('cn-north-1', 'cn-northwest-1') else ''
    sts_regional_endpoint = 'https://sts.{}.amazonaws.com{}'.format(region, domain_suffix)
    sts = boto_session.client(
        'sts',
        region_name=region,
        endpoint_url=sts_regional_endpoint
    )
    account = sts.get_caller_identity()['Account']
    return 'sagemaker-{}-{}'.format(region, account)


def find_or_put_model_data(region, boto_session, local_path):
    model_file = os.path.basename(local_path)

    bucket = _test_bucket(region, boto_session)
    key = 'test-tfs/{}'.format(model_file)

    s3 = boto_session.client('s3', region)

    try:
        s3.head_bucket(Bucket=bucket)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] != '404':
            raise

        # bucket doesn't exist, create it
        if region == 'us-east-1':
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(Bucket=bucket,
                             CreateBucketConfiguration={'LocationConstraint': region})

    try:
        s3.head_object(Bucket=bucket, Key=key)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] != '404':
            raise

        # file doesn't exist - upload it
        s3.upload_file(local_path, bucket, key)

    return 's3://{}/{}'.format(bucket, key)


@contextlib.contextmanager
def sagemaker_endpoint(sagemaker_client, model_name, instance_type, accelerator_type=None):
    logger.info('creating endpoint %s', model_name)

    # Add jitter so we can run tests in parallel without running into service side limits.
    delay = round(random.random()*5, 3)
    logger.info('waiting for {} seconds'.format(delay))
    time.sleep(delay)

    production_variants = _production_variants(model_name, instance_type, accelerator_type)
    sagemaker_client.create_endpoint_config(EndpointConfigName=model_name,
                                            ProductionVariants=production_variants)

    sagemaker_client.create_endpoint(EndpointName=model_name, EndpointConfigName=model_name)

    try:
        sagemaker_client.get_waiter('endpoint_in_service').wait(EndpointName=model_name)
    finally:
        status = sagemaker_client.describe_endpoint(EndpointName=model_name)['EndpointStatus']
        if status != 'InService':
            raise ValueError('failed to create endpoint {}'.format(model_name))

    try:
        yield model_name  # return the endpoint name
    finally:
        logger.info('deleting endpoint and endpoint config %s', model_name)
        sagemaker_client.delete_endpoint(EndpointName=model_name)
        sagemaker_client.delete_endpoint_config(EndpointConfigName=model_name)


def _create_transform_job_request(model_name, batch_output, batch_input, instance_type):
    return {
        'TransformJobName': model_name,
        'ModelName': model_name,
        'BatchStrategy': 'MultiRecord',
        'TransformOutput': {
            'S3OutputPath': batch_output
        },
        'TransformInput': {
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': batch_input
                }
            },
            'ContentType': 'text/csv',
            'SplitType': 'Line',
            'CompressionType': 'None'
        },
        'TransformResources': {
            'InstanceType': instance_type,
            'InstanceCount': 1
        }
    }


def _read_batch_output(region, boto_session, bucket, model_name):
    s3 = boto_session.client('s3', region)
    output_file = '/tmp/{}.out'.format(model_name)
    s3.download_file(bucket, 'output/{}/batch.csv.out'.format(model_name), output_file)
    return json.loads(open(output_file, 'r').read())['predictions']


def _wait_for_transform_job(region, boto_session, sagemaker_client, model_name, poll, timeout):
    status = sagemaker_client.describe_transform_job(TransformJobName=model_name)['TransformJobStatus']
    job_runtime = 0
    while status == 'InProgress':

        logger.info('Waiting for batch transform job {} to finish'.format(model_name))
        time.sleep(poll)
        job_runtime += poll
        if job_runtime > timeout:
            raise ValueError('Batch transform job {} exceeded maximum runtime {} seconds'.format(model_name, timeout))

        status = sagemaker_client.describe_transform_job(TransformJobName=model_name)['TransformJobStatus']
        if status == 'Completed':
            return _read_batch_output(region=region,
                                      boto_session=boto_session,
                                      bucket=_test_bucket(region, boto_session),
                                      model_name=model_name)
        if status == 'Failed':
            raise ValueError('Failed to execute batch transform job {}'.format(model_name))
        if status in ['Stopped', 'Stopping']:
            raise ValueError('Batch transform job {} was stopped'.format(model_name))


def run_batch_transform_job(region, boto_session, model_data, image_uri,
                            sagemaker_client, model_name,
                            instance_type):

    with sagemaker_model(boto_session, sagemaker_client, image_uri, model_name, model_data):
        batch_input = find_or_put_model_data(region, boto_session, BATCH_CSV)
        bucket = _test_bucket(region, boto_session)
        batch_output = 's3://{}/output/{}'.format(bucket, model_name)

        request = _create_transform_job_request(
            model_name=model_name, batch_input=batch_input,
            batch_output=batch_output, instance_type=instance_type)
        sagemaker_client.create_transform_job(**request)

    return _wait_for_transform_job(
               region=region,
               boto_session=boto_session,
               sagemaker_client=sagemaker_client,
               model_name=model_name,
               poll=5, # seconds
               timeout=600) # seconds


def invoke_endpoint(sagemaker_runtime_client, endpoint_name, input_data):
    response = sagemaker_runtime_client.invoke_endpoint(EndpointName=endpoint_name,
                                                        ContentType='application/json',
                                                        Body=json.dumps(input_data))
    result = json.loads(response['Body'].read().decode())
    assert result['predictions'] is not None
    return result


def create_and_invoke_endpoint(boto_session, sagemaker_client, sagemaker_runtime_client,
                               model_name, model_data, image_uri, instance_type, accelerator_type,
                               input_data):
    with sagemaker_model(boto_session, sagemaker_client, image_uri, model_name, model_data):
        with sagemaker_endpoint(sagemaker_client, model_name, instance_type, accelerator_type):
            return invoke_endpoint(sagemaker_runtime_client, model_name, input_data)
