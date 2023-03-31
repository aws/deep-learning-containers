import os
import logging
import pytest
from . import utils

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def test_endpoint(boto_session, sagemaker_client, tfs_model, image_uri, instance_type):
    endpoint_name = utils.create_endpoint_name("sagemaker-mxnet-serving")
    with utils.sagemaker_model(boto_session, sagemaker_client, image_uri, endpoint_name, tfs_model):
        _create_endpoint(sagemaker_client, endpoint_name, instance_type)


def _create_endpoint(sagemaker_client, endpoint_name, instance_type):
    
    LOGGER.info("creating endpoint %s", endpoint_name)

    production_variants = [{
        "VariantName": "AllTraffic",
        "ModelName": endpoint_name,
        "InitialInstanceCount": 1,
        "InstanceType": instance_type
    }]

    sagemaker_client.create_endpoint_config(EndpointConfigName=endpoint_name, ProductionVariants=production_variants)
    sagemaker_client.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_name)
    try:
        sagemaker_client.get_waiter("endpoint_in_service").wait(EndpointName=endpoint_name)
    finally:
        status = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
        if status != "InService":
            raise ValueError("failed to create endpoint {}".format(endpoint_name))

        # delete endpoint
        LOGGER.info("deleting endpoint and endpoint config %s", endpoint_name)
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)