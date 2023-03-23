import contextlib
import os
import logging
import pytest

from ..sagemaker import util

logger = logging.getLogger(__name__)


@pytest.fixture(params=os.environ["TEST_VERSIONS"].split(","))
def version(request):
    return request.param


@pytest.fixture(scope="session")
def repo(request):
    return request.config.getoption("--repo") or "sagemaker-tensorflow-serving"


@pytest.fixture
def processor(request, instance_type):
    return request.config.getoption("--processor") or (
        "gpu"
        if instance_type.startswith("ml.p") or instance_type.startswith("ml.g")
        else "cpu"
    )


@pytest.fixture
def tag(request, version, instance_type, processor):
    if request.config.getoption("--tag"):
        return request.config.getoption("--tag")
    return f"{version}-{processor}"


@pytest.fixture
def image_uri(registry, region, repo, tag):
    return util.image_uri(registry, region, repo, tag)


@pytest.fixture(params=os.environ["TEST_INSTANCE_TYPES"].split(","))
def instance_type(request, region):
    return request.param


@pytest.fixture(scope="module")
def accelerator_type():
    return None


@pytest.fixture(scope="session")
def tfs_model(region, boto_session):
    return util.find_or_put_model_data(region, boto_session, "data/tfs-model.tar.gz")


def _create_endpoint(sagemaker_client, model_name, instance_type, accelerator_type=None):

    logger.info("creating endpoint %s", model_name)
    
    production_variants = [{
        "VariantName": "AllTraffic",
        "ModelName": model_name,
        "InitialInstanceCount": 1,
        "InstanceType": instance_type
    }]

    if accelerator_type:
        production_variants[0]["AcceleratorType"] = accelerator_type

    sagemaker_client.create_endpoint_config(EndpointConfigName=model_name, ProductionVariants=production_variants)
    sagemaker_client.create_endpoint(EndpointName=model_name, EndpointConfigName=model_name)
    try:
        sagemaker_client.get_waiter("endpoint_in_service").wait(EndpointName=model_name)
    finally:
        status = sagemaker_client.describe_endpoint(EndpointName=model_name)["EndpointStatus"]
        if status != "InService":
            raise ValueError("failed to create endpoint {}".format(model_name))

        # delete endpoint
        logger.info("deleting endpoint and endpoint config %s", model_name)
        sagemaker_client.delete_endpoint(EndpointName=model_name)
        sagemaker_client.delete_endpoint_config(EndpointConfigName=model_name)
        


@pytest.mark.model("unknown_model")
def test_endpoint(boto_session, sagemaker_client, model_name, tfs_model, image_uri, instance_type, accelerator_type):
    
    with util.sagemaker_model(boto_session, sagemaker_client, image_uri, model_name, tfs_model):
        _create_endpoint(sagemaker_client, model_name, instance_type, accelerator_type)
        
