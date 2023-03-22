import os

import boto3
import pytest

from sagemaker import utils
from sagemaker.tensorflow import TensorFlowModel
from ...integration import get_ecr_registry, RESOURCE_PATH
from ...integration.sagemaker import timeout
from ...... import invoke_sm_helper_function


DEFAULT_HANDLER_PATH = os.path.join(RESOURCE_PATH, 'default_handlers')
MODEL_PATH = os.path.join(DEFAULT_HANDLER_PATH, 'model.tar.gz')
SCRIPT_PATH = os.path.join(DEFAULT_HANDLER_PATH, 'model', 'code', 'empty_module.py')


@pytest.fixture(scope="session")
def framework_version(request):
    return request.config.getoption("--versions")


@pytest.fixture
def instance_type(request, processor):
    provided_instance_type = request.config.getoption('--instance-types')
    default_instance_type = 'ml.c4.xlarge' if processor == 'cpu' else 'ml.p2.xlarge'
    return provided_instance_type if provided_instance_type is not None else default_instance_type


@pytest.fixture(scope="session")
def docker_base_name(request):
    return request.config.getoption("--repo") or "sagemaker-tensorflow-serving"


@pytest.fixture
def processor(request):
    return request.config.getoption("--processor") 


@pytest.fixture
def tag(request, framework_version, instance_type, processor):
    if request.config.getoption("--tag"):
        return request.config.getoption("--tag")
    return f"{version}-{processor}"


@pytest.fixture(scope='session')
def account_id(request):
    return request.config.getoption('--registry')


@pytest.fixture
def ecr_image(account_id, region, docker_base_name, tag):
    registry = get_ecr_registry(account_id, region)
    return f"{registry}/{docker_base_name}:{tag}"


@pytest.fixture(scope='session')
def sagemaker_session(region):
    return Session(boto_session=boto3.Session(region_name=region))


@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
def test_sagemaker_endpoint_gpu(ecr_image, sagemaker_regions, instance_type, framework_version):
    instance_type = instance_type or 'ml.p2.xlarge'
    invoke_sm_helper_function(ecr_image, sagemaker_regions, _test_sagemaker_endpoint_function, instance_type, framework_version)


@pytest.mark.processor("cpu")
@pytest.mark.cpu_test
def test_sagemaker_endpoint_cpu(ecr_image, sagemaker_regions, instance_type, framework_version):
    instance_type = instance_type or 'ml.c4.xlarge'
    invoke_sm_helper_function(ecr_image, sagemaker_regions, _test_sagemaker_endpoint_function, instance_type, framework_version)


def _test_sagemaker_endpoint_function(ecr_image, sagemaker_session, instance_type, framework_version):
    prefix = 'sagemaker-tensorflow-serving/models'
    model_data = sagemaker_session.upload_data(path=MODEL_PATH, key_prefix=prefix)
    model = TensorFlowModel(model_data=model_data,
                            role="SageMakerRole",
                            entry_point=SCRIPT_PATH,
                            framework_version=framework_version,
                            image_uri=ecr_image,
                            sagemaker_session=sagemaker_session,
    )

    endpoint_name = utils.unique_name_from_base("sagemaker-tensorflow-serving")
    with timeout.timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        predictor = model.deploy(initial_instance_count=1, instance_type=instance_type, endpoint_name=endpoint_name)
