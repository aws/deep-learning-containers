import boto3
import pytest
import logging
from botocore.config import Config

from . import utils

logger = logging.getLogger(__name__)
logging.getLogger('boto').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.INFO)

def pytest_addoption(parser):
    parser.addoption('--account-id', default=None)
    parser.addoption('--registry', default=None)
    parser.addoption('--repository', default=None)
    parser.addoption('--region', default="us-west-2")
    parser.addoption('--tag', default=None)
    parser.addoption('--py-version', default='py3')
    parser.addoption('--instance-type', default='ml.p2.xlarge')
    parser.addoption('--framework-version', default=None)
    parser.addoption('--sagemaker-region', default='us-west-2')


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "sagemaker_canary(message): mark test to run as a part of sagemaker canary tests."
    )

@pytest.fixture(scope='session')
def repository(request):
    return request.config.getoption('--repository')


@pytest.fixture(scope='session')
def region(request):
    return request.config.getoption('--region')


@pytest.fixture(scope='session')
def framework_version(request):
    return request.config.getoption('--framework-version')


@pytest.fixture(scope='session')
def account_id(request):
    return request.config.getoption('--account-id')


@pytest.fixture(scope='session')
def registry(request):
    return request.config.getoption('--registry')


@pytest.fixture(scope='session')
def tag(request):
    return request.config.getoption('--tag')


@pytest.fixture(scope='session')
def py_version(request):
    return request.config.getoption('--py-version')


@pytest.fixture(scope='session')
def instance_type(request):
    return request.config.getoption('--instance-type')


@pytest.fixture(scope='session')
def image_uri(registry, region, repository, tag):
    ecr_registry = utils.get_ecr_registry(registry, region)
    return f'{ecr_registry}/{repository}:{tag}'

@pytest.fixture(scope="session")
def tfs_model(region, boto_session):
    return utils.find_or_put_model_data(region, boto_session, "tfs-model.tar.gz")


@pytest.fixture(scope='session')
def model_dir(request):
    return "tfs-model.tar.gz"


@pytest.fixture(scope='session')
def model_script(request):
    return "empty_module.py"


@pytest.fixture(scope='session')
def sagemaker_session(region):
    return Session(boto_session=boto3.Session(region_name=region))


@pytest.fixture(scope='session')
def sagemaker_region(request):
    sagemaker_region = request.config.getoption('--sagemaker-region')
    return sagemaker_region


@pytest.fixture(scope='session')
def sagemaker_local_session(sagemaker_region):
    return LocalSession(boto_session=boto3.Session(region_name=sagemaker_region))


@pytest.fixture(scope="session")
def boto_session(sagemaker_region):
    return boto3.Session(region_name=sagemaker_region)


@pytest.fixture(scope="session")
def sagemaker_client(boto_session):
    return boto_session.client("sagemaker", config=Config(retries={"max_attempts": 10}))