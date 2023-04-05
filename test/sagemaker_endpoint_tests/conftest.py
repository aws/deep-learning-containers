import os
import boto3
import pytest
import logging
import utils as utils

logger = logging.getLogger(__name__)
logging.getLogger('boto').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.INFO)

RESOURCE_PATH = os.path.abspath(os.path.dirname(__file__))

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
    parser.addoption(
        "-F",
        action="store",
        metavar="NAME",
        help="only run tests matching the framework NAME.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "sagemaker_endpoint(name): mark test to run as a part of sagemaker canary tests."
    )


def pytest_runtest_setup(item):
    envnames = [mark.args[0] for mark in item.iter_markers(name="sagemaker_endpoint")]
    if envnames:
        if item.config.getoption("-F") not in envnames:
            pytest.skip(f"test requires sagemaker_endpoint in {envnames}")


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



@pytest.fixture(scope='session')
def mx_model_dir(request):
    return "mxnet/model.tar.gz"


@pytest.fixture(scope='session')
def mx_model_script(request):
    return "mxnet/empty_module.py"


@pytest.fixture(scope='session')
def pt_model_dir(request):
    return "pytorch/model.tar.gz"


@pytest.fixture(scope='session')
def pt_model_script(request):
    return "pytorch/mnist.py"


@pytest.fixture(scope='session')
def tf_model_dir(request):
    return "tensorflow/tfs-model.tar.gz"


@pytest.fixture(scope='session')
def tf_model_script(request):
    return "tensorflow/empty_module.py"


@pytest.fixture(scope='session')
def sagemaker_session(region):
    return Session(boto_session=boto3.Session(region_name=region))


@pytest.fixture(scope='session')
def sagemaker_region(request):
    return request.config.getoption('--sagemaker-region')
