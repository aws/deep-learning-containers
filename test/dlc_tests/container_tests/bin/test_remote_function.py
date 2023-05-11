import sys
import logging
import boto3
import sagemaker
from sagemaker.remote_function import remote

LOGGER = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

DEFAULT_REGION = "us-west-2"

sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=DEFAULT_REGION))


@remote(
    role="SageMakerRole",
    instance_type="ml.m5.xlarge",
    sagemaker_session=sagemaker_session,
)
def dlc_remote_function_test_divide(x, y):
    return x / y


def main():
    assert dlc_remote_function_test_divide(8, 4) == 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
