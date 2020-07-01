import os
import sys

from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


if sys.version_info.major == 2:
    raise EnvironmentError("This package requires Python 3.6.9 or above.")


setup(
    name="DLCScheduler",
    version="0.1",
    packages=["job_requester", "log_return"],
    install_requires=["boto3", "botocore", "pytest"],
)
