import os
from glob import glob
from os.path import basename
from os.path import splitext
import sys

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


if sys.version_info.major == 2:
    raise EnvironmentError("This package requires Python 3.6.9 or above.")


setup(name='Scheduler',
      version='0.1',
      packages=["JobRequester", "LogReturn"],
      install_requires=[
          "boto3",
          "botocore",
          "pytest"
      ]
      )
