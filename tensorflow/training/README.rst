===============================
SageMaker TensorFlow Containers
===============================

SageMaker TensorFlow Containers is an open source library for making the
TensorFlow framework run on `Amazon SageMaker <https://aws.amazon.com/documentation/sagemaker/>`__.

This repository also contains Dockerfiles which install this library, TensorFlow, and dependencies
for building SageMaker TensorFlow images.

For information on running TensorFlow jobs on SageMaker: `Python
SDK <https://github.com/aws/sagemaker-python-sdk>`__.

For notebook examples: `SageMaker Notebook
Examples <https://github.com/awslabs/amazon-sagemaker-examples>`__.

Table of Contents
-----------------

#. `Getting Started <#getting-started>`__
#. `Building your Image <#building-your-image>`__
#. `Running the tests <#running-the-tests>`__

Getting Started
---------------

Prerequisites
~~~~~~~~~~~~~

Make sure you have installed all of the following prerequisites on your
development machine:

- `Docker <https://www.docker.com/>`__

For Testing on GPU
^^^^^^^^^^^^^^^^^^

-  `Nvidia-Docker <https://github.com/NVIDIA/nvidia-docker>`__

Recommended
^^^^^^^^^^^

-  A Python environment management tool. (e.g.
   `PyEnv <https://github.com/pyenv/pyenv>`__,
   `VirtualEnv <https://virtualenv.pypa.io/en/stable/>`__)

Building your Image
-------------------

`Amazon SageMaker <https://aws.amazon.com/documentation/sagemaker/>`__
utilizes Docker containers to run all training jobs & inference endpoints.

The Docker images are built from the Dockerfiles specified in
`Docker/ <https://github.com/aws/sagemaker-tensorflow-containers/tree/master/docker>`__.

The Docker files are grouped based on TensorFlow version and separated
based on Python version and processor type.

The Docker files for TensorFlow 2.0 are available in the
`tf-2 <https://github.com/aws/sagemaker-tensorflow-container/tree/tf-2>`__ branch, in
`docker/2.0.0/ <https://github.com/aws/sagemaker-tensorflow-container/tree/tf-2/docker/2.0.0>`__.

The Docker images, used to run training & inference jobs, are built from
both corresponding "base" and "final" Dockerfiles.

Base Images
~~~~~~~~~~~

The "base" Dockerfile encompass the installation of the framework and all of the dependencies
needed. It is needed before building image for TensorFlow 1.8.0 and before.
Building a base image is not required for images for TensorFlow 1.9.0 and onwards.

Tagging scheme is based on <tensorflow_version>-<processor>-<python_version>. (e.g. 1.4
.1-cpu-py2)

All "final" Dockerfiles build images using base images that use the tagging scheme
above.

If you want to build your "base" Docker image, then use:

::

    # All build instructions assume you're building from the same directory as the Dockerfile.

    # CPU
    docker build -t tensorflow-base:<tensorflow_version>-cpu-<python_version> -f Dockerfile.cpu .

    # GPU
    docker build -t tensorflow-base:<tensorflow_version>-gpu-<python_version> -f Dockerfile.gpu .

::

    # Example

    # CPU
    docker build -t tensorflow-base:1.4.1-cpu-py2 -f Dockerfile.cpu .

    # GPU
    docker build -t tensorflow-base:1.4.1-gpu-py2 -f Dockerfile.gpu .

Final Images
~~~~~~~~~~~~

The "final" Dockerfiles encompass the installation of the SageMaker specific support code.

For images of TensorFlow 1.8.0 and before, all "final" Dockerfiles use `base images for building <https://github
.com/aws/sagemaker-tensorflow-containers/blob/master/docker/1.4.1/final/py2/Dockerfile.cpu#L2>`__.

These "base" images are specified with the naming convention of
tensorflow-base:<tensorflow_version>-<processor>-<python_version>.

Before building "final" images:

Build your "base" image. Make sure it is named and tagged in accordance with your "final"
Dockerfile. Skip this step if you want to build image of Tensorflow Version 1.9.0 and above.

Then prepare the SageMaker TensorFlow Container python package in the image folder like below:

::

    # Create the SageMaker TensorFlow Container Python package.
    cd sagemaker-tensorflow-containers
    python setup.py sdist

    #. Copy your Python package to "final" Dockerfile directory that you are building.
    cp dist/sagemaker_tensorflow_container-<package_version>.tar.gz docker/<tensorflow_version>/final/py2

If you want to build "final" Docker images, for versions 1.6 and above, you will first need to download the appropriate tensorflow pip wheel, then pass in its location as a build argument. These can be obtained from pypi. For example, the files for 1.6.0 are here:

https://pypi.org/project/tensorflow/1.6.0/#files
https://pypi.org/project/tensorflow-gpu/1.6.0/#files

Note that you need to use the tensorflow-gpu wheel when building the GPU image.

Then run:

::

    # All build instructions assumes you're building from the same directory as the Dockerfile.

    # CPU
    docker build -t <image_name>:<tag> --build-arg py_version=<py_version> --build-arg framework_installable=<path to tensorflow binary> -f Dockerfile.cpu .

    # GPU
    docker build -t <image_name>:<tag> --build-arg py_version=<py_version> --build-arg framework_installable=<path to tensorflow binary> -f Dockerfile.gpu .

::

    # Example
    docker build -t preprod-tensorflow:1.6.0-cpu-py2 --build-arg py_version=2
    --build-arg framework_installable=tensorflow-1.6.0-cp27-cp27mu-manylinux1_x86_64.whl -f Dockerfile.cpu .

The dockerfiles for 1.4 and 1.5 build from source instead, so when building those, you don't need to download the wheel beforehand:

::

    # All build instructions assumes you're building from the same directory as the Dockerfile.

    # CPU
    docker build -t <image_name>:<tag> -f Dockerfile.cpu .

    # GPU
    docker build -t <image_name>:<tag> -f Dockerfile.gpu .

::

    # Example

    # CPU
    docker build -t preprod-tensorflow:1.4.1-cpu-py2 -f Dockerfile.cpu .

    # GPU
    docker build -t preprod-tensorflow:1.4.1-gpu-py2 -f Dockerfile.gpu .


Running the tests
-----------------

Running the tests requires installation of the SageMaker TensorFlow Container code and its test
dependencies.

::

    git clone https://github.com/aws/sagemaker-tensorflow-containers.git
    cd sagemaker-tensorflow-containers
    pip install -e .[test]

Tests are defined in
`test/ <https://github.com/aws/sagemaker-tensorflow-containers/tree/master/test>`__
and include unit, integration and functional tests.

Unit Tests
~~~~~~~~~~

If you want to run unit tests, then use:

::

    # All test instructions should be run from the top level directory

    pytest test/unit

Integration Tests
~~~~~~~~~~~~~~~~~

Running integration tests require `Docker <https://www.docker.com/>`__ and `AWS
credentials <https://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/setup-credentials.html>`__,
as the integration tests make calls to a couple AWS services. The integration and functional
tests require configurations specified within their respective
`conftest.py <https://github.com/aws/sagemaker-tensorflow-containers/blob/master/test/integ/conftest.py>`__.

Integration tests on GPU require `Nvidia-Docker <https://github.com/NVIDIA/nvidia-docker>`__.

Before running integration tests:

#. Build your Docker image.
#. Pass in the correct pytest arguments to run tests against your Docker image.

If you want to run local integration tests, then use:

::

    # Required arguments for integration tests are found in test/integ/conftest.py

    pytest test/integ --docker-base-name <your_docker_image> \
                      --tag <your_docker_image_tag> \
                      --framework-version <tensorflow_version> \
                      --processor <cpu_or_gpu>

::

    # Example
    pytest test/integ --docker-base-name preprod-tensorflow \
                      --tag 1.0 \
                      --framework-version 1.4.1 \
                      --processor cpu

Functional Tests
~~~~~~~~~~~~~~~~

Functional tests require your Docker image to be within an `Amazon ECR repository <https://docs
.aws.amazon.com/AmazonECS/latest/developerguide/ECS_Console_Repositories.html>`__.

The Docker-base-name is your `ECR repository namespace <https://docs.aws.amazon
.com/AmazonECR/latest/userguide/Repositories.html>`__.

The instance-type is your specified `Amazon SageMaker Instance Type
<https://aws.amazon.com/sagemaker/pricing/instance-types/>`__ that the functional test will run on.


Before running functional tests:

#. Build your Docker image.
#. Push the image to your ECR repository.
#. Pass in the correct pytest arguments to run tests on SageMaker against the image within your ECR repository.

If you want to run a functional end to end test on `Amazon
SageMaker <https://aws.amazon.com/sagemaker/>`__, then use:

::

    # Required arguments for integration tests are found in test/functional/conftest.py

    pytest test/functional --aws-id <your_aws_id> \
                           --docker-base-name <your_docker_image> \
                           --instance-type <amazon_sagemaker_instance_type> \
                           --tag <your_docker_image_tag> \

::

    # Example
    pytest test/functional --aws-id 12345678910 \
                           --docker-base-name preprod-tensorflow \
                           --instance-type ml.m4.xlarge \
                           --tag 1.0

Contributing
------------

Please read
`CONTRIBUTING.md <https://github.com/aws/sagemaker-tensorflow-containers/blob/master/CONTRIBUTING.md>`__
for details on our code of conduct, and the process for submitting pull
requests to us.

License
-------

SageMaker TensorFlow Containers is licensed under the Apache 2.0 License. It is copyright 2018
Amazon.com, Inc. or its affiliates. All Rights Reserved. The license is available at:
http://aws.amazon.com/apache2.0/
