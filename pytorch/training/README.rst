
===========================
SageMaker PyTorch Container
===========================

SageMaker PyTorch Container is an open source library for making the
PyTorch framework run on Amazon SageMaker.

This repository also contains Dockerfiles which install this library, PyTorch, and dependencies
for building SageMaker PyTorch images.

The SageMaker team uses this repository to build its official PyTorch image. To use this image on SageMaker,
see `Python SDK <https://github.com/aws/sagemaker-python-sdk>`__.
For end users, this repository is typically of interest if you need implementation details for
the official image, or if you want to use it to build your own customized PyTorch image.

For information on running PyTorch jobs on SageMaker: `SageMaker PyTorch Estimators and Models
<https://github.com/aws/sagemaker-python-sdk/tree/master/src/sagemaker/pytorch>`__.

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

-  A Python environment management tool (e.g.
   `PyEnv <https://github.com/pyenv/pyenv>`__,
   `VirtualEnv <https://virtualenv.pypa.io/en/stable/>`__)

Building your image
-------------------

`Amazon SageMaker <https://aws.amazon.com/documentation/sagemaker/>`__
utilizes Docker containers to run all training jobs & inference endpoints.

The Docker images are built from the Dockerfiles specified in
`Docker/ <https://github.com/aws/sagemaker-pytorch-container/tree/master/docker>`__.

The Docker files are grouped based on PyTorch version and separated
based on Python version and processor type.

The Docker images, used to run training & inference jobs, are built from
both corresponding "base" and "final" Dockerfiles.

Base Images
~~~~~~~~~~~

The "base" Dockerfile encompass the installation of the framework and all of the dependencies
needed.

Tagging scheme is based on <PyTorch_version>-<processor>-py<python_version>. (e.g.1.0.0-cpu-py3)

All "final" Dockerfiles build images using base images that use the tagging scheme
above.

If you want to build your base docker image, then use:

::

    # All build instructions assume you're building from the root directory of the sagemaker-pytorch-container.

    # CPU
    docker build -t pytorch-base:<PyTorch_version>-cpu-py<python_version> -f docker/<PyTorch_version>/base/Dockerfile.cpu --build-arg py_version=<python_version> .

    # GPU
    docker build -t pytorch-base:<PyTorch_version>-gpu-py<python_version> -f docker/<PyTorch_version>/base/Dockerfile.gpu --build-arg py_version=<python_version> .

::

    # Example

    # CPU
    docker build -t pytorch-base:1.0.0-cpu-py3 -f docker/1.0.0/base/Dockerfile.cpu --build-arg py_version=3 .

    # GPU
    docker build -t pytorch-base:1.0.0-gpu-py3 -f docker/1.0.0/base/Dockerfile.gpu --build-arg py_version=3 .

Final Images
~~~~~~~~~~~~

The "final" Dockerfiles encompass the installation of the SageMaker specific support code.

All "final" Dockerfiles use `base images for building <https://github.com/aws/sagemaker-pytorch-container/blob/master/docker/1.0.0/final/Dockerfile.cpu#L2>`__.

These "base" images are specified with the naming convention of
pytorch-base:<PyTorch_version>-<processor>-py<python_version>.

Before building "final" images:

Build your "base" image. Make sure it is named and tagged in accordance with your "final"
Dockerfile.


::

    # Create the SageMaker PyTorch Container Python package.
    cd sagemaker-pytorch-container
    python setup.py bdist_wheel

If you want to build "final" Docker images, then use:

::

    # All build instructions assume you're building from the root directory of the sagemaker-pytorch-container.

    # CPU
    docker build -t <image_name>:<tag> -f docker/<PyTorch_version>/final/Dockerfile.cpu --build-arg py_version=<python_version> .

    # GPU
    docker build -t <image_name>:<tag> -f docker/<PyTorch_version>/final/Dockerfile.gpu --build-arg py_version=<python_version> .

::

    # Example

    # CPU
    docker build -t preprod-pytorch:1.0.0-cpu-py3 -f docker/1.0.0/final/Dockerfile.cpu --build-arg py_version=3 .

    # GPU
    docker build -t preprod-pytorch:1.0.0-gpu-py3 -f docker/1.0.0/final/Dockerfile.gpu --build-arg py_version=3 .


Running the tests
-----------------

Running the tests requires installation of the SageMaker PyTorch Container code and its test
dependencies.

::

    git clone https://github.com/aws/sagemaker-pytorch-container.git
    cd sagemaker-pytorch-container
    pip install -e .[test]

Tests are defined in
`test/ <https://github.com/aws/sagemaker-pytorch-container/tree/master/test>`__
and include unit, local integration, and SageMaker integration tests.

Unit Tests
~~~~~~~~~~

If you want to run unit tests, then use:

::

    # All test instructions should be run from the top level directory

    pytest test/unit

    # or you can use tox to run unit tests as well as flake8 and code coverage

    tox


Local Integration Tests
~~~~~~~~~~~~~~~~~~~~~~~

Running local integration tests require `Docker <https://www.docker.com/>`__ and `AWS
credentials <https://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/setup-credentials.html>`__,
as the local integration tests make calls to a couple AWS services. The local integration tests and
SageMaker integration tests require configurations specified within their respective
`conftest.py <https://github.com/aws/sagemaker-pytorch-container/blob/master/test/conftest.py>`__.

Local integration tests on GPU require `Nvidia-Docker <https://github.com/NVIDIA/nvidia-docker>`__.

Before running local integration tests:

#. Build your Docker image.
#. Pass in the correct pytest arguments to run tests against your Docker image.

If you want to run local integration tests, then use:

::

    # Required arguments for integration tests are found in test/conftest.py

    pytest test/integration/local --docker-base-name <your_docker_image> \
                      --tag <your_docker_image_tag> \
                      --py-version <2_or_3> \
                      --framework-version <PyTorch_version> \
                      --processor <cpu_or_gpu>

::

    # Example
    pytest test/integration/local --docker-base-name preprod-pytorch \
                      --tag 1.0 \
                      --py-version 3 \
                      --framework-version 1.0.0 \
                      --processor cpu

SageMaker Integration Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~

SageMaker integration tests require your Docker image to be within an `Amazon ECR repository <https://docs
.aws.amazon.com/AmazonECS/latest/developerguide/ECS_Console_Repositories.html>`__.

The Docker base name is your `ECR repository namespace <https://docs.aws.amazon
.com/AmazonECR/latest/userguide/Repositories.html>`__.

The instance type is your specified `Amazon SageMaker Instance Type
<https://aws.amazon.com/sagemaker/pricing/instance-types/>`__ that the SageMaker integration test will run on.

Before running SageMaker integration tests:

#. Build your Docker image.
#. Push the image to your ECR repository.
#. Pass in the correct pytest arguments to run tests on SageMaker against the image within your ECR repository.

If you want to run a SageMaker integration end to end test on `Amazon
SageMaker <https://aws.amazon.com/sagemaker/>`__, then use:

::

    # Required arguments for integration tests are found in test/conftest.py

    pytest test/integration/sagemaker --aws-id <your_aws_id> \
                           --docker-base-name <your_docker_image> \
                           --instance-type <amazon_sagemaker_instance_type> \
                           --tag <your_docker_image_tag> \

::

    # Example
    pytest test/integration/sagemaker --aws-id 12345678910 \
                           --docker-base-name preprod-pytorch \
                           --instance-type ml.m4.xlarge \
                           --tag 1.0

Contributing
------------

Please read
`CONTRIBUTING.md <https://github.com/aws/sagemaker-pytorch-container/blob/master/CONTRIBUTING.md>`__
for details on our code of conduct, and the process for submitting pull
requests to us.

License
-------

SageMaker PyTorch Container is licensed under the Apache 2.0 License. It is copyright 2018 Amazon
.com, Inc. or its affiliates. All Rights Reserved. The license is available at:
http://aws.amazon.com/apache2.0/
