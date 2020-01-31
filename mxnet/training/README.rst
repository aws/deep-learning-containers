=========================
SageMaker MXNet Container
=========================

SageMaker MXNet Container is an open-source library for making Docker images for using MXNet to train models on Amazon SageMaker.
For serving images, see `SageMaker MXNet Serving Container <https://github.com/aws/sagemaker-mxnet-serving-container>`__.
For information on running MXNet jobs on Amazon SageMaker, please refer to the `SageMaker Python SDK documentation <https://github.com/aws/sagemaker-python-sdk>`__.

The information in this README is for MXNet versions 1.4.0 and higher.
For versions 0.12.0-1.3.0 (including serving images), see `the previous version of this README <https://github.com/aws/sagemaker-mxnet-container/blob/4f4492ba71ab5210bb0594449d3996f0bc3e5807/README.rst>`__.

-----------------
Table of Contents
-----------------
.. contents::
    :local:

Getting Started
---------------

Prerequisites
~~~~~~~~~~~~~

Make sure you have installed all of the following prerequisites on your development machine:

- `Docker <https://www.docker.com/>`__
- For GPU testing: `nvidia-docker2 <https://github.com/NVIDIA/nvidia-docker>`__

Recommended
^^^^^^^^^^^

-  A Python environment management tool (e.g. `PyEnv <https://github.com/pyenv/pyenv>`__,
   `VirtualEnv <https://virtualenv.pypa.io/en/stable/>`__)

Building Images
---------------

The Dockerfiles in this repository are intended to be used for building Docker images to run training jobs on `Amazon SageMaker <https://aws.amazon.com/documentation/sagemaker/>`__.

The current master branch of this repository contains Dockerfiles and support code for MXNet versions 1.4.0 and higher.
For MXNet version 1.3.0, check out `v2.0.0 of this repository <https://github.com/aws/sagemaker-mxnet-container/releases/tag/v2.0.0>`__.
For MXNet versions 0.12.1-1.2.1, check out `v1.0.0 of this repository <https://github.com/aws/sagemaker-mxnet-container/releases/tag/v1.0.0>`__.

For each supported MXNet version, Dockerfiles can be found for each processor type (i.e. CPU and GPU).
They install the SageMaker-specific support code found in this repository.

Before building these images, you need to have a pip-installable binary of this repository saved locally.
To create the SageMaker MXNet Container Python package:

::

    # Create the binary
    git clone https://github.com/aws/sagemaker-mxnet-container.git
    cd sagemaker-mxnet-container
    python setup.py sdist

Once you have those binaries, you can then build the image.

If you are building images for Python 3 with MXNet 1.6.0, the Dockerfiles don't require any build arguments.
You do need to copy the pip-installable binary from above to ``docker/1.6.0/``.

If you are building images for Python 2 or Python 3 with MXNet 1.4.0 or lower, the Dockerfiles expect two build arguments:

- ``py_version``: the Python version (i.e. 2 or 3).
- ``framework_support_installable``: the pip-installable binary created with the command above

The integration tests expect the Docker images to be tagged as ``preprod-mxnet:<tag>``, where ``<tag>`` looks like <mxnet_version>-<processor>-<python_version> (e.g. 1.6.0-cpu-py3).

Example commands for building images:

::

    # All build instructions assume you're starting from the root directory of this repository

    # MXNet 1.6.0, Python 3, CPU
    $ cp dist/sagemaker_mxnet_container*.tar.gz docker/1.6.0/sagemaker_mxnet_container.tar.gz
    $ cp -r docker/artifacts/* docker/1.6.0/py3
    $ cd docker/1.6.0/py3
    $ docker build -t preprod-mxnet:1.6.0-cpu-py3 -f Dockerfile.cpu .

Don't forget the period at the end of the command!

Running the tests
-----------------

Running the tests requires installation of the SageMaker MXNet Container code and its test dependencies.

::

    git clone https://github.com/aws/sagemaker-mxnet-container.git
    cd sagemaker-mxnet-container
    pip install -e .[test]

Alternatively, instead of pip installing the dependencies yourself, you can use `tox <https://tox.readthedocs.io/en/latest>`__.

Tests are defined in `test/ <https://github.com/aws/sagemaker-mxnet-containers/tree/master/test>`__ and include unit and integration tests.
The integration tests include both running the Docker containers locally and running them on SageMaker.
The tests are compatible with only the Docker images built by Dockerfiles in the current branch.

All test instructions should be run from the top level directory

Unit Tests
~~~~~~~~~~

To run unit tests:

::

    tox test/unit

Local Integration Tests
~~~~~~~~~~~~~~~~~~~~~~~

Running local integration tests require `Docker <https://www.docker.com/>`__ and `AWS credentials <https://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/setup-credentials.html>`__,
as the integration tests make calls to a couple AWS services.
Local integration tests on GPU require `nvidia-docker2 <https://github.com/NVIDIA/nvidia-docker>`__.
You Docker image must also be built in order to run the tests against it.

Local integration tests use the following pytest arguments:

- ``docker-base-name``: the Docker image's repository. Defaults to 'preprod-mxnet'.
- ``framework-version``: the MXNet version. Defaults to the latest supported version.
- ``py-version``: the Python version. Defaults to '3'.
- ``processor``: CPU or GPU. Defaults to 'cpu'.
- ``tag``: the Docker image's tag. Defaults to <mxnet_version>-<processor>-py<py-version>

To run local integration tests:

::

    tox -- test/integration/local --docker-base-name <your_docker_image> \
                                  --tag <your_docker_image_tag> \
                                  --py-version <2_or_3> \
                                  --framework-version <mxnet_version> \
                                  --processor <cpu_or_gpu>

::

    # Example
    tox -- test/integration/local --docker-base-name preprod-mxnet \
                                  --tag 1.6.0-cpu-py3 \
                                  --py-version 3 \
                                  --framework-version 1.6.0 \
                                  --processor cpu

SageMaker Integration Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~

SageMaker integration tests require your Docker image to be within an `Amazon ECR repository <https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ECS_Console_Repositories.html>`__.
They also require that you have the setup described under "Integration Tests" at https://github.com/aws/sagemaker-python-sdk#running-tests.

SageMaker integration tests use the following pytest arguments:

- ``docker-base-name``: the Docker image's `ECR repository namespace <https://docs.aws.amazon.com/AmazonECR/latest/userguide/Repositories.html>`__.
- ``framework-version``: the MXNet version. Defaults to the latest supported version.
- ``py-version``: the Python version. Defaults to '3'.
- ``processor``: CPU or GPU. Defaults to 'cpu'.
- ``tag``: the Docker image's tag. Defaults to <mxnet_version>-<processor>-py<py-version>
- ``aws-id``: your AWS account ID.
- ``instance-type``: the specified `Amazon SageMaker Instance Type <https://aws.amazon.com/sagemaker/pricing/instance-types/>`__ that the tests will run on.
  Defaults to 'ml.c4.xlarge' for CPU and 'ml.p2.xlarge' for GPU.

To run SageMaker integration tests:

::

    tox -- test/integration/sagmaker --aws-id <your_aws_id> \
                                     --docker-base-name <your_docker_image> \
                                     --instance-type <amazon_sagemaker_instance_type> \
                                     --tag <your_docker_image_tag> \

::

    # Example
    tox -- test/integration/sagemaker --aws-id 12345678910 \
                                      --docker-base-name preprod-mxnet \
                                      --instance-type ml.m4.xlarge \
                                      --tag 1.6.0-cpu-py3

Contributing
------------

Please read `CONTRIBUTING.md <https://github.com/aws/sagemaker-mxnet-containers/blob/master/CONTRIBUTING.md>`__
for details on our code of conduct, and the process for submitting pull requests to us.

License
-------

SageMaker MXNet Containers is licensed under the Apache 2.0 License.
It is copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
The license is available at: http://aws.amazon.com/apache2.0/
