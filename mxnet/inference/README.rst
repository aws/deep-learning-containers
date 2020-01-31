=================================
SageMaker MXNet Serving Container
=================================

SageMaker MXNet Serving Container is an open-source library for making Docker images for serving MXNet on Amazon SageMaker.

This library provides default pre-processing, predict and postprocessing for certain MXNet model types.

This library utilizes the `SageMaker Inference Toolkit <https://github.com/aws/sagemaker-inference-toolkit>`__ for starting up the model server, which is responsible for handling inference requests.

Only MXNet version 1.4 and higher are supported. For previous versions, see `SageMaker MXNet container <https://github.com/aws/sagemaker-mxnet-container>`__.

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

The Dockerfiles in this repository are intended to be used for building Docker images to run inference endpoints on `Amazon SageMaker <https://aws.amazon.com/documentation/sagemaker/>`__.

The current master branch of this repository contains Dockerfiles and support code for MXNet versions 1.4.0 and higher. For previous versions, see `SageMaker MXNet container <https://github.com/aws/sagemaker-mxnet-container>`__.
The instructions in this version of this README are for MXNet 1.4.1 and higher. For MXNet 1.4.0, see `the previous version of this file <https://github.com/aws/sagemaker-mxnet-serving-container/blob/5ec2328c20612c2aa3474c978e459b4bca033f27/README.rst>`__.

Before building these images, you need to have the pip-installable binary of this repository. To create the SageMaker MXNet Container Python package:

::

    git clone https://github.com/aws/sagemaker-mxnet-serving-container.git
    cd sagemaker-mxnet-serving-container
    python setup.py sdist

For the Python 2 and EI images, this binary should remain in ``dist/``.
For the Python 3 CPU and GPU images, this binary should be copied to ``docker/<framework_version>/py3``.
In both cases, the binary should be renamed to ``sagemaker_mxnet_serving_container.tar.gz``.

Once you have created this binary, you can then build the image.
The integration tests expect the Docker images to be tagged as ``preprod-mxnet-serving:<tag>``, where ``<tag>`` looks like <mxnet_version>-<processor>-<python_version> (e.g. 1.4.1-cpu-py3).

Example commands for building images:

::

    # All build instructions assume you're starting from this repository's root directory.

    # MXNet 1.6.0, Python 3, CPU
    $ cp dist/sagemaker_mxnet_serving_container-*.tar.gz docker/1.6.0/py3/sagemaker_mxnet_serving_container.tar.gz
    $ cp -r docker/artifacts/* docker/1.6.0/py3
    $ cd docker/1.6.0/py3
    $ docker build -t preprod-mxnet-serving:1.6.0-cpu-py3 -f Dockerfile.cpu .

    # MXNet 1.6.0, Python 2, GPU
    $ cp dist/sagemaker_mxnet_serving_container-*.tar.gz docker/1.6.0/py2/sagemaker_mxnet_serving_container.tar.gz
    $ cp -r docker/artifacts/* docker/1.6.0/py2
    $ cd docker/1.6.0/py2
    $ docker build -t preprod-mxnet-serving:1.6.0-gpu-py2 -f docker/1.6.0/py2/Dockerfile.gpu .

Don't forget the period at the end of the command!

Amazon Elastic Inference with MXNet in SageMaker
------------------------------------------------
`Amazon Elastic Inference <https://aws.amazon.com/machine-learning/elastic-inference/>`__ allows you to to attach
low-cost GPU-powered acceleration to Amazon EC2 and Amazon SageMaker instances to reduce the cost of running deep
learning inference by up to 75%. Currently, Amazon Elastic Inference supports TensorFlow, Apache MXNet, and ONNX
models, with more frameworks coming soon.

Support for using MXNet with Amazon Elastic Inference in SageMaker is supported in the public SageMaker MXNet containers.

* For information on how to use the Python SDK to create an endpoint with Amazon Elastic Inference and MXNet in SageMaker, see `Deploying MXNet Models <https://sagemaker.readthedocs.io/en/stable/using_mxnet.html#deploying-mxnet-models>`__.
* For information on how Amazon Elastic Inference works, see `How EI Works <https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html#ei-how-it-works>`__.
* For more information in regards to using Amazon Elastic Inference in SageMaker, see `Amazon SageMaker Elastic Inference <https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html>`__.
* For notebook examples on how to use Amazon Elastic Inference with MXNet through the Python SDK in SageMaker, see `EI Sample Notebooks <https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html#ei-intro-sample-nb>`__.

Building the SageMaker Elastic Inference MXNet Serving container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Amazon Elastic Inference is designed to be used with AWS enhanced versions of TensorFlow serving or Apache MXNet. These enhanced
versions of the frameworks are automatically built into containers when you use the Amazon SageMaker Python SDK, or you can
download them as binary files and import them into your own Docker containers. The enhanced MXNet binaries are available on Amazon S3 at https://s3.console.aws.amazon.com/s3/buckets/amazonei-apachemxnet.

The SageMaker MXNet containers with Amazon Elastic Inference support were built utilizing the
same instructions listed `above <https://github.com/aws/sagemaker-mxnet-serving-container#building-images>`__ with the
EIA Dockerfiles, which are all named ``Dockerfile.eia``, and can be found in the same ``docker/`` directory.

Example:

::

    # MXNet 1.4.1, Python 3, EI
    $ cp dist/sagemaker_mxnet_serving_container-*.tar.gz dist/sagemaker_mxnet_serving_container.tar.gz
    $ docker build -t preprod-mxnet-serving-eia:1.4.1-cpu-py3 -f docker/1.4.1/py3/Dockerfile.eia .


* For information about downloading and installing the enhanced binary for Apache MXNet, see `Install Amazon EI Enabled Apache MXNet <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ei-mxnet.html#ei-apache>`__.
* For information on which versions of MXNet is supported for Elastic Inference within SageMaker, see `MXNet SageMaker Estimators <https://github.com/aws/sagemaker-python-sdk#mxnet-sagemaker-estimators>`__.

Running the tests
-----------------

Running the tests requires tox.

::

    git clone https://github.com/aws/sagemaker-mxnet-serving-container.git
    cd sagemaker-mxnet-serving-container
    tox

Tests are defined in `test/ <https://github.com/aws/sagemaker-mxnet-serving-container/tree/master/test>`__ and include unit and integration tests.
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

- ``docker-base-name``: the Docker image's repository. Defaults to 'preprod-mxnet-serving'.
- ``framework-version``: the MXNet version. Defaults to the latest supported version.
- ``py-version``: the Python version. Defaults to '3'.
- ``processor``: CPU or GPU. Defaults to 'cpu'.
- ``tag``: the Docker image's tag. Defaults to <mxnet_version>-<processor>-py<py-version>

To run local integration tests:

::

    tox test/integration/local -- --docker-base-name <your_docker_image> \
                                  --tag <your_docker_image_tag> \
                                  --py-version <2_or_3> \
                                  --framework-version <mxnet_version> \
                                  --processor <cpu_or_gpu>

::

    # Example
    tox test/integration/local -- --docker-base-name preprod-mxnet-serving \
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

    tox test/integration/sagmaker -- --aws-id <your_aws_id> \
                                     --docker-base-name <your_docker_image> \
                                     --instance-type <amazon_sagemaker_instance_type> \
                                     --tag <your_docker_image_tag> \

::

    # Example
    tox test/integration/sagemaker -- --aws-id 12345678910 \
                                      --docker-base-name preprod-mxnet-serving \
                                      --instance-type ml.m4.xlarge \
                                      --tag 1.6.0-cpu-py3

If you want to run a SageMaker end to end test for your Elastic Inference container, you will need to provide an ``accelerator_type`` as an additional pytest argument.

The ``accelerator-type`` is your specified `Amazon Elastic Inference Accelerator <https://aws.amazon.com/sagemaker/pricing/instance-types/>`__ type that will be attached to your instance type.

::

    # Example for running Elastic Inference SageMaker end to end test
    tox test/integration/sagemaker/test_elastic_inference.py -- --aws-id 12345678910 \
                                                                --docker-base-name preprod-mxnet-serving \
                                                                --instance-type ml.m4.xlarge \
                                                                --accelerator-type ml.eia1.medium \
                                                                --tag 1.0

Contributing
------------

Please read `CONTRIBUTING.md <https://github.com/aws/sagemaker-mxnet-serving-container/blob/master/CONTRIBUTING.md>`__
for details on our code of conduct, and the process for submitting pull requests to us.

License
-------

SageMaker MXNet Containers is licensed under the Apache 2.0 License.
It is copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
The license is available at: http://aws.amazon.com/apache2.0/
