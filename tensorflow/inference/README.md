# <img alt="SageMaker" src="branding/icon/sagemaker-banner.png" height="100">

# SageMaker TensorFlow Serving Container

SageMaker TensorFlow Serving Container is an a open source project that builds
docker images for running TensorFlow Serving on
[Amazon SageMaker](https://aws.amazon.com/documentation/sagemaker/).

Supported versions of TensorFlow: ``1.4.1``, ``1.5.0``, ``1.6.0``, ``1.7.0``, ``1.8.0``, ``1.9.0``, ``1.10.0``, ``1.11.0``, ``1.12.0``, ``1.13.1``, ``1.14.0``, ``1.15.0``, ``2.0.0``.

Supported versions of TensorFlow for Elastic Inference: ``1.11.0``, ``1.12.0``, ``1.13.1``, ``1.14.0``.

ECR repositories for SageMaker built TensorFlow Serving Container:

- `'tensorflow-inference'` for any new version starting with ``1.13.0`` in the following AWS accounts:
  - `"871362719292"` in `"ap-east-1"`;
  - `"217643126080"` in `"me-south-1"`;
  - `"886529160074"` in `"us-iso-east-1"`;
  - `"763104351884"` in other SageMaker public regions.
- `'sagemaker-tensorflow-serving'` for ``1.4.1``, ``1.5.0``, ``1.6.0``, ``1.7.0``, ``1.8.0``, ``1.9.0``, ``1.10.0``, ``1.11.0``, ``1.12.0`` versions in the following AWS accounts:
  - `"057415533634"` in `"ap-east-1"`;
  - `"724002660598"` in `"me-south-1"`;
  - `"520713654638"` in other SageMaker public regions.

ECR repositories for SageMaker built TensorFlow Serving Container for Elastic Inference:

- `'tensorflow-inference-eia'` for any new version starting with ``1.14.0`` in the same AWS accounts as TensorFlow Serving Container for newer TensorFlow versions listed above;
- `'sagemaker-tensorflow-serving-eia'` for ``1.11.0``, ``1.12.0``, ``1.13.1`` versions in the same AWS accounts as TensorFlow Serving Container for older TensorFlow versions listed above.

This documentation covers building and testing these docker images.

For information about using TensorFlow Serving on SageMaker, see:
[Deploying to TensorFlow Serving Endpoints](https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/tensorflow/deploying_tensorflow_serving.rst)
in the [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk) documentation.

For notebook examples, see: [Amazon SageMaker Examples](https://github.com/awslabs/amazon-sagemaker-examples).

## Table of Contents

1. [Getting Started](#getting-started)
2. [Building your image](#building-your-image)
3. [Running the tests](#running-the-tests)
4. [Pre/Post-Processing](#pre/post-processing)
5. [Deploying a TensorFlow Serving Model](#deploying-a-tensorflow-serving-model)

## Getting Started

### Prerequisites

Make sure you have installed all of the following prerequisites on your
development machine:

- [Docker](https://www.docker.com/)
- [AWS CLI](https://aws.amazon.com/cli/)

For testing, you will also need:

- [Python 3.6](https://www.python.org/)
- [tox](https://tox.readthedocs.io/en/latest/)
- [npm](https://npmjs.org/)
- [jshint](https://jshint.com/about/)

To test GPU images locally, you will also need:

- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

**Note:** Some of the build and tests scripts interact with resources in your AWS account. Be sure to
set your default AWS credentials and region using `aws configure` before using these scripts.

## Building your image

Amazon SageMaker uses Docker containers to run all training jobs and inference endpoints.

The Docker images are built from the Dockerfiles in
[docker/](https://github.com/aws/sagemaker-tensorflow-serving-container/tree/master/docker>).

The Dockerfiles are grouped based on the version of TensorFlow Serving they support. Each supported
processor type (e.g. "cpu", "gpu", "ei") has a different Dockerfile in each group.

To build an image, run the `./scripts/build.sh` script:

```bash
./scripts/build.sh --version 1.13 --arch cpu
./scripts/build.sh --version 1.13 --arch gpu
./scripts/build.sh --version 1.13 --arch eia
```


If your are testing locally, building the image is enough. But if you want to your updated image
in SageMaker, you need to publish it to an ECR repository in your account. The
`./scripts/publish.sh` script makes that easy:

```bash
./scripts/publish.sh --version 1.13 --arch cpu
./scripts/publish.sh --version 1.13 --arch gpu
./scripts/publish.sh --version 1.13 --arch eia
```

Note: this will publish to ECR in your default region. Use the `--region` argument to
specify a different region.

### Running your image in local docker

You can also run your container locally in Docker to test different models and input
inference requests by hand. Standard `docker run` commands (or `nvidia-docker run` for
GPU images) will work for this, or you can use the provided `start.sh`
and `stop.sh` scripts:

```bash
./scripts/start.sh [--version x.xx] [--arch cpu|gpu|eia|...]
./scripts/stop.sh [--version x.xx] [--arch cpu|gpu|eia|...]
```

When the container is running, you can send test requests to it using any HTTP client. Here's
and an example using the `curl` command:

```bash
curl -X POST --data-binary @test/resources/inputs/test.json \
     -H 'Content-Type: application/json' \
     -H 'X-Amzn-SageMaker-Custom-Attributes: tfs-model-name=half_plus_three' \
     http://localhost:8080/invocations
```

Additional `curl` examples can be found in `./scripts/curl.sh`.

## Running the tests

The package includes automated tests and code checks. The tests use Docker to run the container
image locally, and do not access resources in AWS. You can run the tests and static code
checkers using `tox`:

```bash
tox
```

To run local tests against a single container or with other options, you can use the following command:

```bash
python -m pytest test/integration/local
    [--docker-name-base <docker_name_base>]
    [--framework-version <framework_version>]
    [--processor-type <processor_type>]
```

To test against Elastic Inference with Accelerator, you will need an AWS account, publish your built image to ECR repository and run the following command:

    tox -e py36 -- test/integration/sagemaker/test_ei.py
        [--repo <ECR_repository_name>]
        [--instance-types <instance_type>,...]
        [--accelerator-type <accelerator_type>]
        [--versions <version>,...]

For example:

    tox -e py36 -- test/integration/sagemaker/test_ei.py \
        --repo sagemaker-tensorflow-serving-eia \
        --instance_type ml.m5.xlarge \
        --accelerator-type ml.eia1.medium \
        --versions 1.13.0


## Pre/Post-Processing

**NOTE: There is currently no support for pre-/post-processing with multi-model containers.**

SageMaker TensorFlow Serving Container supports the following Content-Types for requests:

* `application/json` (default)
* `text/csv`
* `application/jsonlines`

And the following content types for responses:

* `application/json` (default)
* `application/jsonlines`

The container will convert data in these formats to [TensorFlow Serving REST API](https://www.tensorflow.org/tfx/serving/api_rest) requests,
and will send these requests to the default serving signature of your SavedModel bundle.

You can also add customized Python code to process your input and output data. To use this feature, you need to:
1. Add a python file named `inference.py` to the code directory inside your model archive.
2. In `inference.py`, implement either a pair of `input_handler` and `output_handler` functions or a single `handler` function. Note that if `handler` function is implemented, `input_handler` and `output_handler` will be ignored.

To implement pre/post-processing handler(s), you will need to make use of the `Context` object created by Python service. The `Context` is a `namedtuple` with following attributes:
- `model_name (string)`: the name of the model you will to use for inference, for example 'half_plus_three'
- `model_version (string)`: version of the model, for example '5'
- `method (string)`: inference method, for example, 'predict', 'classify' or 'regress', for more information on methods, please see [Classify and Regress API](https://www.tensorflow.org/tfx/serving/api_rest#classify_and_regress_api) and [Predict API](https://www.tensorflow.org/tfx/serving/api_rest#predict_api)
- `rest_uri (string)`: the TFS REST uri generated by the Python service, for example, 'http://localhost:8501/v1/models/half_plus_three:predict'
- `grpc_uri (string)`: the GRPC port number generated by the Python service, for example, '9000'
- `custom_attributes (string)`: content of 'X-Amzn-SageMaker-Custom-Attributes' header from the original request, for example, 'tfs-model-name=half_plus_three,tfs-method=predict'
- `request_content_type (string)`: the original request content type, defaulted to 'application/json' if not provided
- `accept_header (string)`: the original request accept type, defaulted to 'application/json' if not provided
- `content_length (int)`: content length of the original request

Here's a code example implementing `input_handler` and `output_handler`. By providing these, the Python service will post the request to TFS REST uri with the data pre-processed by `input_handler` and pass the response to `output_handler` for post-processing.

```python
import json

def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    if context.request_content_type == 'application/json':
        # pass through json (assumes it's correctly formed)
        d = data.read().decode('utf-8')
        return d if len(d) else ''

    if context.request_content_type == 'text/csv':
        # very simple csv handler
        return json.dumps({
            'instances': [float(x) for x in data.read().decode('utf-8').split(',')]
        })

    raise ValueError('{{"error": "unsupported content type {}"}}'.format(
        context.request_content_type or "unknown"))


def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))

    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type
```

Here's another code example implementing `input_handler` and `output_handler` to format image data into a TFS request that expects image data as an encoded string rather than as a numeric tensor:

```python
import base64
import io
import json
import requests

def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API

    Args:
        data (obj): the request data stream
        context (Context): an object containing request and configuration details

    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """

    if context.request_content_type == 'application/x-image':
        payload = data.read()
        encoded_image = base64.b64encode(payload).decode('utf-8')
        instance = [{"b64": encoded_image}]
        return json.dumps({"instances": instance})
    else:
        _return_error(415, 'Unsupported content type "{}"'.format(
            context.request_content_type or 'Unknown'))


def output_handler(response, context):
    """Post-process TensorFlow Serving output before it is returned to the client.

    Args:
        response (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details

    Returns:
        (bytes, string): data to return to client, response content type
    """
    if response.status_code != 200:
        _return_error(response.status_code, response.content.decode('utf-8'))
    response_content_type = context.accept_header
    prediction = response.content
    return prediction, response_content_type


def _return_error(code, message):
    raise ValueError('Error: {}, {}'.format(str(code), message))
```

The `input_handler` above creates requests that match the input of the following TensorFlow Serving SignatureDef, displayed
using the TensorFlow `saved_model_cli`:

```
signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['image_bytes'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: input_tensor:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['classes'] tensor_info:
        dtype: DT_INT64
        shape: (-1)
        name: ArgMax:0
    outputs['probabilities'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1001)
        name: softmax_tensor:0
  Method name is: tensorflow/serving/predict
```


There are occasions when you might want to have complete control over the request handler. For example, making TFS request (REST or GRPC) to one model, and then making a request to a second model. In this case, you may implement the `handler` instead of the `input_handler` and `output_handler` pair:

```python
import json
import requests


def handler(data, context):
    """Handle request.
    Args:
        data (obj): the request data
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, (optional) response content type
    """
    processed_input = _process_input(data, context)
    response = requests.post(context.rest_uri, data=processed_input)
    return _process_output(response, context)


def _process_input(data, context):
    if context.request_content_type == 'application/json':
        # pass through json (assumes it's correctly formed)
        d = data.read().decode('utf-8')
        return d if len(d) else ''

    if context.request_content_type == 'text/csv':
        # very simple csv handler
        return json.dumps({
            'instances': [float(x) for x in data.read().decode('utf-8').split(',')]
        })

    raise ValueError('{{"error": "unsupported content type {}"}}'.format(
        context.request_content_type or "unknown"))


def _process_output(data, context):
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))

    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type
```

You can also bring in external dependencies to help with your data processing. There are 2 ways to do this:
1. If your model archive contains `code/requirements.txt`, the container will install the Python dependencies at runtime using `pip install -r`.
2. If you are working in a network-isolation situation or if you don't want to install dependencies at runtime everytime your Endpoint starts or Batch Transform job runs, you may want to put pre-downloaded dependencies under `code/lib` directory in your model archive, the container will then add the modules to the Python path. Note that if both `code/lib` and `code/requirements.txt` are present in the model archive, the `requirements.txt` will be ignored.

Your untarred model directory structure may look like this if you are using `requirements.txt`:

        model1
            |--[model_version_number]
                |--variables
                |--saved_model.pb
        model2
            |--[model_version_number]
                |--assets
                |--variables
                |--saved_model.pb
        code
            |--inference.py
            |--requirements.txt

Your untarred model directory structure may look like this if you have downloaded modules under `code/lib`:

        model1
            |--[model_version_number]
                |--variables
                |--saved_model.pb
        model2
            |--[model_version_number]
                |--assets
                |--variables
                |--saved_model.pb
        code
            |--lib
                |--external_module
            |--inference.py

## Deploying a TensorFlow Serving Model

To use your TensorFlow Serving model on SageMaker, you first need to create a SageMaker Model. After creating a SageMaker Model, you can use it to create [SageMaker Batch Transform Jobs](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-batch.html)
 for offline inference, or create [SageMaker Endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-hosting.html) for real-time inference.


### Creating a SageMaker Model

A SageMaker Model contains references to a `model.tar.gz` file in S3 containing serialized model data, and a Docker image used to serve predictions with that model.

You must package the contents in a model directory (including models, inference.py and external modules) in .tar.gz format in a file named "model.tar.gz" and upload it to S3. If you're on a Unix-based operating system, you can create a "model.tar.gz" using the `tar` utility:

```
tar -czvf model.tar.gz 12345 code
```

where "12345" is your TensorFlow serving model version which contains your SavedModel.

After uploading your `model.tar.gz` to an S3 URI, such as `s3://your-bucket/your-models/model.tar.gz`, create a [SageMaker Model](https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateModel.html) which will be used to generate inferences. Set `PrimaryContainer.ModelDataUrl` to the S3 URI where you uploaded the `model.tar.gz`, and set `PrimaryContainer.Image` to an image following this format:

```
520713654638.dkr.ecr.{REGION}.amazonaws.com/sagemaker-tensorflow-serving:{SAGEMAKER_TENSORFLOW_SERVING_VERSION}-{cpu|gpu}
```

```
763104351884.dkr.ecr.{REGION}.amazonaws.com/tensorflow-inference:{TENSORFLOW_INFERENCE_VERSION}-{cpu|gpu}
```

For those using Elastic Inference set the image following this format instead:

```
520713654638.dkr.ecr.{REGION}.amazonaws.com/sagemaker-tensorflow-serving-eia:{SAGEMAKER_TENSORFLOW_SERVING_EIA_VERSION}-cpu
```

```
763104351884.dkr.ecr.{REGION}.amazonaws.com/tensorflow-inference-eia:{TENSORFLOW_INFERENCE_EIA_VERSION}-cpu
```

Where `REGION` is your AWS region, such as "us-east-1" or "eu-west-1"; `SAGEMAKER_TENSORFLOW_SERVING_VERSION`, `SAGEMAKER_TENSORFLOW_SERVING_EIA_VERSION`, `TENSORFLOW_INFERENCE_VERSION`, `TENSORFLOW_INFERENCE_EIA_VERSION` are one of the supported versions mentioned above; and "gpu" for use on GPU-based instance types like ml.p3.2xlarge, or "cpu" for use on CPU-based instances like `ml.c5.xlarge`.

The code examples below show how to create a SageMaker Model from a `model.tar.gz` containing a TensorFlow Serving model using the AWS CLI (though you can use any language supported by the [AWS SDK](https://aws.amazon.com/tools/)) and the [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk).

#### AWS CLI
```bash
timestamp() {
  date +%Y-%m-%d-%H-%M-%S
}


MODEL_NAME="image-classification-tfs-$(timestamp)"
MODEL_DATA_URL="s3://my-sagemaker-bucket/model/model.tar.gz"

aws s3 cp model.tar.gz $MODEL_DATA_URL

REGION="us-west-2"
TFS_VERSION="1.12.0"
PROCESSOR_TYPE="gpu"
IMAGE="520713654638.dkr.ecr.$REGION.amazonaws.com/sagemaker-tensorflow-serving:$TFS_VERSION-$PROCESSOR_TYPE"

# See the following document for more on SageMaker Roles:
# https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html
ROLE_ARN="[SageMaker-compatible IAM Role ARN]"

aws sagemaker create-model \
    --model-name $MODEL_NAME \
    --primary-container Image=$IMAGE,ModelDataUrl=$MODEL_DATA_URL \
    --execution-role-arn $ROLE_ARN
```

#### SageMaker Python SDK

```python
import os
import sagemaker
from sagemaker.tensorflow.serving import Model

sagemaker_session = sagemaker.Session()
role = 'arn:aws:iam::038453126632:role/service-role/AmazonSageMaker-ExecutionRole-20180718T141171'
bucket = 'am-datasets'
prefix = 'sagemaker/high-throughput-tfs-batch-transform'
s3_path = 's3://{}/{}'.format(bucket, prefix)

model_data = sagemaker_session.upload_data('model.tar.gz',
                                           bucket,
                                           os.path.join(prefix, 'model'))

# The "Model" object doesn't create a SageMaker Model until a Transform Job or Endpoint is created.
tensorflow_serving_model = Model(model_data=model_data,
                                 role=role,
                                 framework_version='1.13',
                                 sagemaker_session=sagemaker_session)

```

After creating a SageMaker Model, you can refer to the model name to create Transform Jobs and Endpoints. Code examples are given below.

### Creating a Batch Transform Job

A Batch Transform job runs an offline-inference job using your TensorFlow Serving model. Input data in S3 is converted to HTTP requests,
and responses are saved to an output bucket in S3.

#### CLI
```bash
TRANSFORM_JOB_NAME="tfs-transform-job"
TRANSFORM_S3_INPUT="s3://my-sagemaker-input-bucket/sagemaker-transform-input-data/"
TRANSFORM_S3_OUTPUT="s3://my-sagemaker-output-bucket/sagemaker-transform-output-data/"

TRANSFORM_INPUT_DATA_SOURCE={S3DataSource={S3DataType="S3Prefix",S3Uri=$TRANSFORM_S3_INPUT}}
CONTENT_TYPE="application/x-image"

INSTANCE_TYPE="ml.p2.xlarge"
INSTANCE_COUNT=2

MAX_PAYLOAD_IN_MB=1
MAX_CONCURRENT_TRANSFORMS=16

aws sagemaker create-transform-job \
    --model-name $MODEL_NAME \
    --transform-input DataSource=$TRANSFORM_INPUT_DATA_SOURCE,ContentType=$CONTENT_TYPE \
    --transform-output S3OutputPath=$TRANSFORM_S3_OUTPUT \
    --transform-resources InstanceType=$INSTANCE_TYPE,InstanceCount=$INSTANCE_COUNT \
    --max-payload-in-mb $MAX_PAYLOAD_IN_MB \
    --max-concurrent-transforms $MAX_CONCURRENT_TRANSFORMS \
    --transform-job-name $JOB_NAME
```

#### SageMaker Python SDK

```python
output_path = 's3://my-sagemaker-output-bucket/sagemaker-transform-output-data/'
tensorflow_serving_transformer = tensorflow_serving_model.transformer(
                                     framework_version = '1.12',
                                     instance_count=2,
                                     instance_type='ml.p2.xlarge',
                                     max_concurrent_transforms=16,
                                     max_payload=1,
                                     output_path=output_path)

input_path = 's3://my-sagemaker-input-bucket/sagemaker-transform-input-data/'
tensorflow_serving_transformer.transform(input_path, content_type='application/x-image')
```

### Creating an Endpoint

A SageMaker Endpoint hosts your TensorFlow Serving model for real-time inference. The [InvokeEndpoint](https://docs.aws.amazon.com/sagemaker/latest/dg/API_runtime_InvokeEndpoint.html) API
is used to send data for predictions to your TensorFlow Serving model.

#### AWS CLI

```bash
ENDPOINT_CONFIG_NAME="my-endpoint-config"
VARIANT_NAME="TFS"
INITIAL_INSTANCE_COUNT=1
INSTANCE_TYPE="ml.p2.xlarge"
aws sagemaker create-endpoint-config \
    --endpoint-config-name $ENDPOINT_CONFIG_NAME \
    --production-variants VariantName=$VARIANT_NAME,ModelName=$MODEL_NAME,InitialInstanceCount=$INITIAL_INSTANCE_COUNT,InstanceType=$INSTANCE_TYPE

ENDPOINT_NAME="my-tfs-endpoint"
aws sagemaker create-endpoint \
    --endpoint-name $ENDPOINT_NAME \
    --endpoint-config-name $ENDPOINT_CONFIG_NAME

BODY="fileb://myfile.jpeg"
CONTENT_TYPE='application/x-image'
OUTFILE="response.json"
aws sagemaker-runtime invoke-endpoint \
    --endpoint-name $ENDPOINT_NAME \
    --content-type=$CONTENT_TYPE \
    --body $BODY \
    $OUTFILE
```

#### SageMaker Python SDK

```python
predictor = tensorflow_serving_model.deploy(initial_instance_count=1,
                                            framework_version='1.12',
                                            instance_type='ml.p2.xlarge')
prediction = predictor.predict(data)
```

## Enabling Batching

You can configure SageMaker TensorFlow Serving Container to batch multiple records together before
performing an inference. This uses [TensorFlow Serving's](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/batching/README.md)
underlying batching feature.

You may be able to significantly improve throughput, especially on GPU instances, by
enabling and configuring batching. To get the best performance, it may be necessary to tune batching parameters,
especially the batch size and batch timeout, to your model, input data, and instance type.

You can set the following environment variables on a SageMaker Model or Transform Job to enable
and configure batching:

```bash
# Configures whether to enable record batching.
# Defaults to false.
SAGEMAKER_TFS_ENABLE_BATCHING="true"

# Configures how many records
# Corresponds to "max_batch_size" in TensorFlow Serving.
# Defaults to 8.
SAGEMAKER_TFS_MAX_BATCH_SIZE="32"

# Configures how long to wait for a full batch, in microseconds.
# Corresponds to "batch_timeout_micros" in TensorFlow Serving.
# Defaults to 1000 (1ms).
SAGEMAKER_TFS_BATCH_TIMEOUT_MICROS="100000"

# Configures how many batches to process concurrently.
# Corresponds to "num_batch_threads" in TensorFlow Serving
# Defaults to number of CPUs.
SAGEMAKER_TFS_NUM_BATCH_THREADS="16"

# Configures number of batches that can be enqueued.
# Corresponds to "max_enqueued_batches" in TensorFlow Serving.
# Defaults to number of CPUs for real-time inference,
# or arbitrarily large for batch transform (because batch transform).
SAGEMAKER_TFS_MAX_ENQUEUED_BATCHES="10000"
```

## Contributing

Please read [CONTRIBUTING.md](https://github.com/aws/sagemaker-tensorflow-serving-container/blob/master/CONTRIBUTING.md)
for details on our code of conduct, and the process for submitting pull requests to us.

## License

This library is licensed under the Apache 2.0 License.
