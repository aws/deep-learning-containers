# sb face detection in pytorch serving

## local development

first, run the init script -- this will
clone [DSFD-Pytorch-Inference](https://github.com/hukkelas/DSFD-Pytorch-Inference/), the pytorch implementation
of `dsfd` and `retinaface` we use for face detection. it will also check out the most recent currently working commit

next, build the container from `pytorch/inference/docker/X.Y.Z/py3/Dockerfile.cpu`. I set up a pycharm docker run config
which amounts to:

```shell script
docker build -f Dockerfile.cpu -t pytorch-inference:X.Y.Z-cpu-py36 .
```

run this container, mounting your eventual `model.tar.gz` contents at `/opt/ml/model`. this mount *must* be `ro`, as the
sagemaker deployment mounts it into a `ro` directory. I also did this via a pycharm run config, which amounts to:

```shell script
docker run \
    -p 8080:8080 -p 8081:8081 \
    --env SAGEMAKER_PROGRAM=inference.py \
    --env AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
    --env AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
    --env MODEL_NAME=RetinaNetResNet50 \
    --env WITH_LANDMARKS=TRUE \
    --name sb-face-detect \
    --rm \
    sb-face-detect serve
```

where

+ the `AWS_` keys could be from your environment or hard-coded when you execute the command
+ `MODEL_NAME` should be either `RetinaNetResNet50` or `DSFDDetector`
+ if `WITH_LANDMARKS=TRUE` and `MODEL_NAME=RetinaNetResNet50`, we will return face landmarks
    + any other value of `WITH_LANDMARKS` will not attempt to return landmarks
    + `WITH_LANDMARKS=TRUE` and a different model name will result in an error

## test commands

```shell script
# definitely must work
curl localhost:8080/invocations -H "Content-Type: text/csv" -d "videoblocks-ml/data/object-detection-research/videoblocks/dev/sampled-items/jpg/fps-method-01/000023419/000216-9.0090.jpg"
curl localhost:8080/invocations -H "Content-Type: application/json" -d '{"bucket":"videoblocks-ml","key":"data/object-detection-research/videoblocks/dev/sampled-items/jpg/fps-method-01/000023419/000216-9.0090.jpg"}'
```

## how I deployed a live endpoint

high level:

1. build archive
1. post archive to s3
1. run sagemaker code

### build archive

```shell
./model-archive-build.sh
```

### post archive to s3

use the neighboring `model-archive-deploy.sh` script, which takes one argument: the environment we are posting this
archive to (one of `dev`, `staging`, or `prod`; default is `dev`), e.g.

```shell
./model-archive-build.sh prod
```

### run sagemaker code

the output `s3` path of your archive file will be used in the below, so capture that from the output
of `model-archive-build.sh`.

presumably you compiled a container and tested your model against that container; the `X.Y.Z/pyV` version of that
compiled container should be provided as the `framework_version` and `py_version` below

note: you must have sagemaker python sdk version 2+

#### deploy realtime, make prediction, and tear down

```python
from sagemaker.pytorch.model import PyTorchModel

env = {'SAGEMAKER_PROGRAM': 'inference.py',
       'MODEL_NAME': 'RetinaNetResNet50',
       'WITH_LANDMARKS': 'TRUE', }
# 'CONFIDENCE_THRESHOLD': 0.5
# 'NMS_IOU_THRESHOLD': 0.3

# UPDATE THESE!
framework_version = '1.4.0'
py_version = 'py3'
s3_model_archive = 's3://videoblocks-ml/models/retinaface/videoblocks/dev/20201204T164853/model.tar.gz'

model = PyTorchModel(model_data=s3_model_archive,
                     framework_version=framework_version,
                     py_version=py_version,
                     role='sagemaker-ds',
                     entry_point='inference.py',
                     env=env)

predictor = model.deploy(initial_instance_count=1,
                         instance_type='ml.p3.2xlarge')

# hard to predictor.predict here for networking reasons

import boto3
import json

runtime = boto3.client('runtime.sagemaker')
bucket = 'videoblocks-ml'
key = 'data/object-detection-research/videoblocks/dev/sampled-items/jpg/fps-method-01/000023419/000216-9.0090.jpg'

# csv test
result = runtime.invoke_endpoint(EndpointName=model.endpoint_name,
                                 Body=f"{bucket}/{key}",
                                 ContentType='text/csv')
print(result["Body"].read())

# json test
result = runtime.invoke_endpoint(EndpointName=model.endpoint_name,
                                 Body=json.dumps({'bucket': bucket, 'key': key}),
                                 ContentType='application/json')
print(result["Body"].read())

# bring 'er down cap'n
predictor.delete_endpoint()
model.delete_model()
```

#### deploy batch transform

see [this databricks notebook](https://dbc-eceaffad-4e12.cloud.databricks.com/?o=6154618236860539#notebook/1266580902362296)