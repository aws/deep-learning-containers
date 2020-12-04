# sb face detection in pytorch serving

## local development

first, run the init script -- this will
clone [Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch), the pytorch
implementation of `efficientdet` and we use for object detection.

next, build the container from `pytorch/inference/docker/X.Y.Z/py3/Dockerfile.cpu`. I set up a pycharm docker run config
which amounts to:

```shell script
docker build -f Dockerfile.cpu -t pytorch-inference:X.Y.Z-cpu-py36 .
```

run this container, mounting your eventual `model.tar.gz` contents at `/opt/ml/model`. this mount *must* be `ro`, as the
sagemaker deployment mounts it into a `ro` directory. I also did this via a pycharm run config, which amounts to:

```shell script
docker build -f Dockerfile.cpu -t sb-object-detect-ed-d0-cpu:1.4.0-cpu-py3 .
docker run \
    -p 8080:8080 -p 8081:8081 \
    --env SAGEMAKER_PROGRAM=inference.py \
    --env AWS_ACCESS_KEY_ID=[fill this in] \
    --env AWS_SECRET_ACCESS_KEY=[fill this in] \
    --env EFFICIENTDET_COMPOUND_COEF=0 \
    --name sb-object-detect-ed-d0-cpu \
    --rm \
    sb-object-detect-ed-d0-cpu:1.4.0-cpu-py3 serve
```

where

+ the `AWS_` keys should be supplied, and should correspond to an account that has read permission for the input s3
  files
+ `EFFICIENTDET_VERSION` is the 0 - 8 indicator of the scaling coefficient for the model
+ other custom variables that could be provided:
    + `DEFAULT_PRED_THRESHOLD`: the threshold over which detections will be returned and under which they won't
    + `DEFAULT_IOU_THRESHOLD`: the threshold for intersection-over-union suppression of overlapping detections
    + `USE_FLOAT16`: whether or not to use foat16 precision instead of 32

## test commands

```shell script
# definitely must work
curl localhost:8080/invocations -H "Content-Type: text/csv" -d "videoblocks-ml/data/object-detection-research/videoblocks/dev/sampled-items/jpg/fps-method-01/000023419/000216-9.0090.jpg"
curl localhost:8080/invocations -H "Content-Type: application/json" -d '{"bucket":"videoblocks-ml","key":"data/object-detection-research/videoblocks/dev/sampled-items/jpg/fps-method-01/000023419/000216-9.0090.jpg"}'

# custom threshold values
curl localhost:8080/invocations -H "Content-Type: application/json" -d '{"bucket":"videoblocks-ml","key":"data/object-detection-research/videoblocks/dev/sampled-items/jpg/fps-method-01/000023419/000216-9.0090.jpg","pred_threshold":0.01,"iou_threshold":0.6}'

# error threshold values (should get 400)
curl localhost:8080/invocations -H "Content-Type: application/json" -d '{"bucket":"videoblocks-ml","key":"data/object-detection-research/videoblocks/dev/sampled-items/jpg/fps-method-01/000023419/000216-9.0090.jpg","pred_threshold":10}'
curl localhost:8080/invocations -H "Content-Type: application/json" -d '{"bucket":"videoblocks-ml","key":"data/object-detection-research/videoblocks/dev/sampled-items/jpg/fps-method-01/000023419/000216-9.0090.jpg","iou_threshold":10}'
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

EFFICIENTDET_COMPOUND_COEF = 0
DEFAULT_PRED_THRESHOLD = 0.01

env = {'SAGEMAKER_PROGRAM': 'inference.py',
       'EFFICIENTDET_COMPOUND_COEF': str(EFFICIENTDET_COMPOUND_COEF),
       'DEFAULT_PRED_THRESHOLD': str(DEFAULT_PRED_THRESHOLD), }

# UPDATE THESE!
framework_version = '1.4.0'
py_version = 'py3'
s3_model_archive = 's3://videoblocks-ml/models/efficientdet/videoblocks/dev/20201204T190238/model.tar.gz'

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

# json test x20, for timing purposes
import datetime

N = 100
loop_start = datetime.datetime.now()
for i in range(100):
    t0 = datetime.datetime.now()
    result = runtime.invoke_endpoint(EndpointName=model.endpoint_name,
                                     Body=json.dumps({'bucket': bucket, 'key': key}),
                                     ContentType='application/json')
    t1 = datetime.datetime.now()
    print(f'attempt {i} took {t1 - t0}')
loop_end = datetime.datetime.now()

print(f"processed {N} requests in {t1 - t0}")
print(f"that is {N / (loop_end - loop_start).total_seconds()} FPS")
print(f"that is {(loop_end - loop_start).total_seconds() / N} SPF")

# bring 'er down cap'n
predictor.delete_endpoint()
model.delete_model()
```

#### deploy batch transform

see [this databricks notebook](https://dbc-eceaffad-4e12.cloud.databricks.com/?o=6154618236860539#notebook/1266580902362296)
for an implementation of face detection. currently no implementation of this model as a batch transform