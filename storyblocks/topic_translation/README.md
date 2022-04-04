# sb topic translation in pytorch serving

## local development

next, build the container from `pytorch/inference/docker/X.Y.Z/py3/Dockerfile.cpu`. I set up a pycharm docker run config
which amounts to:

```shell script
docker build -f Dockerfile.cpu -t pytorch-inference:X.Y.Z-cpu-py36 .
```

*note*: for some versions of the `pytorch` container you may also need to copy the file `src/deep_learning_container.py`
into the `build_artifacts` directory for that particular version. I think that the `buildspec.yml` files would take care
of that if I knew how to use them

run this container, mounting your eventual `model.tar.gz` contents at `/opt/ml/model`. this mount *must* be `ro`, as the
sagemaker deployment mounts it into a `ro` directory. I also did this via a pycharm run config, which amounts to:

```shell script
docker build -f Dockerfile.cpu -t sb-topic-translation:video-audio-1.11-cpu-py3 .
docker run \
    -p 8080:8080 -p 8081:8081 \
    --env SAGEMAKER_PROGRAM=inference.py \
    --env AWS_ACCESS_KEY_ID=[fill this in] \
    --env AWS_SECRET_ACCESS_KEY=[fill this in] \
    --name sb-topic-translation \
    --rm \
    sb-topic-translation:video-audio-1.11-cpu-py3 serve
```

where

+ the `AWS_` keys should be supplied, and should correspond to an account that has read permission for the input s3
  files


## test commands

```shell script
# definitely must work
curl localhost:8080/invocations -H "Content-Type: application/json" -d '{"srcContentType":"footage","tgtContentType":"music","sparseVector":{"dim":200,"vector":{"1":0.2,"5":0.3,"100":0.5}}}'
```

## how I deployed a live endpoint

high level:

1. build archive
1. post archive to s3
1. run sagemaker code

### build archive

```shell
./model_archive_build.sh
```

### post archive to s3

use the neighboring `model_archive_deploy.sh` script, which takes one argument: the environment we are posting this
archive to (one of `dev`, `staging`, or `prod`; default is `dev`), e.g.

```shell
./model_archive_deploy.sh prod
```

### run sagemaker code

the output `s3` path of your archive file will be used in the below, so capture that from the output of
`model-archive-build.sh`.

presumably you compiled a container and tested your model against that container; the `X.Y.Z/pyV` version of that
compiled container should be provided as the `framework_version` and `py_version` below

note: you must have sagemaker python sdk version 2+

#### deploy realtime, make prediction, and tear down

```python
from sagemaker.pytorch.model import PyTorchModel

env = {'SAGEMAKER_PROGRAM': 'inference.py'}

# UPDATE THESE!
framework_version = '1.10'
py_version = 'py38'
s3_model_archive = 's3://videoblocks-ml/models/topic-translation/video-audio/prod/20220401T192503/model.tar.gz'

model = PyTorchModel(model_data=s3_model_archive,
                     framework_version=framework_version,
                     py_version=py_version,
                     role='sagemaker-ds',
                     entry_point='inference.py',
                     env=env)

predictor = model.deploy(initial_instance_count=1,
                         instance_type='ml.c5.large')

# hard to predictor.predict here for networking reasons

import boto3
import json

runtime = boto3.client('runtime.sagemaker')
body = json.dumps({
  "srcContentType": "footage",
  "tgtContentType": "music",
  "sparseVector": {
    "dim": 200,
    "vector": {"1": 0.2, "5": 0.5,  "100": 0.3}
  }
})

# json test
result = runtime.invoke_endpoint(EndpointName=model.endpoint_name,
                                 Body=body,
                                 ContentType='application/json')
print(result["Body"].read())

# json test x20, for timing purposes
import datetime

N = 100
loop_start = datetime.datetime.now()
for i in range(N):
    t0 = datetime.datetime.now()
    result = runtime.invoke_endpoint(EndpointName=model.endpoint_name,
                                     Body=body,
                                     ContentType='application/json')
    t1 = datetime.datetime.now()
    print(f'attempt {i} took {t1 - t0}')
loop_end = datetime.datetime.now()
total_time = loop_end - loop_start

print(f"processed {N} requests in {total_time}")
print(f"that is {N / (total_time).total_seconds()} RPS")
print(f"that is {(total_time).total_seconds() / N} SPR")

# bring 'er down cap'n
predictor.delete_endpoint()
model.delete_model()
```

#### deploy batch transform

see [this databricks notebook](https://dbc-eceaffad-4e12.cloud.databricks.com/?o=6154618236860539#notebook/1266580902362296)
for an implementation of face detection. currently no implementation of this model as a batch transform
