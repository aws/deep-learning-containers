# sb topic translation in pytorch serving

## local development

first, run the init script 

```shell
bash init_local_dev.sh ENV SRC_TYPE TGT_TYPE
```

where `ENV`, `SRC_TYPE`, and `TGT_TYPE` are related to the values used to create the topic translation arrays. e.g.
`ENV = prod`, `SRC_TYPE = video`, and `TGT_TYPE = audio` would refer to the topic translation matrices saved at
`s3://videoblocks-ml/models/topic-tran/storyblocks/prod/video-audio-arrays`

this init script will download those arrays into `model/code/{SRC_TYPE}-{TGT_TYPE}-arrays`.

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

this container supports three custom environment variables that can be set locally or in the terraform script that
publishes these as endpoints:

+ `SRC_CLASS` (default: `video`): the content class of the input topic vector
+ `TGT_CLASS` (default: `audio`): the content class of the output topic vector
+ `TTL_SECONDS` (default: 4 hours, 14,400 seconds): the maximum age in seconds of the topic translation array (a new
  request that comes in more than this many seconds after a translation array was loaded from s3 will trigger a reload
  of the same key in s3, so if that key has been updated we will refresh)

in particular, this means that we can re-use this code for other paris of source / content type by simply updating the
environment variables for the model concept in sagemaker.


## test commands

```shell script
# definitely must work
curl localhost:8080/invocations -H "Content-Type: application/json" -d '{"srcContentType":"footage","tgtContentType":"music","sparseVector":{"dim":200,"vector":{"1":0.2,"5":0.3,"100":0.5}}}'
```

```python
import requests

resp = requests.post(url='http://localhost:8080/invocations',
                     json={"srcContentType": "footage",
                           "tgtContentType": "music",
                           "sparseVector": {"dim": 200,
                                            "vector": {"1": 0.2, "5": 0.3, "100": 0.5}}})
assert resp.status_code == 200
print(resp.json())
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

use the neighboring `model_archive_deploy.sh` script, which takes three arguments:

1. the environment we are posting this archive to (one of `dev`, `staging`, or `prod`; default is `dev`)
2. the source content class for translation
3. the target content class for translation

2 and 3 are technically irrelevant or even misleading (the same code could be deployed in sagemaker with different
environment variables and it would suddenly support different source and target content classes!), but we are including
them here to support future decisions to have more tailored translation models

```shell
./model_archive_deploy.sh prod video audio
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

env = {'SAGEMAKER_PROGRAM': 'inference.py',
       'SRC_CLASS': 'video',
       'TGT_CLASS': 'audio'}

# UPDATE THESE!
framework_version = '1.10'
py_version = 'py38'
s3_model_archive = 's3://videoblocks-ml/models/topic-tran/storyblocks/prod/video-audio-20220405T171430/model.tar.gz'

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
        "vector": {"1": 0.2, "5": 0.5, "100": 0.3}
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
