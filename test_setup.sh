#!/bin/bash

# 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.2.0-gpu-py37-cu101-ubuntu18.04-v1.1

# 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.2.0-cpu-py37-ubuntu18.04-v1.1

# 763104351884.dkr.ecr.us-west-2.amazonaws.com/aws-samples-tensorflow-training:2.2.0-gpu-py37-cu101-ubuntu18.04-example-v1.1

# 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference:2.2.0-gpu-py37-cu102-ubuntu18.04-v1.0

# 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference:2.2.0-cpu-py37-ubuntu18.04-v1.0

# 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.2.0-gpu-py37-cu102-ubuntu18.04-v2.0

# 763104351884.dkr.ecr.us-west-2.amazonaws.com/aws-samples-tensorflow-training:2.2.0-gpu-py37-cu102-ubuntu18.04-example-v2.0


export TARGET_ACCOUNT_ID_CLASSIC=763104351884
export REGION=us-west-2
export CODEBUILD_RESOLVED_SOURCE_VERSION="19c65e418b3c2b9087fa1931d267aa1f8fef5c98"
export RELEASE_SUCCESSFUL="1"


export TAG_WITH_DLC_VERSION=2.2.0-gpu-py37-cu101-ubuntu18.04-v1.1
export TARGET_ECR_REPOSITORY=tensorflow-training
python generate_dlc_image_release_information.py  --artifact-bucket dlc-release-artifacts-nikhilsk

export TAG_WITH_DLC_VERSION=2.2.0-cpu-py37-ubuntu18.04-v1.1
export TARGET_ECR_REPOSITORY=tensorflow-training
python generate_dlc_image_release_information.py  --artifact-bucket dlc-release-artifacts-nikhilsk

export TAG_WITH_DLC_VERSION=2.2.0-gpu-py37-cu101-ubuntu18.04-example-v1.1
export TARGET_ECR_REPOSITORY=aws-samples-tensorflow-training
python generate_dlc_image_release_information.py  --artifact-bucket dlc-release-artifacts-nikhilsk

export TAG_WITH_DLC_VERSION=2.2.0-gpu-py37-cu102-ubuntu18.04-v1.0
export TARGET_ECR_REPOSITORY=tensorflow-inference
python generate_dlc_image_release_information.py  --artifact-bucket dlc-release-artifacts-nikhilsk

export TAG_WITH_DLC_VERSION=2.2.0-cpu-py37-ubuntu18.04-v1.0
export TARGET_ECR_REPOSITORY=tensorflow-inference
python generate_dlc_image_release_information.py  --artifact-bucket dlc-release-artifacts-nikhilsk

export TAG_WITH_DLC_VERSION=2.2.0-gpu-py37-cu102-ubuntu18.04-v2.0
export TARGET_ECR_REPOSITORY=tensorflow-training
python generate_dlc_image_release_information.py  --artifact-bucket dlc-release-artifacts-nikhilsk

export TAG_WITH_DLC_VERSION=2.2.0-gpu-py37-cu102-ubuntu18.04-v2.0
export TARGET_ECR_REPOSITORY=tensorflow-training
python generate_dlc_image_release_information.py  --artifact-bucket dlc-release-artifacts-nikhilsk

export TAG_WITH_DLC_VERSION=2.2.0-gpu-py37-cu102-ubuntu18.04-example-v2.0
export TARGET_ECR_REPOSITORY=aws-samples-tensorflow-training
python generate_dlc_image_release_information.py  --artifact-bucket dlc-release-artifacts-nikhilsk


