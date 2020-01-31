#!/bin/bash

set -e

arch=${1:-'cpu'}
aws_region=$(aws configure get region)
aws_account=$(aws --region $aws_region sts --endpoint-url https://sts.$aws_region.amazonaws.com get-caller-identity --query 'Account' --output text)

# change this to match SageMaker execution role in your account
sagemaker_role="arn:aws:iam::$aws_account:role/service-role/AmazonSageMaker-ExecutionRole-20180510T114550"

tar -C test/resources/models -czf /tmp/sagemaker-tensorflow-serving-model.tar.gz .
aws s3 mb s3://sagemaker-$aws_region-$aws_account || true
aws s3 cp /tmp/sagemaker-tensorflow-serving-model.tar.gz s3://sagemaker-$aws_region-$aws_account/sagemaker-tensorflow-serving/test-models/sagemaker-tensorflow-serving-model.tar.gz
rm /tmp/sagemaker-tensorflow-serving-model.tar.gz


aws sagemaker create-model \
    --model-name sagemaker-tensorflow-serving-model-$arch \
    --primary-container '{
        "Image": "'$aws_account'.dkr.ecr.'$aws_region'.amazonaws.com/sagemaker-tensorflow-serving:1.11.1-'$arch'",
        "ModelDataUrl": "s3://sagemaker-'$aws_region'-'$aws_account'/sagemaker-tensorflow-serving/test-models/sagemaker-tensorflow-serving-model.tar.gz",
        "Environment": {
            "SAGEMAKER_TFS_DEFAULT_MODEL_NAME": "half_plus_three"
        }
    }' \
    --execution-role-arn "$sagemaker_role"
