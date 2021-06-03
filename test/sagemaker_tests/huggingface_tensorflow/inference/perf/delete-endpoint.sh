#!/bin/bash

endpoint=${1-'sagemaker-tensorflow-serving-cpu-c5-xlarge'}
aws sagemaker delete-endpoint --endpoint-name $endpoint
aws sagemaker delete-endpoint-config --endpoint-config-name $endpoint