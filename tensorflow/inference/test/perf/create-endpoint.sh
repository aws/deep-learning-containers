#!/bin/bash

instance_type="${1:-c5.xlarge}"
if [[ "$instance_type" == p* ]]; then
    arch='gpu'
else
    arch='cpu'
fi

endpoint_name=$(echo "sagemaker-tensorflow-serving-$instance_type" | tr . -)

aws sagemaker create-endpoint-config \
    --endpoint-config-name $endpoint_name \
    --production-variants '[{
        "VariantName": "variant-name-1",
        "ModelName": "sagemaker-tensorflow-serving-model-'$arch'",
        "InitialInstanceCount": 1,
        "InstanceType": "ml.'$instance_type'"
    }]'

aws sagemaker create-endpoint --endpoint-name $endpoint_name --endpoint-config-name $endpoint_name
