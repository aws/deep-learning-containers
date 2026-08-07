#!/bin/bash

source .env

# Login to DLC registry
echo ""
echo "Logging in to DLC registry: ${REGISTRY_DLC} ..."
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $REGISTRY_DLC

echo "start building image"
docker build --progress=plain -t ${REGISTRY}${IMAGE}${TAG} -f ./Dockerfile.llama2-efa-dlc .

## Build image with docker
# login to account registry
aws ecr get-login-password | docker login --username AWS --password-stdin $REGISTRY

# create repository
aws ecr create-repository --repository-name ${IMAGE}

# push image
docker image push ${REGISTRY}${IMAGE}${TAG}