#!/bin/bash
#/ Usage: ./env_setup.sh 

set -e

# aws caller identity
aws sts get-caller-identity

#install kubectl
curl -o kubectl https://amazon-eks.s3.us-west-2.amazonaws.com/1.18.9/2020-11-02/bin/linux/amd64/kubectl
chmod +x ./kubectl
mv ./kubectl $(which kubectl)
kubectl version --short --client

# install eksctl

# rolling the eksctl version to 0.34.0 due to a bug https://github.com/weaveworks/eksctl/issues/3005 causing the nodes not joining cluster
# as kubelet service on worker node fails. Fix included in v0.36.0 which is in pre-release state
# "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz"

curl --silent --location "https://github.com/weaveworks/eksctl/releases/download/0.34.0/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
mv /tmp/eksctl /usr/local/bin
eksctl version



