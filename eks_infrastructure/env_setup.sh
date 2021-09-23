#!/bin/bash
#/ Usage: ./env_setup.sh

set -ex

# The below url/version is based on EKS v1.20. The same needs to be updated for EKS version upgrade.
KUBECTL_CLIENT="https://amazon-eks.s3-us-west-2.amazonaws.com/1.20.4/2021-04-12/bin/linux/amd64/kubectl"
EKSCTL_CLIENT="https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz"
AWS_IAM_AUTHENTICATOR="https://amazon-eks.s3-us-west-2.amazonaws.com/1.20.4/2021-04-12/bin/linux/amd64/aws-iam-authenticator"
LATEST_KUBECTL_CLIENT_VERSION=1.20

function install_kubectl_client() {
    curl --silent --location ${KUBECTL_CLIENT} -o /usr/local/bin/kubectl
    chmod +x /usr/local/bin/kubectl
}
# aws caller identity
aws sts get-caller-identity

# install aws-iam-authenticator
if ! [ -x "$(command -v aws-iam-authenticator)" ]; then
    curl --silent --location ${AWS_IAM_AUTHENTICATOR} -o /usr/local/bin/aws-iam-authenticator
    chmod +x /usr/local/bin/aws-iam-authenticator
fi

#aws-iam-authenticator version
aws-iam-authenticator version

# install kubectl
if ! [ -x "$(command -v kubectl)" ]; then
    install_kubectl_client
else
    # check if the kubectl client version is less than 1.20
    KUBECTL_VERSION=$(kubectl version --client -o json)
    CURRENT_KUBECTL_CLIENT_VERSION=$(echo "$KUBECTL_VERSION" | jq -r '.clientVersion.major').$(echo "$KUBECTL_VERSION" | jq -r '.clientVersion.minor' | sed 's/+$//g')

    if (($(echo "$CURRENT_KUBECTL_CLIENT_VERSION < $LATEST_KUBECTL_CLIENT_VERSION" | bc -l))); then
        install_kubectl_client
    fi
fi

#kubectl version
kubectl version --short --client

# install eksctl
if ! [ -x "$(command -v eksctl)" ]; then
    curl --silent --location ${EKSCTL_CLIENT} | tar xz -C /tmp
    mv /tmp/eksctl /usr/local/bin
fi

# eksctl version
eksctl version
