#!/bin/bash
#/ Usage: ./env_setup.sh

set -ex

# The below url/version is based on EKS v1.32.0. The same needs to be updated for EKS version upgrade.
# https://docs.aws.amazon.com/eks/latest/userguide/install-kubectl.html find the link
KUBECTL_CLIENT="https://s3.us-west-2.amazonaws.com/amazon-eks/1.35.0/2026-01-29/bin/linux/amd64/kubectl"
EKSCTL_CLIENT="https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz"
# https://github.com/kubernetes-sigs/aws-iam-authenticator/releases?page=1 to find the version support
AWS_IAM_AUTHENTICATOR="https://github.com/kubernetes-sigs/aws-iam-authenticator/releases/download/v0.7.10/aws-iam-authenticator_0.7.10_linux_amd64"

LATEST_KUBECTL_CLIENT_VERSION=1.35

function install_kubectl_client() {
    curl --silent --location ${KUBECTL_CLIENT} -o /usr/local/bin/kubectl
    chmod +x /usr/local/bin/kubectl
}

# aws caller identity
aws sts get-caller-identity

# install aws-iam-authenticator
curl --silent --location ${AWS_IAM_AUTHENTICATOR} -o /usr/local/bin/aws-iam-authenticator
chmod +x /usr/local/bin/aws-iam-authenticator

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
kubectl version --client

# install eksctl
if ! [ -x "$(command -v eksctl)" ]; then
    curl --silent --location ${EKSCTL_CLIENT} | tar xz -C /tmp
    mv /tmp/eksctl /usr/local/bin
fi

# eksctl version
eksctl version
