#!/bin/bash

set -e
set -x

# install kubectl
if [ ! -f "$HOME/bin/kubectl" ]; then
    curl -O https://s3.us-west-2.amazonaws.com/amazon-eks/1.31.2/2024-11-15/bin/linux/amd64/kubectl
    chmod +x ./kubectl
    mkdir -p $HOME/bin && cp ./kubectl $HOME/bin/kubectl && export PATH=$HOME/bin:$PATH
    echo 'export PATH=$HOME/bin:$PATH' >> ~/.bashrc
    rm ./kubectl
fi
echo "kubectl version: $(kubectl version --client)"


# install eksctl: you need aws credential
if [ ! -f "$HOME/bin/eksctl" ]; then
    ARCH=amd64
    PLATFORM=$(uname -s)_$ARCH
    curl -sLO "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_$PLATFORM.tar.gz"
    tar -xzf eksctl_$PLATFORM.tar.gz -C /tmp && rm eksctl_$PLATFORM.tar.gz
    sudo mv /tmp/eksctl $HOME/bin
fi
echo "eksctl version: $(eksctl version)"


# install aws-iam-authenticator
if [ ! -f "/usr/local/bin/aws-iam-authenticator" ]; then
    sudo curl --location https://github.com/kubernetes-sigs/aws-iam-authenticator/releases/download/v0.6.29/aws-iam-authenticator_0.6.29_linux_amd64 -o /usr/local/bin/aws-iam-authenticator
    sudo chmod +x /usr/local/bin/aws-iam-authenticator
fi
aws-iam-authenticator version


# # install AWSCLI V2
if [ ! -f "/usr/local/bin/aws" ]; then
    sudo yum remove awscli
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
fi
aws --version


# install helm v3
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh
helm version