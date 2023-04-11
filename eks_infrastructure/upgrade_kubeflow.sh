#!/bin/bash

#/ Usage:
#/ export AWS_REGION=<AWS-Region>
#/ export EKS_CLUSTER_MANAGER_ROLE=<ARN-of-IAM-role>
#/ ./upgrade_kubeflow.sh eks_cluster_name

set -ex

# Function to update kubeconfig at ~/.kube/config
function update_kubeconfig() {

  eksctl utils write-kubeconfig \
    --cluster ${1} \
    --authenticator-role-arn ${2} \
    --region ${3}

  kubectl config get-contexts
}

# Check for eks cluster name
if [ $# -ne 1 ]; then
  echo "usage: ./${0} eks_cluster_name"
  exit 1
fi

# Check for environment variables
if [ -z "$AWS_REGION" ]; then
  echo "AWS region not configured"
  exit 1
fi

CLUSTER_NAME=${1}

if [ -n "${EKS_CLUSTER_MANAGER_ROLE}" ]; then
  update_kubeconfig ${CLUSTER_NAME} ${EKS_CLUSTER_MANAGER_ROLE} ${AWS_REGION}
fi

# install kubeflow
../test/dlc_tests/eks/eks_manifest_templates/kubeflow/install_kubeflow.sh ${CLUSTER_NAME} ${AWS_REGION}
