#!/bin/bash

#/ Usage:
#/ export AWS_REGION=<AWS-Region>
#/ ./upgrade_kubeflow.sh eks_cluster_name

set -ex

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

# install kubeflow
../test/dlc_tests/eks/eks_manifest_templates/kubeflow/install_kubeflow.sh ${CLUSTER_NAME} ${AWS_REGION}
