#!/bin/bash

#/ Usage: 
#/ export AWS_REGION=<AWS-Region>
#/ ./install_cluster_components.sh eks_cluster_name cluster_autoscalar_image_version

set -e

# Log color
RED='\033[0;31m'

# Function to install cluster autoscalar
function install_cluster_autoscalar(){
  #TODO: check for multiple clusters
  kubectl apply -f cluster-autoscalar-autodiscover.yaml
  kubectl -n kube-system annotate deployment.apps/cluster-autoscaler cluster-autoscaler.kubernetes.io/safe-to-evict="false"
  sed -e 's/<CLUSTER_NAME>/'"${1}"'/g;s/<VERSION>/'"${2}"'/g' cluster-autoscalar-autodiscover.yaml > /tmp/cluster-autoscalar-autodiscover-${1}.yaml &&
  kubectl replace -f /tmp/cluster-autoscalar-autodiscover-${1}.yaml

}

# Check for input arguments
if [ $# -ne 2 ]; then
    echo "${RED}$0: usage: ./install_cluster_components.sh eks_cluster_name cluster_autoscalar_image_version"
    exit 1
fi

# Check for environment variables
if [ -z "$AWS_REGION" ]; then
  echo "AWS region not configured"
  exit 1
fi

CLUSTER_NAME=${1}
CLUSTER_AUTOSCALAR_IMAGE_VERSION=${2}
REGION=${AWS_REGION}

install_cluster_autoscalar ${CLUSTER_NAME} ${CLUSTER_AUTOSCALAR_IMAGE_VERSION}

# install kubeflow
./install_kubeflow_custom_kfctl.sh ${CLUSTER_NAME} ${REGION}
