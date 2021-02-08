#!/bin/bash

#/ Usage: 
#/ export AWS_REGION=<AWS-Region>
#/ ./install_cluster_components.sh eks_cluster_name cluster_autoscalar_image_version

set -ex

# Function to install cluster autoscalar
function install_cluster_autoscalar(){
  kubectl apply -f cluster-autoscalar-autodiscover.yaml
  kubectl -n kube-system annotate deployment.apps/cluster-autoscaler cluster-autoscaler.kubernetes.io/safe-to-evict="false"
  sed -e 's/<CLUSTER_NAME>/'"${1}"'/g;s/<VERSION>/'"${2}"'/g' cluster-autoscalar-autodiscover.yaml > /tmp/cluster-autoscalar-autodiscover-${1}.yaml &&
  kubectl replace -f /tmp/cluster-autoscalar-autodiscover-${1}.yaml

}

# Function to install neuron plugin
function install_neuron_plugin(){
  kubectl apply -f ../test/dlc_tests/eks/eks_manifest_templates/neuron/neuron_device_plugin.yaml
}

# Check for input arguments
if [ $# -ne 2 ]; then
    echo "usage: ./${0} eks_cluster_name cluster_autoscalar_image_version"
    exit 1
fi

# Check for environment variables
if [ -z "$AWS_REGION" ]; then
  echo "AWS region not configured"
  exit 1
fi

CLUSTER_NAME=${1}
CLUSTER_AUTOSCALAR_IMAGE_VERSION=${2}

install_cluster_autoscalar ${CLUSTER_NAME} ${CLUSTER_AUTOSCALAR_IMAGE_VERSION}
install_neuron_plugin

# install kubeflow
../test/dlc_tests/eks/eks_manifest_templates/kubeflow/install_kubeflow_custom_kfctl.sh ${CLUSTER_NAME} ${AWS_REGION}
