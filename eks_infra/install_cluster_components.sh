#!/bin/bash

#/ Usage: 
#/ export AWS_REGION=<AWS-Region>
#/ ./install_cluster_components.sh eks_cluster_name cluster_autoscalar_image_version

set -e

# Function to install cluster autoscalar
function install_cluster_autoscalar(){
  kubectl apply -f cluster-autoscalar-autodiscover.yaml
  kubectl -n kube-system annotate deployment.apps/cluster-autoscaler cluster-autoscaler.kubernetes.io/safe-to-evict="false"
  sed -e 's/<CLUSTER_NAME>/'"${1}"'/g;s/<VERSION>/'"${2}"'/g' cluster-autoscalar-autodiscover.yaml > /tmp/cluster-autoscalar-autodiscover-${1}.yaml &&
  kubectl replace -f /tmp/cluster-autoscalar-autodiscover-${1}.yaml

}

# Check for input arguments
if [ $# -ne 2 ]; then
    echo "${0}: usage: ./install_cluster_components.sh eks_cluster_name cluster_autoscalar_image_version"
    exit 1
fi

# Check for environment variables
if [ -z "$AWS_REGION" ]; then
  echo "AWS region not configured"
  exit 1
fi

function update_kubeconfig(){

    eksctl utils write-kubeconfig \
    --cluster ${1} \
    --authenticator-role-arn ${2} \
    --region ${3}

    kubectl config get-contexts
    cat /root/.kube/config
}

CLUSTER_NAME=${1}
CLUSTER_AUTOSCALAR_IMAGE_VERSION=${2}
REGION=${AWS_REGION}
EKS_ROLE=${EKS_CLUSTER_MANAGER_ROLE}

update_kubeconfig ${CLUSTER} ${EKS_ROLE} ${REGION}
#install_cluster_autoscalar ${CLUSTER_NAME} ${CLUSTER_AUTOSCALAR_IMAGE_VERSION}

# install kubeflow
./install_kubeflow_custom_kfctl.sh ${CLUSTER_NAME} ${REGION}
