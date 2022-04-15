#!/bin/bash
#/ Usage:
#/ export AWS_REGION=<AWS-Region>
#/ export EKS_CLUSTER_MANAGER_ROLE=<ARN-of-IAM-role>
#/ target can be one of [ cluster | nodegroup ]
#/ cluster_autoscalar_image_version option is not required for [nodegroup] target
#/ ./upgrade_operation.sh <target> eks_cluster_name eks_version cluster_autoscalar_image_version
set -ex

# Function to update kubeconfig at ~/.kube/config
function update_kubeconfig() {

  eksctl utils write-kubeconfig \
    --cluster ${1} \
    --authenticator-role-arn ${2} \
    --region ${3}

  kubectl config get-contexts
}

# Function to upgrade eks control plane
function upgrade_eks_control_plane() {

  eksctl upgrade cluster \
    --name ${1} \
    --version ${2} \
    --timeout 180m \
    --approve
}

# Function to control scaling of cluster autoscalar
function scale_cluster_autoscalar() {
  kubectl scale deployments/cluster-autoscaler \
    --replicas=${1} \
    -n kube-system
}
# Function to upgrade autoscalar image
function upgrade_autoscalar_image() {
  kubectl -n kube-system \
    set image deployment.apps/cluster-autoscaler cluster-autoscaler=k8s.gcr.io/autoscaling/cluster-autoscaler:${1}
}

# Function to upgrade nodegroups
function upgrade_nodegroups() {
  CLUSTER=${1}
  EKS_VERSION=${2}
  REGION=${3}

  LIST_NODE_GROUPS=$(eksctl get nodegroup --cluster ${CLUSTER} -o json | jq -r '.[].Name')

  if [ -n "${LIST_NODE_GROUPS}" ]; then

    for NODEGROUP in ${LIST_NODE_GROUPS}; do
      eksctl upgrade nodegroup \
        --name ${NODEGROUP} \
        --cluster ${CLUSTER} \
        --kubernetes-version ${EKS_VERSION} \
        --timeout 90m \
        --region ${REGION}
    done
  else
    echo "No Nodegroups present in the EKS cluster ${1}"
  fi
}

#Function to upgrade core k8s components
function update_eksctl_utils() {
  eksctl utils update-kube-proxy \
    --cluster ${1} \
    --region ${2} \
    --approve

  eksctl utils update-aws-node \
    --cluster ${1} \
    --region ${2} \
    --approve

  eksctl utils update-coredns \
    --cluster ${1} \
    --region ${2} \
    --approve
}

if [ $# -le 3 ]; then
  echo "usage: ./${0} target eks_cluster_name eks_version cluster_autoscalar_image_version"
  exit 1
fi

if [ -z "${AWS_REGION}" ]; then
  echo "AWS region not configured"
  exit 1
fi

TARGET=${1}
CLUSTER=${2}
EKS_VERSION=${3}
CLUSTER_AUTOSCALAR_IMAGE_VERSION=${4}

if [ -n "${EKS_CLUSTER_MANAGER_ROLE}" ]; then
  update_kubeconfig ${CLUSTER} ${EKS_CLUSTER_MANAGER_ROLE} ${AWS_REGION}
fi

if [ "${TARGET}" = "CLUSTER" ]; then
  #scale to 0 to avoid unwanted scaling
  scale_cluster_autoscalar 0
  upgrade_autoscalar_image ${CLUSTER_AUTOSCALAR_IMAGE_VERSION}
  upgrade_eks_control_plane ${CLUSTER} ${EKS_VERSION}
  upgrade_nodegroups ${CLUSTER} ${EKS_VERSION} ${AWS_REGION}
  update_eksctl_utils ${CLUSTER} ${AWS_REGION}
  #scale back to 1
  scale_cluster_autoscalar 1
elif [ "${TARGET}" = "NODEGROUP" ]; then
  upgrade_nodegroups ${CLUSTER} ${EKS_VERSION} ${AWS_REGION}
fi
