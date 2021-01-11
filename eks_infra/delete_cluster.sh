#!/bin/bash

#/ Usage: 
#/ export AWS_REGION=<AWS-Region>
#/ export EKS_CLUSTER_MANAGEMENT_ROLE=<ARN-of-IAM-role>
#/ ./delete.sh eks_cluster_name

set -e

# Log color
RED='\033[0;31m'

# Function to delete EKS cluster
function delete_cluster(){
    eksctl delete cluster \
    --name ${1} \
    --region ${2}
}

# Function to update kubeconfig at ~/.kube/config
function update_kubeconfig(){

    eksctl utils write-kubeconfig \
    --cluster ${1} \
    --authenticator-role-arn ${2} \
    --region ${3}
}

# Check for input arguments
if [ $# -ne 1 ]; then
    echo "${RED}$0: usage: ./delete_cluster.sh eks_cluster_name"
    exit 1
fi

# Check for environment variables
if [ -z "${AWS_REGION}" ]; then
  echo "${RED}AWS region not configured"
  exit 1
fi

if [ -z "${EKS_CLUSTER_MANAGEMENT_ROLE}" ]; then
  echo "${RED}EKS cluster management role not set"
  exit 1
fi

CLUSTER=${1}
REGION=${AWS_REGION}
EKS_ROLE=${EKS_CLUSTER_MANAGEMENT_ROLE}

update_kubeconfig ${CLUSTER} ${EKS_ROLE $REGION}
delete_cluster ${CLUSTER} ${REGION}