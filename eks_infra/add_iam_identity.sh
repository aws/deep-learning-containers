#!/bin/bash
#/ Usage: 
#/ export AWS_REGION=<AWS-Region>
#/ export EKS_CLUSTER_MANAGER_ROLE=<ARN-of-IAM-role>
#/ export EKS_TEST_ROLE=<ARN-of-IAM-role>
#/ ./add_iam_identity.sh eks_cluster_name

set -ex

# Function to add cluster manager IAM role to EKS cluster
function add_cluster_manager_identity(){ 

    eksctl create iamidentitymapping \
    --cluster ${1} \
    --arn ${2} \
    --group system:masters \
    --username cluster_manager
    
    eksctl get iamidentitymapping \
    --cluster ${1}
}

# Function to add test IAM role to EKS cluster
function add_test_build_role(){

    eksctl create iamidentitymapping \
    --cluster ${1} \
    --arn ${2} \
    --username test-role

    eksctl get iamidentitymapping \
    --cluster ${1}
}

# Function to create rbac rules for test role in EKS cluster
function create_rbac_rules(){
    kubectl create -f rbac.yaml
}

# Check for input arguments
if [ $# -lt 1 ]; then
    echo "usage: ./${0} eks_cluster_name"
    exit 1
fi

CLUSTER=${1}

# Check for IAM role environment variables
if [ -n "${EKS_CLUSTER_MANAGER_ROLE}" ]; then
  add_cluster_manager_identity ${CLUSTER} ${EKS_CLUSTER_MANAGER_ROLE}
fi

# Check for IAM role environment variables
if [ -n "${EKS_TEST_ROLE}" ]; then
  add_test_build_role ${CLUSTER} ${EKS_TEST_ROLE}
fi

if [ -n "${EKS_CLUSTER_MANAGER_ROLE}" ] && [ -n "${EKS_TEST_ROLE}" ]; then
  create_rbac_rules
fi