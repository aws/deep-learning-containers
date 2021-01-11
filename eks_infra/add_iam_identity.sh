#!/bin/bash
#/ Usage: 
#/ export AWS_REGION=<AWS-Region>
#/ export EKS_CLUSTER_MANAGEMENT_ROLE=<ARN-of-IAM-role>
#/ export EKS_TEST_BUILD_ROLE=<ARN-of-IAM-role>
#/ ./add_iam_identity.sh eks_cluster_name

set -e

# Log color
RED='\033[0;31m'

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
    echo "${RED}${0}: usage: ./add_iam_identity.sh eks_cluster_name"
    exit 1
fi

# Check for IAM role environment variables
if [ -z "${EKS_CLUSTER_MANAGEMENT_ROLE}" ] || [ -z "${EKS_TEST_BUILD_ROLE}" ]; then
  echo "${RED}One or more IAM role is not set"
  exit 1
fi

CLUSTER=${1}

add_cluster_manager_identity ${CLUSTER} ${EKS_CLUSTER_MANAGEMENT_ROLE}
create_rbac_rules
add_test_build_role ${CLUSTER} ${EKS_TEST_BUILD_ROLE}