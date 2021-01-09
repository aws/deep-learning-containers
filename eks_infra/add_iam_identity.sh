#!/bin/bash
set -e

function add_cluster_manager_identity(){ 

    CLUSTER_ROLE=$(aws iam get-role --role-name ${2} --query Role.Arn --output text)

    eksctl create iamidentitymapping \
    --cluster ${1} \
    --arn ${CLUSTER_ROLE} \
    --group system:masters \
    --username cluster_manager
    
    eksctl get iamidentitymapping \
    --cluster ${1}
}

function add_test_build_role(){

    TEST_ROLE=$(aws iam get-role --role-name ${2} --query Role.Arn --output text)

    kubectl create -f rbac.yaml

    eksctl create iamidentitymapping \
    --cluster ${1} \
    --arn ${TEST_ROLE} \
    --username test-role

    eksctl get iamidentitymapping \
    --cluster ${1}
}

if [ $# -lt 1 ]; then
    echo ${0}: usage: ./add_iam_identity.sh cluster_name
    exit 1
fi

if [ -z "$EKS_CLUSTER_MANAGEMENT_ROLE" ] || [ -z "$EKS_TEST_BUILD_ROLE" ]; then
  echo "One or more IAM role not set"
  exit 1
fi

CLUSTER=$1

add_cluster_manager_identity $CLUSTER $EKS_CLUSTER_MANAGEMENT_ROLE
add_test_build_role $CLUSTER $EKS_TEST_BUILD_ROLE