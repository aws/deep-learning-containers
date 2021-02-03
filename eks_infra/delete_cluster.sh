#!/bin/bash

#/ Usage: 
#/ export AWS_REGION=<AWS-Region>
#/ export EKS_CLUSTER_MANAGER_ROLE=<ARN-of-IAM-role>
#/ ./delete.sh eks_cluster_name eks_version

set -ex

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
    kubectl config get-contexts
    
    cat /root/.kube/config
}

# Detach S3 role policy from the IAM role of GPU nodegroup worker nodes
function detach_s3_policy_for_gpu_nodes(){

  NODE_GROUP_NAME="gpu-nodegroup-${1/./-}"

  INSTANCE_PROFILE_PREFIX=$(aws cloudformation describe-stacks | jq -r '.Stacks[].StackName' | grep ${NODE_GROUP_NAME})

  if [ -n "${INSTANCE_PROFILE_PREFIX}" ]; then
    INSTANCE_PROFILE_NAME=$(aws iam list-instance-profiles | jq -r '.InstanceProfiles[].InstanceProfileName' | grep $INSTANCE_PROFILE_PREFIX)
    if [ -n "${INSTANCE_PROFILE_NAME}" ]; then
      S3_POLICY_ARN="arn:aws:iam::aws:policy/AmazonS3FullAccess"
      ROLE_NAME=$(aws iam get-instance-profile --instance-profile-name $INSTANCE_PROFILE_NAME | jq -r '.InstanceProfile.Roles[] | .RoleName')
      aws iam detach-role-policy --role-name $ROLE_NAME --policy-arn $S3_POLICY_ARN
    else  
      echo "Instance Profile $INSTANCE_PROFILE_NAME does not exist for the $NODE_GROUP_NAME nodegroup"
    fi
  else
    echo "CloudFormation stack for $NODE_GROUP_NAME nodegroup does not exist"
  fi

}

# Check for input arguments
if [ $# -ne 2 ]; then
    echo "usage: ./${0} eks_cluster_name eks_version"
    exit 1
fi

# Check for environment variables
if [ -z "${AWS_REGION}" ]; then
  echo "AWS region not configured"
  exit 1
fi

if [ -z "${EKS_CLUSTER_MANAGER_ROLE}" ]; then
  echo "EKS cluster management role not set"
  exit 1
fi

CLUSTER=${1}
EKS_VERSION=${2}

update_kubeconfig ${CLUSTER} ${EKS_CLUSTER_MANAGER_ROLE} ${AWS_REGION}
detach_s3_policy_for_gpu_nodes ${EKS_VERSION}
delete_cluster ${CLUSTER} ${AWS_REGION}