#!/bin/bash

#/ Usage: 
#/ export AWS_REGION=<AWS-Region>
#/ export EKS_CLUSTER_MANAGER_ROLE=<ARN-of-IAM-role>
#/ ./delete.sh eks_cluster_name

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
}

# Detach S3 policy from nodegroup IAM role
function remove_s3_access_policy(){
  NODE_GROUP_NAME=${1}
  REGION=${2}

  INSTANCE_PROFILE_PREFIX=$(aws cloudformation describe-stacks --region ${REGION} | jq -r '.Stacks[].StackName' | grep ${NODE_GROUP_NAME})

  if [ -n "${INSTANCE_PROFILE_PREFIX}" ]; then
    INSTANCE_PROFILE_NAME=$(aws iam list-instance-profiles --region ${REGION} | jq -r '.InstanceProfiles[].InstanceProfileName' | grep $INSTANCE_PROFILE_PREFIX)
    if [ -n "${INSTANCE_PROFILE_NAME}" ]; then
      S3_POLICY_ARN="arn:aws:iam::aws:policy/AmazonS3FullAccess"
      ROLE_NAME=$(aws iam get-instance-profile --instance-profile-name $INSTANCE_PROFILE_NAME --region ${REGION} | jq -r '.InstanceProfile.Roles[] | .RoleName')
      
      aws iam detach-role-policy \
      --role-name $ROLE_NAME \
      --policy-arn $S3_POLICY_ARN \
      --region ${REGION}
    else  
      echo "Instance Profile $INSTANCE_PROFILE_NAME does not exist for the $NODE_GROUP_NAME nodegroup"
    fi
  else
    echo "CloudFormation stack for $NODE_GROUP_NAME nodegroup does not exist"
  fi
}

function remove_iam_permissions_nodegroup(){
  CLUSTER_NAME=${1}
  REGION=${2}
  LIST_NODE_GROUPS=$(eksctl get nodegroup --cluster ${CLUSTER_NAME} --region ${REGION} -o json | jq -r '.[].Name')

  if [ -n "${LIST_NODE_GROUPS}" ]; then
    for NODEGROUP in ${LIST_NODE_GROUPS}; do
      remove_s3_access_policy ${NODEGROUP} ${REGION}
    done
  else
    echo "No Nodegroups present in the EKS cluster ${CLUSTER_NAME}"
  fi
}

# Check for input arguments
if [ $# -ne 1 ]; then
    echo "usage: ./${0} eks_cluster_name"
    exit 1
fi

# Check for environment variables
if [ -z "${AWS_REGION}" ]; then
  echo "AWS region not configured"
  exit 1
fi

if [ -n "${EKS_CLUSTER_MANAGER_ROLE}" ]; then
  update_kubeconfig ${CLUSTER} ${EKS_CLUSTER_MANAGER_ROLE} ${AWS_REGION}
fi

CLUSTER=${1}

remove_iam_permissions_nodegroup ${CLUSTER} ${AWS_REGION}
delete_cluster ${CLUSTER} ${AWS_REGION}