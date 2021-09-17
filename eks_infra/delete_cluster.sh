#!/bin/bash

#/ Usage:
#/ export AWS_REGION=<AWS-Region>
#/ export EKS_CLUSTER_MANAGER_ROLE=<ARN-of-IAM-role>
#/ ./delete.sh eks_cluster_name

set -ex

# Function to delete EKS cluster
function delete_cluster() {
  eksctl delete cluster \
    --name ${1} \
    --region ${2}
}

# Function to update kubeconfig at ~/.kube/config
function update_kubeconfig() {

  eksctl utils write-kubeconfig \
    --cluster ${1} \
    --authenticator-role-arn ${2} \
    --region ${3}

  kubectl config get-contexts
}

# Get IAM role attached to the nodegroup
function get_eks_nodegroup_iam_role() {
  NODE_GROUP_NAME=${1}
  REGION=${2}

  INSTANCE_PROFILE_PREFIX=$(aws cloudformation describe-stacks --region ${REGION} | jq -r '.Stacks[].StackName' | grep ${NODE_GROUP_NAME})

  if [ -n "${INSTANCE_PROFILE_PREFIX}" ]; then
    INSTANCE_PROFILE_NAME=$(aws iam list-instance-profiles --region ${REGION} | jq -r '.InstanceProfiles[].InstanceProfileName' | grep $INSTANCE_PROFILE_PREFIX)
    if [ -n "${INSTANCE_PROFILE_NAME}" ]; then
      ROLE_NAME=$(aws iam get-instance-profile --instance-profile-name $INSTANCE_PROFILE_NAME --region ${REGION} | jq -r '.InstanceProfile.Roles[] | .RoleName')
      echo ${ROLE_NAME}
    else
      echo "Instance Profile $INSTANCE_PROFILE_NAME does not exist for the $NODE_GROUP_NAME nodegroup"
      exit 1
    fi
  else
    echo "CloudFormation stack for $NODE_GROUP_NAME nodegroup does not exist"
    exit 1
  fi

}

# Detach IAM policy from nodegroup IAM role
function remove_iam_policy() {

  NODE_GROUP_NAME=${1}
  REGION=${2}

  ROLE_NAME=$(get_eks_nodegroup_iam_role ${NODE_GROUP_NAME} ${REGION})

  declare -a POLICY_ARN=("arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
                         "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
                        )

  for policy in ${POLICY_ARN[@]}; do
    aws iam detach-role-policy \
      --role-name $ROLE_NAME \
      --policy-arn $policy \
      --region ${REGION}
  done

}

function remove_iam_permissions_nodegroup() {
  CLUSTER_NAME=${1}
  REGION=${2}
  LIST_NODE_GROUPS=$(eksctl get nodegroup --cluster ${CLUSTER_NAME} --region ${REGION} -o json | jq -r '.[].Name')

  if [ -n "${LIST_NODE_GROUPS}" ]; then
    for NODEGROUP in ${LIST_NODE_GROUPS}; do
      remove_iam_policy ${NODEGROUP} ${REGION}
    done
  else
    echo "No Nodegroups present in the EKS cluster ${CLUSTER_NAME}"
  fi
}

# Function to delete OIDC provider
function delete_oidc_provider() {
  CLUSTER_NAME=${1}

  account_id=$(aws sts get-caller-identity | jq -r '.Account')
  oidc_issuer=$(aws eks describe-cluster --name ${CLUSTER_NAME} | jq -r '.cluster.identity.oidc.issuer')
  oidc_url=$(echo $oidc_issuer | grep -oP 'https://\K\S+')
  oidc_provider_arn="arn:aws:iam::${account_id}:oidc-provider/${oidc_url}"

  if [ -n "${oidc_provider_arn}" ]; then
    echo "Deleting OICD provider ${oidc_provider_arn} attached to EKS cluster ${CLUSTER_NAME}"
    aws iam delete-open-id-connect-provider --open-id-connect-provider-arn ${oidc_provider_arn}
  fi

}

# Function to delete IAM roles related to OIDC
function delete_oidc_iam_roles() {
  CLUSTER_NAME=${1}

  declare -a OIDC_ROLE_LIST=("kf-admin-${CLUSTER_NAME}"
    "kf-user-${CLUSTER_NAME}"
  )

  for role in ${OIDC_ROLE_LIST[@]}; do
    role_policies=$(aws iam list-role-policies --role-name ${role} | jq -r '.PolicyNames[]')
    for policy in ${role_policies[@]}; do
      aws iam delete-role-policy --role-name ${role} --policy-name ${policy}
    done
    aws iam delete-role --role-name ${role}
  done

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
delete_oidc_provider ${CLUSTER}
delete_oidc_iam_roles ${CLUSTER}
delete_cluster ${CLUSTER} ${AWS_REGION}
