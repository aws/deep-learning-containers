#!/bin/bash
set -e

function delete_cluster(){
    eksctl delete cluster \
    --name ${1} \
    --region ${2}
}

function update_kubeconfig(){
    eksctl utils write-kubeconfig \
    --cluster ${1} \
    --authenticator-role-arn ${2} \
    --region ${3}
}

if [ $# -ne 1 ]; then
    echo $0: usage: ./delete_cluster.sh cluster_name
    exit 1
fi

if [ -z "$AWS_REGION" ]; then
  echo "AWS region not configured"
  exit 1
fi

if [ -z "$EKS_CLUSTER_MANAGEMENT_ROLE" ]; then
  echo "EKS cluster management role not set"
  exit 1
fi

CLUSTER=$1
REGION=$AWS_REGION
EKS_ROLE_ARN=$EKS_CLUSTER_MANAGEMENT_ROLE

update_kubeconfig $CLUSTER $EKS_ROLE_ARN $REGION
delete_cluster $CLUSTER $REGION