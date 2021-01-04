#!/bin/bash
set -e

function delete_cluster(){
    eksctl delete cluster \
    --name ${1} \
    --region ${2}
}

function delete_nodegroups(){

    LIST_NODE_GROUPS=$(eksctl get nodegroup --cluster eks-cluster -o json | jq -r '.[].StackName')

    for NODEGROUP in $LIST_NODE_GROUPS; do
      eksctl delete nodegroup --name $NODEGROUP --cluster ${1} --region ${2} --wait
    done
}

if [ $# -ne 2 ]; then
    echo $0: usage: ./delete_cluster.sh cluster_name aws_region
    exit 1
fi

CLUSTER=$1
REGION=$2

delete_cluster $CLUSTER $REGION