#!/bin/bash
set -e

function delete_cluster(){
    eksctl delete cluster \
    --name ${1} \
    --region ${2}
}

function delete_ec2_key_pair() {
    aws ec2 delete-key-pair --key-name "${1}-KeyPair" --region ${2}
}

if [ $# -ne 2 ]; then
    echo $0: usage: ./delete_cluster.sh cluster_name aws_region
    exit 1
fi

CLUSTER=$1
REGION=$2

delete_ec2_key_pair $CLUSTER $REGION
delete_cluster $CLUSTER $REGION