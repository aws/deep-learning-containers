#!/bin/bash
set -e

function create_ec2_key_pair() {
    aws ec2 create-key-pair \
    --key-name "${1}-KeyPair" \
    --query 'KeyMaterial' \
    --output text > ./${1}-KeyPair.pem
}

function create_eks_cluster() {
    
    eksctl create cluster \
    --name ${1} \
    --version ${2} \
    --zones=${3}a,${3}b,${3}c \
    --without-nodegroup
}

function create_node_group(){
    #static nodegroup
    eksctl create nodegroup \
    --name static-nodegroup-${2/./-} \
    --cluster ${1} \
    --node-type m5.large \
    --nodes 1 \
    --node-labels "static=true" \
    --tags "k8s.io/cluster-autoscaler/node-template/label/static=true" \
    --asg-access \
    --ssh-access \
    --ssh-public-key "${3}"

    #gpu nodegroup
    eksctl create nodegroup \
    --name gpu-nodegroup-${2/./-} \
    --cluster ${1} \
    --node-type p3.16xlarge \
    --nodes-min 0 \
    --nodes-max 100 \
    --node-volume-size 80 \
    --node-labels "test_type=gpu" \
    --tags "k8s.io/cluster-autoscaler/node-template/label/test_type=gpu" \
    --asg-access \
    --ssh-access \
    --ssh-public-key "${3}"

    #TODO: inf nodegroup
}

function create_namespaces(){
  kubectl create -f namespace.yaml
}

if [ $# -ne 2 ]; then
    echo ${0}: usage: ./create_cluster.sh cluster_name eks_version
    exit 1
fi

if [ -z "$AWS_REGION" ]; then
  echo "AWS region not configured"
  exit 1
fi

if [ -n "$EC2_KEY_PAIR_NAME" ]; then
  echo "No EC2 key pair name configured. Creating one"
  KEY_NAME=${CLUSTER}-KeyPair
  create_ec2_key_pair $KEY_NAME
  EC2_KEY_PAIR_NAME=$KEY_NAME
else
  EC2_KEY_PAIR_NAME=$EC2_KEY_PAIR_NAME
fi


CLUSTER=$1
EKS_VERSION=$2
REGION=$AWS_REGION

create_eks_cluster $CLUSTER $EKS_VERSION $REGION
create_node_group $CLUSTER $EKS_VERSION $EC2_KEY_PAIR_NAME
create_namespaces