#!/bin/bash
set -e

function create_ec2_key_pair() {
    aws ec2 create-key-pair --key-name "${1}-KeyPair" --query 'KeyMaterial' --output text > ./${1}-KeyPair.pem
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
    --ssh-public-key "${1}-KeyPair"

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
    --ssh-public-key "${1}-KeyPair"

    #TODO: inf nodegroup
}

function add_iam_identity(){ 
    eksctl create iamidentitymapping --cluster ${1} --arn ${2} --group system:masters --username admin
    eksctl get iamidentitymapping --cluster ${1}
}

function update_kubeconfig(){
    eksctl utils write-kubeconfig --cluster ${1} --region ${2}
    kubectl config get-contexts
}

if [ $# -ne 4 ]; then
    echo ${0}: usage: ./create_cluster.sh cluster_name eks_version aws_region iam_role
    exit 1
fi

CLUSTER=$1
EKS_VERSION=$2
REGION=$3
EKS_ROLE_ARN=$4


create_ec2_key_pair $CLUSTER
create_eks_cluster $CLUSTER $EKS_VERSION $REGION

if [ -n "$4" ]; then
  add_iam_identity $CLUSTER $EKS_ROLE_ARN
else
  echo "No IAM role specified for identity mapping"
fi

update_kubeconfig $CLUSTER $REGION
create_node_group $CLUSTER $EKS_VERSION