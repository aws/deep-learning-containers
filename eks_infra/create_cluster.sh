#!/bin/bash
#/ Usage: 
#/ export AWS_REGION=<AWS-Region>
#/ export EC2_KEY_PAIR_NAME=<EC2-Key-Pair-Name>
#/ ./create_cluster.sh eks_cluster_name eks_version

set -e

# Log color
RED='\033[0;31m'

# Function to create EC2 key pair
function create_ec2_key_pair() {
    aws ec2 create-key-pair \
    --key-name ${1} \
    --query 'KeyMaterial' \
    --output text > ./${1}-KeyPair.pem
}

# Function to create EKS cluster using eksctl
function create_eks_cluster() {
    
    eksctl create cluster \
    --name ${1} \
    --version ${2} \
    --zones=${3}a,${3}b,${3}c \
    --without-nodegroup
}

# Function to create static and dynamic nodegroups in EKS cluster
function create_node_group(){
    # static nodegroup
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

    # dynamic gpu nodegroup
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

    # dynamic inf nodegroup

    eksctl create nodegroup \
    --name inf-nodegroup-${2/./-} \
    --cluster ${1} \
    --node-type inf1.xlarge \
    --nodes-min 0 \
    --nodes-max 100 \
    --node-volume-size 500 \
    --node-ami "ami-092059396c7e51f52" \
    --node-labels "test_type=inf" \
    --tags "k8s.io/cluster-autoscaler/node-template/label/test_type=inf" \
    --asg-access \
    --ssh-access \
    --ssh-public-key "${3}"
}

# Function to create namespaces in EKS cluster
function create_namespaces(){
  kubectl create -f namespace.yaml
}

# Check for input arguments
if [ $# -ne 2 ]; then
    echo "${RED}${0}: usage: ./create_cluster.sh eks_cluster_name eks_version"
    exit 1
fi

# Check for IAM role environment variables
if [ -z "${AWS_REGION}" ]; then
  echo "${RED}AWS region not configured"
  exit 1
fi

CLUSTER=${1}
EKS_VERSION=${2}
REGION=${AWS_REGION}

# Check for EC2 keypair environment variable. If empty, create a new key pair. 
if [ -z "${EC2_KEY_PAIR_NAME}" ]; then
  KEY_NAME=${CLUSTER}-KeyPair
  echo "${RED}No EC2 key pair name configured. Creating keypair ${KEY_NAME}"
  create_ec2_key_pair ${KEY_NAME}
  EC2_KEY_PAIR_NAME=${KEY_NAME}
else
  EC2_KEY_PAIR_NAME=${EC2_KEY_PAIR_NAME}
fi

create_eks_cluster ${CLUSTER} ${EKS_VERSION} ${REGION}
create_node_group ${CLUSTER} ${EKS_VERSION} ${EC2_KEY_PAIR_NAME}
create_namespaces