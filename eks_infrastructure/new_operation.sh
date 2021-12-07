#!/bin/bash
#/ Usage:
#/ export AWS_REGION=<AWS-Region>
#/ export EKS_CLUSTER_MANAGER_ROLE=<ARN-of-IAM-role>
#/ ./new_operation.sh eks_cluster_name eks_version
set -ex

# Function to create EC2 key pair
function create_ec2_key_pair() {
  aws ec2 create-key-pair \
    --key-name ${1} \
    --query 'KeyMaterial' \
    --output text >./${1}.pem
}

# Function to create graviton nodegroup in EKS cluster
function create_graviton_node_group() {

  GRAVITON_NODEGROUP_INSTANCE_TYPE="c6g.4xlarge"

  # dynamic graviton nodegroup
  eksctl create nodegroup \
    --name ${1}-graviton-nodegroup-${2/./-} \
    --cluster ${1} \
    --node-type ${GRAVITON_NODEGROUP_INSTANCE_TYPE} \
    --nodes-min 0 \
    --nodes-max 100 \
    --node-volume-size 80 \
    --node-labels "test_type=graviton" \
    --tags "k8s.io/cluster-autoscaler/node-template/label/test_type=graviton" \
    --asg-access \
    --ssh-access \
    --ssh-public-key "${3}"
}


if [ $# -ne 2 ]; then
  echo "usage: ./${0} eks_cluster_name eks_version"
  exit 1
fi

if [ -z "${AWS_REGION}" ]; then
  echo "AWS region not configured"
  exit 1
fi

# Check for EC2 keypair environment variable. If empty, create a new key pair.
if [ -z "${EC2_KEY_PAIR_NAME}" ]; then
  KEY_NAME=${CLUSTER}-KeyPair
  echo "No EC2 key pair name configured. Creating keypair ${KEY_NAME}"
  create_ec2_key_pair ${KEY_NAME}
  EC2_KEY_PAIR_NAME=${KEY_NAME}
else
  exist=$(aws ec2 describe-key-pairs --key-name ${EC2_KEY_PAIR_NAME} --region ${AWS_REGION} | grep KeyName | wc -l)
  if [ ${exist} -eq 0 ]; then
    echo "EC2 key pair ${EC2_KEY_PAIR_NAME} does not exist in ${AWS_REGION} region"
    exit 1
  fi
fi

CLUSTER=${1}

create_graviton_node_group ${CLUSTER} ${EKS_VERSION} ${EC2_KEY_PAIR_NAME}
