#!/bin/bash
#/ Usage:
#/ export AWS_REGION=<AWS-Region>
#/ export EC2_KEY_PAIR_NAME=<EC2-Key-Pair-Name>
#/ ./create_cluster.sh eks_cluster_name eks_version

set -ex

# Function to create EC2 key pair
function create_ec2_key_pair() {
  aws ec2 create-key-pair \
    --key-name ${1} \
    --query 'KeyMaterial' \
    --output text >./${1}.pem
}

# Function to create EKS cluster using eksctl.
# The cluster name follows the {framework}-{build_context} convention
function create_eks_cluster() {

  if [ "${3}" = "us-east-1" ]; then
    ZONE_LIST=(a b d)
  else
    ZONE_LIST=(a b c)
  fi

  eksctl create cluster \
    --name ${1} \
    --version ${2} \
    --zones=${3}${ZONE_LIST[0]},${3}${ZONE_LIST[1]},${3}${ZONE_LIST[2]} \
    --without-nodegroup
}

# Function to create static and dynamic nodegroups in EKS cluster
function create_node_group() {

  STATIC_NODEGROUP_INSTANCE_TYPE="m5.large"
  GPU_NODEGROUP_INSTANCE_TYPE="p3.16xlarge"
  INF_NODEGROUP_INSTANCE_TYPE="inf1.xlarge"

  # static nodegroup
  eksctl create nodegroup \
    --name ${1}-static-nodegroup-${2/./-} \
    --cluster ${1} \
    --node-type ${STATIC_NODEGROUP_INSTANCE_TYPE} \
    --nodes 1 \
    --node-labels "static=true" \
    --tags "k8s.io/cluster-autoscaler/node-template/label/static=true" \
    --asg-access \
    --ssh-access \
    --ssh-public-key "${3}"

  # dynamic gpu nodegroup
  eksctl create nodegroup \
    --name ${1}-gpu-nodegroup-${2/./-} \
    --cluster ${1} \
    --node-type ${GPU_NODEGROUP_INSTANCE_TYPE} \
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
    --name ${1}-inf-nodegroup-${2/./-} \
    --cluster ${1} \
    --node-type ${INF_NODEGROUP_INSTANCE_TYPE} \
    --nodes-min 0 \
    --nodes-max 100 \
    --node-volume-size 500 \
    --node-labels "test_type=inf" \
    --tags "k8s.io/cluster-autoscaler/node-template/label/test_type=inf,k8s.io/cluster-autoscaler/node-template/resources/aws.amazon.com/neuron=1,k8s.io/cluster-autoscaler/node-template/resources/hugepages-2Mi=256Mi" \
    --asg-access \
    --ssh-access \
    --ssh-public-key "${3}"

}

# Attach IAM policy to nodegroup IAM role
function add_iam_policy() {
  NODE_GROUP_NAME=${1}
  CLUSTER_NAME=${2}
  REGION=${3}

  ROLE_ARN=$(aws eks describe-nodegroup --nodegroup-name ${NODE_GROUP_NAME} --cluster-name ${CLUSTER_NAME} --region ${REGION} | jq -r '.nodegroup.nodeRole')
  ROLE_NAME=$(echo ${ROLE_ARN} | grep -oP 'arn:aws:iam::\d+:role/\K\S+')

  declare -a POLICY_ARN=("arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess")

  for policy in ${POLICY_ARN[@]}; do
    aws iam attach-role-policy \
      --role-name $ROLE_NAME \
      --policy-arn $policy \
      --region ${REGION}
  done

}

function add_iam_permissions_nodegroup() {
  CLUSTER_NAME=${1}
  REGION=${2}
  LIST_NODE_GROUPS=$(eksctl get nodegroup --cluster ${CLUSTER_NAME} --region ${REGION} -o json | jq -r '.[].Name')

  if [ -n "${LIST_NODE_GROUPS}" ]; then
    for NODEGROUP in ${LIST_NODE_GROUPS}; do
      add_iam_policy ${NODEGROUP} ${CLUSTER_NAME} ${REGION}
    done
  else
    echo "No Nodegroups present in the EKS cluster ${CLUSTER_NAME}"
  fi
}

#/ Tags added to the nodegroup do not propogate to the underlying Auto Scaling Group.
#/ Hence adding the tags explicitly as it is required for cluster autoscalar functionality
#/ See https://github.com/aws/containers-roadmap/issues/608
function add_tags_asg() {

  CLUSTER_NAME=${1}
  REGION=${2}

  for details in $(eksctl get nodegroup --cluster ${CLUSTER_NAME} --region ${REGION} -o json | jq -c '.[]'); do
    nodegroup_name=$(echo $details | jq -r '.Name')
    asg_name=$(echo $details | jq -r '.AutoScalingGroupName')

    if [[ ${nodegroup_name} == *"gpu"* ]]; then
      aws autoscaling create-or-update-tags \
        --tags ResourceId=${asg_name},ResourceType=auto-scaling-group,Key=k8s.io/cluster-autoscaler/node-template/label/test_type,Value=gpu,PropagateAtLaunch=true
    fi

    if [[ ${nodegroup_name} == *"inf"* ]]; then
      aws autoscaling create-or-update-tags \
        --tags ResourceId=${asg_name},ResourceType=auto-scaling-group,Key=k8s.io/cluster-autoscaler/node-template/label/test_type,Value=inf,PropagateAtLaunch=true \
        ResourceId=${asg_name},ResourceType=auto-scaling-group,Key=k8s.io/cluster-autoscaler/node-template/resources/aws.amazon.com/neuron,Value=1,PropagateAtLaunch=true \
        ResourceId=${asg_name},ResourceType=auto-scaling-group,Key=k8s.io/cluster-autoscaler/node-template/resources/hugepages-2Mi,Value=256Mi,PropagateAtLaunch=true
    fi

  done

}

# Check for input arguments
if [ $# -ne 2 ]; then
  echo "usage: ./${0} eks_cluster_name eks_version"
  exit 1
fi

# Check for IAM role environment variables
if [ -z "${AWS_REGION}" ]; then
  echo "AWS region not configured"
  exit 1
fi

CLUSTER=${1}
EKS_VERSION=${2}

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

create_eks_cluster ${CLUSTER} ${EKS_VERSION} ${AWS_REGION}
create_node_group ${CLUSTER} ${EKS_VERSION} ${EC2_KEY_PAIR_NAME}
add_tags_asg ${CLUSTER} ${AWS_REGION}
add_iam_permissions_nodegroup ${CLUSTER} ${AWS_REGION}
