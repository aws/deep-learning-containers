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
    --output text > ./${1}-KeyPair.pem
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
function create_node_group(){

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
    --tags "k8s.io/cluster-autoscaler/node-template/label/test_type=inf,k8s.io/cluster-autoscaler/node-template/resources/aws.amazon.com/neuron=1" \
    --asg-access \
    --ssh-access \
    --ssh-public-key "${3}"

}

# Attach S3 policy to nodegroup IAM role
function add_s3_access_policy(){
  NODE_GROUP_NAME=${1}
  REGION=${2}

  INSTANCE_PROFILE_PREFIX=$(aws cloudformation describe-stacks --region ${REGION} | jq -r '.Stacks[].StackName' | grep ${NODE_GROUP_NAME})

  if [ -n "${INSTANCE_PROFILE_PREFIX}" ]; then
    INSTANCE_PROFILE_NAME=$(aws iam list-instance-profiles --region ${REGION} | jq -r '.InstanceProfiles[].InstanceProfileName' | grep $INSTANCE_PROFILE_PREFIX)
    if [ -n "${INSTANCE_PROFILE_NAME}" ]; then
      S3_POLICY_ARN="arn:aws:iam::aws:policy/AmazonS3FullAccess"
      ROLE_NAME=$(aws iam get-instance-profile --instance-profile-name $INSTANCE_PROFILE_NAME --region ${REGION} | jq -r '.InstanceProfile.Roles[] | .RoleName')
      
      aws iam attach-role-policy \
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

function add_iam_permissions_nodegroup(){
  CLUSTER_NAME=${1}
  REGION=${2}
  LIST_NODE_GROUPS=$(eksctl get nodegroup --cluster ${CLUSTER_NAME} --region ${REGION} -o json | jq -r '.[].Name')

  if [ -n "${LIST_NODE_GROUPS}" ]; then
    for NODEGROUP in ${LIST_NODE_GROUPS}; do
      add_s3_access_policy ${NODEGROUP} ${REGION}
    done
  else
    echo "No Nodegroups present in the EKS cluster ${CLUSTER_NAME}"
  fi
}

# Function to create namespaces in EKS cluster
function create_namespaces(){
  kubectl create -f namespace.yaml
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
add_iam_permissions_nodegroup ${CLUSTER} ${AWS_REGION}
create_namespaces