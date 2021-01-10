#!/bin/bash
set -e

#cluster upgrades
#https://eksctl.io/usage/cluster-upgrade/
#https://docs.aws.amazon.com/eks/latest/userguide/update-cluster.html

#upgrade cluster autoscalar to the version matching the upgrade https://github.com/kubernetes/autoscaler/releases

function update_kubeconfig(){
    IAM_ROLE=$(aws iam get-role --role-name ${2} --query Role.Arn --output text)

    eksctl utils write-kubeconfig \
    --cluster ${1} \
    --authenticator-role-arn ${IAM_ROLE} \
    --region ${3}

    kubectl config get-contexts
    cat /root/.kube/config
}

function upgrade_eks_control_plane(){

    eksctl upgrade cluster \
    --name=${1} \
    --version ${2} \
    --approve
}

function scale_cluster_autoscalar(){
    kubectl scale deployments/cluster-autoscaler \
    --replicas=${1} \
    -n kube-system
}

function upgrade_autoscalar_image(){
    kubectl -n kube-system \
    set image deployment.apps/cluster-autoscaler cluster-autoscaler=k8s.gcr.io/autoscaling/cluster-autoscaler:${1}
}

function create_nodegroups(){
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
    --ssh-public-key "${4}"

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
    --ssh-public-key "${4}"

    #TODO: inf nodegroup
}

function delete_nodegroups(){

    LIST_NODE_GROUPS=$(eksctl get nodegroup --cluster ${1} -o json | jq -r '.[].Name')
    #TODO: add null check
    for NODEGROUP in $LIST_NODE_GROUPS; do
      eksctl delete nodegroup \
      --name $NODEGROUP \
      --cluster ${1} \
      --region ${2} \
      --wait
    done
}

#upgrade control plane

function upgrade_nodegroups(){
    delete_nodegroups ${1} ${3}
    create_nodegroups ${1} ${2} ${4}
}

#Updating default add-ons
function update_eksctl_utils(){
    eksctl utils update-kube-proxy \
    --cluster ${1} \
    --region ${2} \
    --approve

    eksctl utils update-aws-node \
    --cluster ${1} \
    --region ${2} \
    --approve

    eksctl utils update-coredns \
    --cluster ${1} \
    --region ${2} \
    --approve
}

if [ $# -ne 3 ]; then
    echo $0: usage: ./upgrade_cluster.sh cluster_name eks_version cluster_autoscalar_image_version
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
EKS_VERSION=$2
CLUSTER_AUTOSCALAR_IMAGE_VERSION=$3
REGION=$AWS_REGION
EKS_ROLE=$EKS_CLUSTER_MANAGEMENT_ROLE

if [ -z "$EC2_KEY_PAIR_NAME" ]; then
  echo "No EC2 key pair name configured. Creating one"
  KEY_NAME=${CLUSTER}-KeyPair
  create_ec2_key_pair $KEY_NAME
  EC2_KEY_PAIR_NAME=$KEY_NAME
else
  EC2_KEY_PAIR_NAME=$EC2_KEY_PAIR_NAME
fi



update_kubeconfig $CLUSTER $EKS_ROLE $REGION

#scale to 0 to avoid unwanted scaling
scale_cluster_autoscalar 0

upgrade_autoscalar_image $CLUSTER_AUTOSCALAR_IMAGE_VERSION
upgrade_eks_control_plane $CLUSTER $EKS_VERSION
upgrade_nodegroups $CLUSTER $EKS_VERSION $REGION $EC2_KEY_PAIR_NAME
update_eksctl_utils $CLUSTER $REGION

#scale back to 1
scale_cluster_autoscalar 1