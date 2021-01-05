#!/bin/bash
set -e

#cluster upgrades
#https://eksctl.io/usage/cluster-upgrade/
#https://docs.aws.amazon.com/eks/latest/userguide/update-cluster.html

#upgrade cluster autoscalar to the version matching the upgrade https://github.com/kubernetes/autoscaler/releases

function update_kubeconfig(){
    eksctl utils write-kubeconfig \
    --cluster ${1} \
    --authenticator-role-arn ${2} \
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
    kubectl scale deployments/cluster-autoscaler --replicas=${1} -n kube-system
}

function upgrade_autoscalar_image(){
    kubectl -n kube-system set image deployment.apps/cluster-autoscaler cluster-autoscaler=k8s.gcr.io/autoscaling/cluster-autoscaler:$1
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

function delete_nodegroups(){

    LIST_NODE_GROUPS=$(eksctl get nodegroup --cluster ${1} -o json | jq -r '.[].StackName')

    for NODEGROUP in $LIST_NODE_GROUPS; do
      eksctl delete nodegroup --name $NODEGROUP --cluster ${1} --region ${2} --wait
    done
}

#upgrade control plane

function upgrade_nodegroups(){
    create_node_group ${1} ${2}
    delete_nodegroups ${1} ${4}
}


#Updating default add-ons
function update_eksctl_utils(){
    eksctl utils update-kube-proxy
    eksctl utils update-aws-node
    eksctl utils update-coredns
}

if [ $# -ne 5 ]; then
    echo $0: usage: ./upgrade_cluster.sh cluster_name eks_version cluster_autoscalar_image_version iam_role aws_region
    exit 1
fi

CLUSTER=$1
EKS_VERSION=$2
CLUSTER_AUTOSCALAR_IMAGE_VERSION=$3
EKS_ROLE_ARN=$4
REGION=$5

update_kubeconfig $CLUSTER $EKS_ROLE_ARN $REGION

#scale to 0 to avoid unwanted scaling
scale_cluster_autoscalar 0

upgrade_autoscalar_image $EKS_VERSION
upgrade_eks_control_plane $CLUSTER $EKS_VERSION
upgrade_nodegroups $CLUSTER $EKS_VERSION $REGION
update_eksctl_utils

#scale back to 1
scale_cluster_autoscalar 1