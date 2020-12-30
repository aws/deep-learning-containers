#!/bin/bash
set -e

create_ec2_key_pair() {
    aws ec2 create-key-pair --key-name "${CLUSTER}-KeyPair" --query 'KeyMaterial' --output text > ./${CLUSTER}-KeyPair.pem
}

create_eks_cluster() {
    #TODO: add version/zones
    eksctl create cluster \
    --name ${CLUSTER} \
    --without-nodegroup
}

create_node_group(){
    #static
    eksctl create nodegroup \
    --name static-nodegroup \
    --cluster ${CLUSTER}\
    --node-type m5.large \
    --nodes 1 \
    --node-labels "static=true"
    --tags "k8s.io/cluster-autoscaler/node-template/label/static=true"
    --asg-access \
    --ssh-access \
    --ssh-public-key "${CLUSTER}-KeyPair"

    #gpu
    eksctl create nodegroup \
    --name gpu-nodegroup \
    --cluster ${CLUSTER}\
    --node-type p3.16xlarge \
    --nodes-min 0 \
    --nodes-max 100 \
    --node-volume-size 80 \
    --node-ami ami-061798711b2adafb4 \
    --node-labels "test_type=gpu"
    --tags "k8s.io/cluster-autoscaler/node-template/label/test_type=gpu"
    --asg-access \
    --ssh-access \
    --ssh-public-key "${CLUSTER}-KeyPair"

#TODO: nodegroup inf
}

function update_kubeconfig(){
    eksctl utils write-kubeconfig --name ${CLUSTER} --region $AWS_DEFAULT_REGION
    kubectl config get-contexts
}

if [ $# -ne 1 ]; then
    echo $0: usage: ./create_cluster.sh cluster_name
    exit 1
fi

CLUSTER=$1

#create_ec2_key_pair
#create_eks_cluster
update_kubeconfig
create_node_group