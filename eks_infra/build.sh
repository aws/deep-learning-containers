#!/bin/bash
set -e

#parse parameters from build_param.json
OPERATION=$(jq -r '.operation' eks_infra/build_param.json)

EKS_CLUSTERS=($(jq -r '.eks_clusters[]' eks_infra/build_param.json))
EKS_VERSION=$(jq -r '.eks_version' eks_infra/build_param.json)
LIST_CLUSTER=($(eksctl get cluster -o json | jq -r '.[].metadata.name'))
CLUSTER_AUTOSCALAR_IMAGE_VERSION=$(jq -r '.cluster_autoscalar_image_version' eks_infra/build_param.json)

#Modify the name
EKS_ROLE="eks_cluster_role"
EKS_ROLE_ARN=$(aws iam get-role --role-name $EKS_ROLE --query Role.Arn --output text)

function create_cluster(){
  cd eks_infra
  
  for CLUSTER in "${EKS_CLUSTERS[@]}"; do
    echo $CLUSTER
    if [[ ! " ${LIST_CLUSTER[@]} " =~ " ${CLUSTER} " ]]; then
      ./create_cluster.sh $CLUSTER $EKS_VERSION $AWS_REGION $EKS_ROLE_ARN
      ./install_cluster_components.sh $CLUSTER $CLUSTER_AUTOSCALAR_IMAGE_VERSION $AWS_REGION
    else
      echo "EKS Cluster ${CLUSTER} already exist. Skipping creation of cluster"
    fi
  done
}

function upgrade_cluster(){

  cd eks_infra

  for CLUSTER in "${EKS_CLUSTERS[@]}"; do
    if [[ " ${LIST_CLUSTER[@]} " =~ " ${CLUSTER} " ]]; then
      ./upgrade_cluster.sh $CLUSTER $EKS_VERSION $CLUSTER_AUTOSCALAR_IMAGE_VERSION $EKS_ROLE_ARN $AWS_REGION
    else
      echo "EKS Cluster ${CLUSTER} does not exist"
    fi
  done
  
}

function delete_cluster(){

  cd eks_infra
  for CLUSTER in "${EKS_CLUSTERS[@]}"; do
   echo $CLUSTER
    if [[ " ${LIST_CLUSTER[@]} " =~ " ${CLUSTER} " ]]; then
      ./delete_cluster.sh $CLUSTER $EKS_ROLE_ARN $AWS_REGION
    else
      echo "EKS Cluster ${CLUSTER} does not exist"
    fi
  done
  
}

case $OPERATION in 
  
  create)
    create_cluster
  ;;

  upgrade)
    upgrade_cluster
  ;;

  delete)
    delete_cluster
  ;;
  *)
    echo "Invalid operation"
  ;;
esac 