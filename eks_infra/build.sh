#!/bin/bash
set -e

#parse parameters from build_param.json
OPERATION=$(jq -r '.operation' eks_infra/build_param.json)
EKS_CLUSTERS=($(jq -r '.eks_clusters[]' eks_infra/build_param.json))
CONTEXTS=($(jq -r '.contexts[]' eks_infra/build_param.json))
EKS_VERSION=$(jq -r '.eks_version' eks_infra/build_param.json)
LIST_CLUSTER=($(eksctl get cluster -o json | jq -r '.[].metadata.name'))
CLUSTER_AUTOSCALAR_IMAGE_VERSION=$(jq -r '.cluster_autoscalar_image_version' eks_infra/build_param.json)

function create_cluster(){
  cd eks_infra
  
  for CONTEXT in "${CONTEXTS[@]}"; do
    for CLUSTER in "${EKS_CLUSTERS[@]}"; do
      CLUSTER_NAME=${CLUSTER}-${CONTEXT}

      if [[ ! " ${LIST_CLUSTER[@]} " =~ " ${CLUSTER_NAME} " ]]; then
        #./create_cluster.sh $CLUSTER_NAME $EKS_VERSION
        ./add_iam_identity.sh $CLUSTER_NAME
        ./install_cluster_components.sh $CLUSTER_NAME $CLUSTER_AUTOSCALAR_IMAGE_VERSION
      else
        echo "EKS Cluster ${CLUSTER_NAME} already exist. Skipping creation of cluster"
      fi
    done
  done
}

function upgrade_cluster(){

  cd eks_infra

  for CONTEXT in "${CONTEXTS[@]}"; do
    for CLUSTER in "${EKS_CLUSTERS[@]}"; do
      CLUSTER_NAME=${CLUSTER}-${CONTEXT}
      if [[ " ${LIST_CLUSTER[@]} " =~ " ${CLUSTER_NAME} " ]]; then
        ./upgrade_cluster.sh $CLUSTER_NAME $EKS_VERSION $CLUSTER_AUTOSCALAR_IMAGE_VERSION
      else
        echo "EKS Cluster ${CLUSTER_NAME} does not exist"
      fi
    done
  done
  
}

function delete_cluster(){

  cd eks_infra
  for CONTEXT in "${CONTEXTS[@]}"; do
    for CLUSTER in "${EKS_CLUSTERS[@]}"; do
      CLUSTER_NAME=${CLUSTER}-${CONTEXT}
      if [[ " ${LIST_CLUSTER[@]} " =~ " ${CLUSTER_NAME} " ]]; then
        ./delete_cluster.sh $CLUSTER_NAME
      else
        echo "EKS Cluster ${CLUSTER_NAME} does not exist"
      fi
    done
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
    echo "Specify valid operation"
  ;;
esac 