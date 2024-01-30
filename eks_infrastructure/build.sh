#!/bin/bash
#/ Usage: ./build.sh

set -ex
source ./helper.sh

# Check for input arguments
if [ -z "$1" ]; then
  EKS_CLUSTERS=($(jq -r '.eks_clusters[]' build_param.json))
else
  EKS_CLUSTERS=${1}-`date +%s`
fi

if [ -z "$2" ]; then
  CONTEXTS=($(jq -r '.contexts[]' build_param.json))
else
  CONTEXTS=${2}
fi


# Parse parameters from build_param.json config file
OPERATION=$(jq -r '.operation' build_param.json)
EKS_VERSION=$(jq -r '.eks_version' build_param.json)
CLUSTER_AUTOSCALAR_IMAGE_VERSION=$(jq -r '.cluster_autoscalar_image_version' build_param.json)

# Create a cluster if the context is predeploy and operation is other than create
if [ "${CONTEXTS}" = "BETA" ] && [ "${OPERATION}" != "create" ]; then
  echo "Create a cluster inorder to perform operations other than creating cluster"
  create_cluster
fi


case ${OPERATION} in

create)
  create_cluster
  ;;

upgrade_cluster)
  upgrade_cluster
  ;;

upgrade_nodegroup)
  upgrade_nodegroup
  ;;

upgrade_kubeflow)
  upgrade_kubeflow
  ;;

delete)
  delete_cluster
  ;;

new_operation)
  new_operation
  ;;
*)
  echo "Specify valid operation"
  ;;
esac

# Cleanup the EKS cluster if the context is BETA
if [ "${CONTEXT}" = "BETA" ]; then
  echo "Delete the pre-deploy cluster"
  delete_cluster
fi
