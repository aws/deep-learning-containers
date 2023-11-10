#!/bin/bash
#/ Usage: ./patch_hosts.sh

set -ex
source ./helper.sh

# Parse parameters from build_param.json config file
EKS_CLUSTERS=($(jq -r '.eks_clusters[]' build_param.json))
CONTEXTS=($(jq -r '.contexts[]' build_param.json))
EKS_VERSION=$(jq -r '.eks_version' build_param.json)
CLUSTER_AUTOSCALAR_IMAGE_VERSION=$(jq -r '.cluster_autoscalar_image_version' build_param.json)

# Create a cluster if the context is predeploy and operation is other than create
if [ "${CONTEXTS}" = "BETA" ] && [ "${OPERATION}" != "create" ]; then
  echo "Create a cluster inorder to perform operations other than creating cluster"
  create_cluster
fi

# Upgrade hosts to latest ami
upgrade_nodegroup

# Cleanup the EKS cluster if the context is BETA
if [ "${CONTEXT}" = "BETA" ]; then
  echo "Delete the pre-deploy cluster"
  delete_cluster
fi
