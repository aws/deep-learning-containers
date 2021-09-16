#!/bin/bash
#/ Usage: ./build.sh

set -ex

# Parse parameters from build_param.json config file
OPERATION=$(jq -r '.operation' eks_infra/build_param.json)
EKS_CLUSTERS=($(jq -r '.eks_clusters[]' eks_infra/build_param.json))
CONTEXTS=($(jq -r '.contexts[]' eks_infra/build_param.json))
EKS_VERSION=$(jq -r '.eks_version' eks_infra/build_param.json)
CLUSTER_AUTOSCALAR_IMAGE_VERSION=$(jq -r '.cluster_autoscalar_image_version' eks_infra/build_param.json)

# Create operation function
#
# Invokes create_cluster.sh script to create EKS cluster nodegroups and namespaces.
# Invokes add_iam_identity.sh script to create RBAC rules for test IAM role. Add cluster manager IAM role
# and test IAM role authentication to the cluster.
# Invokes install_cluster_components.sh script to install cluster autoscalar and kubeflow components in the cluster.
function create_cluster() {
  cd eks_infra

  for CONTEXT in "${CONTEXTS[@]}"; do
    for CLUSTER in "${EKS_CLUSTERS[@]}"; do
      CLUSTER_NAME=${CLUSTER}-${CONTEXT}

      if ! check_cluster_status $CLUSTER_NAME; then
        ./create_cluster.sh $CLUSTER_NAME $EKS_VERSION
        ./add_iam_identity.sh $CLUSTER_NAME
        ./install_cluster_components.sh $CLUSTER_NAME $CLUSTER_AUTOSCALAR_IMAGE_VERSION
      else
        echo "EKS Cluster :: ${CLUSTER_NAME} :: already exists. Skipping create operation."
      fi
    done
  done
}

# Upgrade operation function
#
# Invokes upgrade_cluster.sh script to upgrade the EKS cluster in the following order
# 1. Scale cluster autoscalar to 0 to prevent unwanted scaling
# 2. Upgrade EKS control plane
# 3. Upgrade EKS nodegroups:
#    i) Delete Exisiting nodegroups
#    ii) Create nodegroups with updated configuration
#
# 4. Upgrade core k8s components
# 5. Scale cluster autoscalar back to 1
function upgrade_cluster() {

  cd eks_infra

  for CONTEXT in "${CONTEXTS[@]}"; do
    for CLUSTER in "${EKS_CLUSTERS[@]}"; do
      CLUSTER_NAME=${CLUSTER}-${CONTEXT}
      if check_cluster_status $CLUSTER_NAME; then
        ./upgrade_cluster.sh $CLUSTER_NAME $EKS_VERSION $CLUSTER_AUTOSCALAR_IMAGE_VERSION
      else
        echo "EKS Cluster :: ${CLUSTER_NAME} :: does not exists. Skipping upgrade operation."
      fi
    done
  done

}

# Delete operation function
#
# Invokes delete_cluster.sh script to delete EKS cluster, nodegroups and other related components
function delete_cluster() {

  cd eks_infra
  for CONTEXT in "${CONTEXTS[@]}"; do
    for CLUSTER in "${EKS_CLUSTERS[@]}"; do
      CLUSTER_NAME=${CLUSTER}-${CONTEXT}
      if check_cluster_status $CLUSTER_NAME; then
        ./delete_cluster.sh $CLUSTER_NAME
      else
        echo "EKS Cluster :: ${CLUSTER_NAME} :: does not exists. Skipping delete operation."
      fi
    done
  done

}

function check_cluster_status() {
  aws eks describe-cluster --name ${1} --region ${AWS_REGION} --query cluster.status --out text | grep -q ACTIVE
}

case ${OPERATION} in

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
