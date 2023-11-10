#!/bin/bash
#/ Usage: helper file to source functions

# Create operation function
#
# Invokes create_cluster.sh script to create EKS cluster nodegroups and namespaces.
# Invokes add_iam_identity.sh script to create RBAC rules for test IAM role. Add cluster manager IAM role
# and test IAM role authentication to the cluster.
# Invokes install_cluster_components.sh script to install cluster autoscalar and kubeflow components in the cluster.
function create_cluster() {

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

# Upgrade cluster operation function
#
# Invokes upgrade_operation.sh script to upgrade the EKS cluster in the following order
# 1. Scale cluster autoscalar to 0 to prevent unwanted scaling
# 2. Upgrade EKS control plane
# 3. Upgrade EKS nodegroups:
#    i) Delete Exisiting nodegroups
#    ii) Create nodegroups with updated configuration
# 4. Upgrade core k8s components
# 5. Scale cluster autoscalar back to 1
function upgrade_cluster() {
  TARGET="CLUSTER"
  for CONTEXT in "${CONTEXTS[@]}"; do
    for CLUSTER in "${EKS_CLUSTERS[@]}"; do
      CLUSTER_NAME=${CLUSTER}-${CONTEXT}
      if check_cluster_status $CLUSTER_NAME; then
        ./upgrade_operation.sh $TARGET $CLUSTER_NAME $EKS_VERSION $CLUSTER_AUTOSCALAR_IMAGE_VERSION
      else
        echo "EKS Cluster :: ${CLUSTER_NAME} :: does not exists. Skipping upgrade operation."
      fi
    done
  done

}

# Upgrade nodegroup operation function
#
# Invokes upgrade_operation.sh script to upgrade the EKS nodegroup for a cluster
function upgrade_nodegroup() {
  TARGET="NODEGROUP"
  for CONTEXT in "${CONTEXTS[@]}"; do
    for CLUSTER in "${EKS_CLUSTERS[@]}"; do
      CLUSTER_NAME=${CLUSTER}-${CONTEXT}
      if check_cluster_status $CLUSTER_NAME; then
        ./upgrade_operation.sh $TARGET $CLUSTER_NAME $EKS_VERSION
      else
        echo "EKS Cluster :: ${CLUSTER_NAME} :: does not exists. Skipping upgrade operation."
      fi
    done
  done

}

# Upgrade kubeflow version
#
# Invokes upgrade_kubeflow script to upgrade kubeflow manifests on each EKS cluster
function upgrade_kubeflow(){
  for CONTEXT in "${CONTEXTS[@]}"; do
    for CLUSTER in "${EKS_CLUSTERS[@]}"; do
      CLUSTER_NAME=${CLUSTER}-${CONTEXT}
      if check_cluster_status $CLUSTER_NAME; then
        ./upgrade_kubeflow.sh $CLUSTER_NAME
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

# new operation function to perform one time operation on the cluster
# Once deployed, the contents of the new_operation.sh should be moved to a dedicated script
# to allow upcoming features to be configured in the new_operation.sh script
# Invokes new_operation.sh script to add graviton nodegroup on the EKS cluster
function new_operation() {

  for CONTEXT in "${CONTEXTS[@]}"; do
    for CLUSTER in "${EKS_CLUSTERS[@]}"; do
      CLUSTER_NAME=${CLUSTER}-${CONTEXT}
      if check_cluster_status $CLUSTER_NAME; then
        ./new_operation.sh $CLUSTER_NAME $EKS_VERSION
      else
        echo "EKS Cluster :: ${CLUSTER_NAME} :: does not exists. Skipping delete operation."
      fi
    done
  done

}

function check_cluster_status() {
  aws eks describe-cluster --name ${1} --region ${AWS_REGION} --query cluster.status --out text | grep -q ACTIVE
}

function lol(){
  echo "lol"
}
