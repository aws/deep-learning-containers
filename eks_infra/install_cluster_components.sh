#!/bin/bash
set -e
CLUSTER_NAME=$1

#install cluster autoscalar
kubectl apply -f eks_infra/cluster-autoscalar-autodiscover.yaml
kubectl -n kube-system annotate deployment.apps/cluster-autoscaler cluster-autoscaler.kubernetes.io/safe-to-evict="false"

sed -e 's/<CLUSTER_NAME>/'"$CLUSTER_NAME"'/' eks_infra/cluster-autoscalar-autodiscover.yaml > /tmp/cluster-autoscalar-autodiscover-$CLUSTER_NAME.yaml &&
kubectl replace -f /tmp/cluster-autoscalar-autodiscover-$CLUSTER_NAME.yaml


#install kubeflow

./eks_infra/install_kubeflow_custom_kfctl.sh $CLUSTER_NAME $AWS_DEFAULT_REGION
