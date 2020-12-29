#!/bin/bash
set -e

#create eks cluster

eksctl create cluster --name <CLUSTER_NAME> --nodegroup-name <NG-NAME> --nodes-min 1 --nodes-max 1 --node-type <INSTANCE_TYPE>

#create nodegroups

eksctl create nodegroup -f eks_infra/nodegroup.yaml

#install cluster autoscalar



#install kubeflow

bash ./eks_infra/install_kubeflow_custom_kfctl.sh


#cluster upgrades
#https://eksctl.io/usage/cluster-upgrade/
