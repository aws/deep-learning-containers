#!/bin/bash

#parse parameters

operation=$(jq -r '.operation' eks_infra/build_param.json)
eks_clusters=$(jq -r '.eks_clusters | .[]' eks_infra/build_param.json)
list_cluster=$(eksctl get cluster -o json | jq -r '.[].metadata.name')
#list_node_groups=$(eksctl get nodegroup --cluster eks-cluster -o json | jq -r '.[].StackName')

case $operation in 
  
  create)
    for cluster in $eks_clusters; do
      echo "cluster $cluster"
      if [[ ! " ${list_cluster[@]} " =~ " ${CLUSTER} " ]]; then
        cd eks_infra
        ./create_cluster.sh $cluster
        ./install_cluster_components.sh $cluster
      else
        echo "eks cluster ${CLUSTER} already exist"
        echo "skipping creation of cluster/ng and component installation"
      fi
    done
  ;;

  upgrade)
    echo "upgrade 1"
  ;;

  delete)
    echo "delete 1"
  ;;

  *)
    echo "something else"
  ;;
esac