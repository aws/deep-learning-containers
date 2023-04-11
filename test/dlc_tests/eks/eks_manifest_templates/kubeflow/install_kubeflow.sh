#!/bin/bash
#/ Usage: 
#/ ./install_kubeflow.sh eks_cluster_name region_name

set -ex

# Function to install kustomize
install_kustomize(){
    KUSTOMIZE_VERSION="v4.5.7"
    KUSTOMIZE_URL="https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2F${KUSTOMIZE_VERSION}/kustomize_${KUSTOMIZE_VERSION}_linux_amd64.tar.gz"

    if ! command -v kustomize &> /dev/null
    then
        wget -O /tmp/kustomize_${KUSTOMIZE_VERSION}_linux.tar.gz ${KUSTOMIZE_URL}
        tar -xvf /tmp/kustomize_${KUSTOMIZE_VERSION}_linux.tar.gz -C /tmp/
        if ! [ -x "$command -v sudo" ]; then
            mv /tmp/kustomize /usr/local/bin
        else
            sudo mv /tmp/kustomize /usr/local/bin
        fi
    fi
}

# Function to install kubeflow in EKS cluster using kustomize
install_kubeflow(){

    echo "> Installing kubeflow namespace"
    kustomize build manifests/common/kubeflow-namespace/base | kubectl apply -f -

    echo "> Installing training operators"
    kustomize build manifests/apps/training-operator/upstream/overlays/kubeflow | kubectl apply -f -
}

# Function to remove kubeflow in EKS cluster using kustomize
uninstall_kubeflow(){

    echo "> Uninstalling training operators"
    while ! kustomize build manifests/apps/training-operator/upstream/overlays/kubeflow | kubectl delete -f -; do echo "Retrying to delete training operator resources"; sleep 10; done
    
    echo "> Uninstalling kubeflow namespace"
    while ! kustomize build manifests/common/kubeflow-namespace/base | kubectl delete -f -; do echo "Retrying to delete namespace resources"; sleep 10; done
}

# Function to create directory and download kubeflow components
setup_kubeflow(){
    KUBEFLOW_VERSION="v1.7.0"
    local EKS_CLUSTER_NAME=$1
    DIRECTORY="${HOME}/${EKS_CLUSTER_NAME}"

    if [ -d "${DIRECTORY}" ]; then
        rm -rf ${DIRECTORY};
    fi
        
    mkdir ${DIRECTORY} 
    cd ${DIRECTORY}
    
    # clones manifests from kubeflow github into a folder named manifests
    git clone -b ${KUBEFLOW_VERSION} --single-branch https://github.com/kubeflow/manifests.git
}



# Check for input arguments
if [ $# -ne 2 ]; then
    echo "usage: ./${0} eks_cluster_name region_name"
    exit 1
fi

EKS_CLUSTER_NAME=${1}
OPERATION=${2}

# Check for environment variables
if [ -z "$AWS_REGION" ]; then
  echo "AWS region not configured"
  exit 1
fi

echo "> Installing kustomize"
install_kustomize

echo "> Setup installation directory"
setup_kubeflow ${EKS_CLUSTER_NAME}

# uninstall the manifest if operation is to upgrade kubeflow
if [ "${OPERATION}" = "UPGRADE" ]; then
    echo "> Uninstalling kubeflow manifest"
    uninstall_kubeflow
fi

echo "> Setting up kubeflow"
install_kubeflow

echo "> Installation complete"
