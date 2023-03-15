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
setup_kubeflow(){
    
    # clones manifests from kubeflow github into a folder named manifests
    git clone https://github.com/kubeflow/manifests.git

    echo "> Installing kubeflow namespace"
    kustomize build manifests/common/kubeflow-namespace/base | kubectl apply -f -

    echo "> Installing training operators"
    kustomize build manifests/apps/training-operator/upstream/overlays/kubeflow | kubectl apply -f -
}

# Function to create directory to install kubeflow components
create_dir(){
    local EKS_CLUSTER_NAME=$1
    DIRECTORY="${HOME}/${EKS_CLUSTER_NAME}"

    if [ -d "${DIRECTORY}" ]; then
        rm -rf ${DIRECTORY};
    fi
        
    mkdir ${DIRECTORY} 
    cd ${DIRECTORY}
}

# Check for input arguments
if [ $# -ne 2 ]; then
    echo "usage: ./${0} eks_cluster_name region_name"
    exit 1
fi

EKS_CLUSTER_NAME=${1}
REGION_NAME=${2}

echo "> Set AWS region"
export AWS_REGION=$2

echo "> Setup installation directory"
create_dir ${EKS_CLUSTER_NAME}

echo "> Installing kustomize"
install_kustomize

echo "> Setting up kubeflow"
setup_kubeflow ${REGION_NAME}

echo "> Installation complete"
