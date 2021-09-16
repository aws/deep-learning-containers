#!/bin/bash
#/ Usage: 
#/ export AWS_REGION=<AWS-Region>
#/ ./install_cluster_components.sh eks_cluster_name cluster_autoscalar_image_version

set -ex

# Function to install kfctl cli
install_kfctl(){

    KFCTL_VERSION="v1.0.2"
    S3_URL="https://kubeflow-kfctl-binary.s3-us-west-2.amazonaws.com/kfctl_v1.0.2-1-g93e95e1_linux.tar.gz"

    if ! command -v kfctl &> /dev/null
    then
        wget -O /tmp/kfctl_${KFCTL_VERSION}_linux.tar.gz ${S3_URL}
        tar -xvf /tmp/kfctl_${KFCTL_VERSION}_linux.tar.gz -C /tmp --strip-components=1
        if ! [ -x "$(command -v sudo)" ]; then
            mv /tmp/kfctl /usr/local/bin
        else
            sudo mv /tmp/kfctl /usr/local/bin
        fi
    fi
}

# Function to install kubeflow in EKS cluster 
setup_kubeflow(){

    local REGION=$1
    KUBEFLOW_URL="https://raw.githubusercontent.com/aws/deep-learning-containers/master/test/dlc_tests/eks/eks_manifest_templates/kubeflow/kfctl_aws.yaml"
    CONFIG_FILE=kfctl_aws.yaml
    wget -O ${CONFIG_FILE} ${KUBEFLOW_URL} 
    sed -i -e 's/<REGION>/'"${REGION}"'/' ${CONFIG_FILE}
    kfctl apply -V -f ${CONFIG_FILE}
}

# Function to install mpi operator in EKS cluster
install_mpi_operator() {
    #install mpi operator in EKS cluster
    MPI_OPERATOR_URL=https://raw.githubusercontent.com/kubeflow/mpi-operator/v0.3.0/deploy/v1alpha2/mpi-operator.yaml
    wget -O mpi-operator.yaml ${MPI_OPERATOR_URL}
    kubectl create -f mpi-operator.yaml
}

# Function to install mxnet operator in EKS cluster
install_mxnet_operator() {

    git clone https://github.com/kubeflow/mxnet-operator.git
    kubectl create -k mxnet-operator/manifests/overlays/v1beta1/
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

echo "> Installing kfctl"
install_kfctl 

echo "> Setting up kubeflow"
setup_kubeflow ${REGION_NAME}

echo "> Installing mxnet operator"
install_mxnet_operator

echo "> Installing mpi operator"
install_mpi_operator

echo "> Installation complete"
