#!/bin/bash
set -e

install_kfctl(){
    #install kfctl cli
    KFCTL_VERSION="v1.0.2"

    if ! command -v kfctl &> /dev/null
    then
        KFCTL_URL=https://github.com/kubeflow/kfctl/releases/download/${KFCTL_VERSION}/kfctl_${KFCTL_VERSION}-0-ga476281_linux.tar.gz
        curl --silent --location ${KFCTL_URL} -o /tmp/kfctl_${KFCTL_VERSION}_linux.tar.gz
        tar -xvf /tmp/kfctl_${KFCTL_VERSION}_linux.tar.gz -C /tmp --strip-components=1
        mv /tmp/kfctl /usr/local/bin
    fi
}

setup_kubeflow(){
    #install kubeflow in EKS cluster
    local REGION=$1
    KUBEFLOW_URL="https://raw.githubusercontent.com/aws/deep-learning-containers/master/test/dlc_tests/eks/eks_manifest_templates/kubeflow/kfctl_aws.yaml"
    CONFIG_FILE=kfctl_aws.yaml
    wget -O ${CONFIG_FILE} ${KUBEFLOW_URL} 
    sed -i -e 's/<REGION>/'"$REGION"'/' ${CONFIG_FILE}
    kfctl apply -V -f ${CONFIG_FILE}
}

install_mpi_operator() {
    #install mpi operator in EKS cluster
    MPI_OPERATOR_URL=https://raw.githubusercontent.com/kubeflow/mpi-operator/v0.2.3/deploy/v1alpha2/mpi-operator.yaml
    wget -O mpi-operator.yaml ${MPI_OPERATOR_URL}
    kubectl create -f mpi-operator.yaml
}

install_mxnet_operator() {
    #install mxnet operator in EKS cluster
    git clone https://github.com/kubeflow/mxnet-operator.git
    kubectl create -k mxnet-operator/manifests/overlays/v1beta1/
}

create_dir(){
    local EKS_CLUSTER_NAME=$1
    DIRECTORY="$HOME/$EKS_CLUSTER_NAME"

    if [ -d "$DIRECTORY" ]; then
        rm -rf $DIRECTORY;
    fi
        
    mkdir $DIRECTORY 
    cd $DIRECTORY
}

if [ $# -ne 2 ]; then
    echo $0: usage: ./install_kubeflow eks_cluster_name region_name
    exit 1
fi

eks_cluster_name=$1
region_name=$2

echo "> Setup installation directory"
create_dir $eks_cluster_name

echo "> Installing kfctl"
install_kfctl 

echo "> Setting up kubeflow"
setup_kubeflow $region_name

echo "> Installing mxnet operator"
install_mxnet_operator

echo "> Installing mpi operator"
install_mpi_operator

echo "> Installation complete"
