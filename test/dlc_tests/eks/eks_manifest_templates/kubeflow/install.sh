#!/bin/bash
set -e

install_kfctl(){
    #install kfctl cli v1.0.2
    if ! command -v kfctl &> /dev/null
    then
        echo "not installed"
        KFCTL_URL=https://github.com/kubeflow/kfctl/releases/download/v1.0.2/kfctl_v1.0.2-0-ga476281_linux.tar.gz
        curl --silent --location ${KFCTL_URL} -o /tmp/kfctl_v1.0.2_linux.tar.gz
        tar -xvf /tmp/kfctl_v1.0.2_linux.tar.gz -C /tmp --strip-components=1
        mv /tmp/kfctl /usr/local/bin
    fi
}

setup_kubeflow(){
    #install kubeflow in EKS cluster
    REGION=$2
    KUBEFLOW_URL="https://raw.githubusercontent.com/aws/deep-learning-containers/master/test/dlc_tests/eks/eks_manifest_templates/kubeflow/kfctl_aws_v1.0.2.yaml"
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

    DIRECTORY="$HOME/$1"

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


echo "> Setup installation directory"
create_dir $1

echo "> Installing kfctl"
install_kfctl 

echo "> Setting up kubeflow"
setup_kubeflow $2 

echo "> Installing mxnet operator"
install_mxnet_operator

echo "> Installing mpi operator"
install_mpi_operator

echo "> Installation complete"