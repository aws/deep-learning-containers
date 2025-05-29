#!/bin/bash

set -ex

function install_efa {
    EFA_VERSION=$1
    OPEN_MPI_PATH="/opt/amazon/openmpi"
    
    # Install build time tools
    echo "deb https://mirror.pilotfiber.com/ubuntu/ noble main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb-src https://mirror.pilotfiber.com/ubuntu/ noble main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirror.pilotfiber.com/ubuntu/ noble-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src https://mirror.pilotfiber.com/ubuntu/ noble-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirror.pilotfiber.com/ubuntu/ noble-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src https://mirror.pilotfiber.com/ubuntu/ noble-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    apt-get update
    apt-get install -y --allow-change-held-packages --no-install-recommends \
        curl \
        build-essential \
        cmake \
        git

    # Install EFA
    mkdir /tmp/efa
    cd /tmp/efa
    curl -O https://s3-us-west-2.amazonaws.com/aws-efa-installer/aws-efa-installer-${EFA_VERSION}.tar.gz
    tar -xf aws-efa-installer-${EFA_VERSION}.tar.gz
    cd aws-efa-installer
    ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify
    rm -rf /tmp/efa
    # Configure Open MPI and configure NCCL parameters
    mv ${OPEN_MPI_PATH}/bin/mpirun ${OPEN_MPI_PATH}/bin/mpirun.real
    echo '#!/bin/bash' > ${OPEN_MPI_PATH}/bin/mpirun
    echo "${OPEN_MPI_PATH}/bin/mpirun.real --allow-run-as-root \"\$@\"" >> ${OPEN_MPI_PATH}/bin/mpirun
    chmod a+x ${OPEN_MPI_PATH}/bin/mpirun
    echo "hwloc_base_binding_policy = none" >> ${OPEN_MPI_PATH}/etc/openmpi-mca-params.conf
    echo "rmaps_base_mapping_policy = slot" >> ${OPEN_MPI_PATH}/etc/openmpi-mca-params.conf
    echo NCCL_DEBUG=INFO >> /etc/nccl.conf
    echo NCCL_SOCKET_IFNAME=^docker0,lo >> /etc/nccl.conf
    
    # Install OpenSSH for MPI to communicate between containers, allow OpenSSH to talk to containers without asking for confirmation
    apt-get install -y --no-install-recommends \
        openssh-client \
        openssh-server
    mkdir -p /var/run/sshd
    cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config
    # Configure OpenSSH so that nodes can communicate with each other
    mkdir -p /var/run/sshd
    sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
    rm -rf /root/.ssh/
    mkdir -p /root/.ssh/
    ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa
    cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys
    printf "Host *\n StrictHostKeyChecking no\n" >> /root/.ssh/config
    
    # Install NCCL-Test
    # cd /tmp
    # git clone -b ${NCCL_TEST_VERSION} https://github.com/NVIDIA/nccl-tests.git
    # cd nccl-tests
    # make MPI=1 MPI_HOME=${OPEN_MPI_PATH} CUDA_HOME=${CUDA_HOME} NCCL_HOME=/usr/local
    # mv build/ /usr/local/nccl-test
    # rm -rf /tmp/nccl-tests
    
    # Remove build time tools
    # apt-get remove -y
    #     curl
    #     build-essential
    #     cmake
    #     git
    
    # Cleanup
    apt-get clean
    apt-get autoremove -y
    rm -rf /var/lib/apt/lists/*
    ldconfig
}

# idiomatic parameter and option handling in sh
while test $# -gt 0
do
    case "$1" in
    [0-9].[0-9]*.[0-9]*) install_efa $1; 
        ;;
    *) echo "bad argument $1"; exit 1
        ;;
    esac
    shift
done
