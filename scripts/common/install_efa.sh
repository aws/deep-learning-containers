#!/bin/bash

set -ex

ARCH=$(uname -m)
case $ARCH in
    x86_64)
        ARCH_DIR="x86_64-linux-gnu"
        ;;
    aarch64)
        ARCH_DIR="aarch64-linux-gnu"
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

function check_libnccl_net_so {

    if [[ "$EFA_VERSION" > "1.44.0" ]] || [[ "$EFA_VERSION" == "1.44.0" ]]; then  # version threshold
        # Newer EFA version - no ARCH_DIR, different filename
        OFI_LIB_DIR="/opt/amazon/ofi-nccl/lib/"
        NCCL_NET_SO="$OFI_LIB_DIR/libnccl-net-ofi.so"
        echo "Using newer EFA path structure"
    else
        # Older EFA version - uses ARCH_DIR
        OFI_LIB_DIR="/opt/amazon/ofi-nccl/lib/${ARCH_DIR}"
        NCCL_NET_SO="$OFI_LIB_DIR/libnccl-net.so"
        echo "Using older EFA path structure with ARCH_DIR: $ARCH_DIR"
    fi

    # Check if file exists
    if [ ! -f "$NCCL_NET_SO" ]; then
        echo "ERROR: $NCCL_NET_SO does not exist"
        return 1
    else
        echo "NCCL OFI plugin found at: $NCCL_NET_SO"
        return 0
    fi
}

function install_efa {
    EFA_VERSION=$1
    OPEN_MPI_PATH="/opt/amazon/openmpi"

    # Install build time tools
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
    check_libnccl_net_so
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

