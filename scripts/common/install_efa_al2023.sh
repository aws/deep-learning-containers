#!/bin/bash

set -ex

ARCH=$(uname -m)

function check_libnccl_net_so {
    if [[ "$EFA_VERSION" > "1.44.0" ]] || [[ "$EFA_VERSION" == "1.44.0" ]]; then
        OFI_LIB_DIR="/opt/amazon/ofi-nccl/lib/"
        NCCL_NET_SO="$OFI_LIB_DIR/libnccl-net-ofi.so"
    else
        case $ARCH in
            x86_64)  ARCH_DIR="x86_64-linux-gnu" ;;
            aarch64) ARCH_DIR="aarch64-linux-gnu" ;;
            *)       echo "Unsupported architecture: $ARCH"; exit 1 ;;
        esac
        OFI_LIB_DIR="/opt/amazon/ofi-nccl/lib/${ARCH_DIR}"
        NCCL_NET_SO="$OFI_LIB_DIR/libnccl-net.so"
    fi

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

    # Install build-time dependencies
    dnf install -y --setopt=install_weak_deps=False \
        curl tar gzip gcc gcc-c++ cmake make git

    # Install EFA
    mkdir /tmp/efa
    cd /tmp/efa
    curl -O https://s3-us-west-2.amazonaws.com/aws-efa-installer/aws-efa-installer-${EFA_VERSION}.tar.gz
    tar -xf aws-efa-installer-${EFA_VERSION}.tar.gz
    cd aws-efa-installer
    ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify
    rm -rf /tmp/efa

    # Configure Open MPI and NCCL parameters
    mv ${OPEN_MPI_PATH}/bin/mpirun ${OPEN_MPI_PATH}/bin/mpirun.real
    echo '#!/bin/bash' > ${OPEN_MPI_PATH}/bin/mpirun
    echo "${OPEN_MPI_PATH}/bin/mpirun.real --allow-run-as-root \"\$@\"" >> ${OPEN_MPI_PATH}/bin/mpirun
    chmod a+x ${OPEN_MPI_PATH}/bin/mpirun
    echo "hwloc_base_binding_policy = none" >> ${OPEN_MPI_PATH}/etc/openmpi-mca-params.conf
    echo "rmaps_base_mapping_policy = slot" >> ${OPEN_MPI_PATH}/etc/openmpi-mca-params.conf
    echo NCCL_DEBUG=INFO >> /etc/nccl.conf
    echo NCCL_SOCKET_IFNAME=^docker0,lo >> /etc/nccl.conf

    # Install OpenSSH for MPI inter-container communication
    dnf install -y --setopt=install_weak_deps=False openssh-clients openssh-server
    mkdir -p /var/run/sshd
    cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config
    mkdir -p /var/run/sshd
    rm -rf /root/.ssh/
    mkdir -p /root/.ssh/
    ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa
    cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys
    printf "Host *\n StrictHostKeyChecking no\n" >> /root/.ssh/config

    # Cleanup
    dnf clean all
    rm -rf /var/cache/dnf
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
