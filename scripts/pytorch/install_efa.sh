#!/bin/bash
# install_efa.sh — Install AWS EFA + OFI NCCL plugin on Amazon Linux 2023
set -ex

EFA_VERSION="${1:?Usage: install_efa.sh <version>}"
OPEN_MPI_PATH="/opt/amazon/openmpi"

# Install EFA
mkdir -p /tmp/efa && cd /tmp/efa
curl -O "https://s3-us-west-2.amazonaws.com/aws-efa-installer/aws-efa-installer-${EFA_VERSION}.tar.gz"
tar -xf "aws-efa-installer-${EFA_VERSION}.tar.gz"
cd aws-efa-installer
./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify
rm -rf /tmp/efa

# Verify OFI NCCL plugin was installed
OFI_LIB_DIR="/opt/amazon/ofi-nccl/lib64"
if [ ! -f "${OFI_LIB_DIR}/libnccl-net-ofi.so" ]; then
    echo "ERROR: ${OFI_LIB_DIR}/libnccl-net-ofi.so not found after EFA install"
    ls -la "${OFI_LIB_DIR}/" 2>/dev/null || echo "Directory does not exist"
    exit 1
fi
echo "NCCL OFI plugin found at: ${OFI_LIB_DIR}/libnccl-net-ofi.so"

# Configure OpenMPI — allow root execution
mv "${OPEN_MPI_PATH}/bin/mpirun" "${OPEN_MPI_PATH}/bin/mpirun.real"
cat > "${OPEN_MPI_PATH}/bin/mpirun" <<'WRAPPER'
#!/bin/bash
/opt/amazon/openmpi/bin/mpirun.real --allow-run-as-root "$@"
WRAPPER
chmod a+x "${OPEN_MPI_PATH}/bin/mpirun"

echo "hwloc_base_binding_policy = none" >> "${OPEN_MPI_PATH}/etc/openmpi-mca-params.conf"
echo "rmaps_base_mapping_policy = slot" >> "${OPEN_MPI_PATH}/etc/openmpi-mca-params.conf"

ldconfig
