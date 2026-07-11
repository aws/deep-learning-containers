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
# EFA 1.48+ auto-detects NGC containers (via /opt/nvidia/nvidia_entrypoint.sh
# in nvidia/cuda:*-amzn2023 bases) and skips the AL2023 libnccl-ofi RPM.
# Force the standard install with --disable-ngc so aws-ofi-nccl gets installed.
ver_ge() { [ "$1" = "$2" ] || [ "$2" = "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" ]; }
EFA_EXTRA_ARGS=""
ver_ge "$EFA_VERSION" "1.48.0" && EFA_EXTRA_ARGS="--disable-ngc"
./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify ${EFA_EXTRA_ARGS}
rm -rf /tmp/efa

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
