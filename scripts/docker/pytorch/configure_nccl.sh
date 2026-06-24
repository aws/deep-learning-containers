#!/bin/bash
# configure_nccl.sh — Write NCCL configuration
# TEMP: no-op edit to force PyTorch PR build+test (validate re-enable; revert before merge)
set -ex

cat > /etc/nccl.conf <<'EOF'
NCCL_DEBUG=INFO
NCCL_SOCKET_IFNAME=^docker0,lo
EOF
