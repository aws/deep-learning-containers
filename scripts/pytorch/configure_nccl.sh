#!/bin/bash
# configure_nccl.sh — Write NCCL configuration
set -ex

cat > /etc/nccl.conf <<'EOF'
NCCL_DEBUG=INFO
NCCL_SOCKET_IFNAME=^docker0,lo
EOF
