#!/bin/bash
# Ported from V1: test/v2/ec2/efa/testEFASanity
# Verify EFA components are installed and functional on a single node.
set -ex

export PATH=/opt/amazon/efa/bin:$PATH

# Check Libfabric EFA provider
fi_info -p efa
fi_info -p efa -t FI_EP_RDM | grep 'FI_EP_RDM'

# Check if ib_uverbs kernel module is loaded (use /sys/module — AL2023 minimal has no `lsmod`)
test -d /sys/module/ib_uverbs

# Run fi_pingpong over EFA loopback
/test/efa/scripts/efa_pingpong.sh

# Query local RDMA devices
ibv_devinfo

# Check GDR device (GPU Direct RDMA)
cat /sys/class/infiniband/**/device/p2p | grep 'NVIDIA'
