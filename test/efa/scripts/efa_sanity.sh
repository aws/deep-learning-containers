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

# fi_pingpong loopback is unsupported by EFA design — AWS documents that
# EFA is "designed for network communication between separate instances, not
# for loopback communication on the same instance." fi_info + the multi-node
# NCCL all_reduce test in nccl_allreduce.sh provide real EFA validation.
# Ref: https://www.repost.aws/questions/QUVeCy27EgRR2mUOWnoTbtDQ

# Query local RDMA devices
ibv_devinfo

# Check GDR device (GPU Direct RDMA)
cat /sys/class/infiniband/**/device/p2p | grep 'NVIDIA'
