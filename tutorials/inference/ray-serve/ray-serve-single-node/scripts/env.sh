#!/bin/bash
# =============================================================================
# env.sh — Single source of truth for all shared variables.
#
# Sourced by every script in this project. Has NO side effects.
#
# Usage: source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# =============================================================================

# ─── Core Configuration ─────────────────────────────────────────────────────
export CLUSTER_NAME=${CLUSTER_NAME:-"eks-cluster"}
export REGION=${REGION:-"us-west-2"}
export K8S_VERSION=${K8S_VERSION:-"1.35"}
export AWS_REGION="$REGION"
export AWS_DEFAULT_REGION="$REGION"
export NAMESPACE=${NAMESPACE:-"inference"}

# ─── System Node Group ──────────────────────────────────────────────────────
export SYSTEM_NODE_TYPE=${SYSTEM_NODE_TYPE:-"m7i.xlarge"}
export SYSTEM_NODE_COUNT=${SYSTEM_NODE_COUNT:-1}

# ─── GPU Node Group ─────────────────────────────────────────────────────────
export GPU_NODE_TYPE=${GPU_NODE_TYPE:-"g5.xlarge"}
export GPU_NODE_COUNT=${GPU_NODE_COUNT:-1}
export GPU_NODEGROUP_NAME=${GPU_NODEGROUP_NAME:-"gpu-workers"}

# ─── Container Images ───────────────────────────────────────────────────────
export DLC_IMAGE=${DLC_IMAGE:-"public.ecr.aws/deep-learning-containers/ray:serve-ml-cuda-v1.4"}

# ─── Ray Configuration ──────────────────────────────────────────────────────
export RAY_CLUSTER_NAME=${RAY_CLUSTER_NAME:-"ray-cluster"}
