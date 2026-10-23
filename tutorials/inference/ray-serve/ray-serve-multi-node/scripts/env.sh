#!/bin/bash
# env.sh - Single source of truth for all shared variables. No side effects.
# Usage: source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

export CLUSTER_NAME=${CLUSTER_NAME:-"ray-llm-multinode"}
export REGION=${REGION:-"sa-east-1"}
export K8S_VERSION=${K8S_VERSION:-"1.35"}
export AWS_REGION="$REGION"
export AWS_DEFAULT_REGION="$REGION"
export NAMESPACE=${NAMESPACE:-"inference"}

export SYSTEM_NODE_TYPE=${SYSTEM_NODE_TYPE:-"m7i.xlarge"}
export SYSTEM_NODE_COUNT=${SYSTEM_NODE_COUNT:-1}

# Smallest g6 with EFA (1 interface). EFA cannot cross AZs, so the GPU node
# group is pinned to a single AZ; GPU_AZ is auto-discovered when left empty.
export GPU_NODE_TYPE=${GPU_NODE_TYPE:-"g6.8xlarge"}
export GPU_NODE_COUNT=${GPU_NODE_COUNT:-2}
export GPU_NODEGROUP_NAME=${GPU_NODEGROUP_NAME:-"gpu-workers"}
export GPU_AZ=${GPU_AZ:-""}

export DLC_IMAGE=${DLC_IMAGE:-"763104351884.dkr.ecr.${REGION}.amazonaws.com/ray:serve-llm-cuda-v1.0"}

export KUBERAY_VERSION=${KUBERAY_VERSION:-"1.4.0"}
export RAY_VERSION=${RAY_VERSION:-"2.58.0"}
export RAY_SERVICE_NAME=${RAY_SERVICE_NAME:-"ray-llm"}

export MODEL_ID=${MODEL_ID:-"qwen3.5-9b"}
export MODEL_SOURCE=${MODEL_SOURCE:-"Qwen/Qwen3.5-9B"}