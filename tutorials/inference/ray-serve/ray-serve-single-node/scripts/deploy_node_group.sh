#!/bin/bash
# =============================================================================
# deploy_node_group.sh — Create a GPU node group for Ray workers.
#
# Usage:
#   bash deploy_node_group.sh            # Create GPU node group
#   bash deploy_node_group.sh cleanup    # Delete GPU node group
#
# Override defaults:
#   GPU_NODE_TYPE=g5.2xlarge GPU_NODE_COUNT=4 bash deploy_node_group.sh
# =============================================================================

set -eo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# ─── Colors ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_section() { echo -e "\n${BLUE}=== $1 ===${NC}"; }
print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error()   { echo -e "${RED}✗ $1${NC}"; }

SECONDS=0

# ─── Prerequisite Check ──────────────────────────────────────────────────────
check_prerequisites() {
    local missing=()
    command -v aws &>/dev/null || missing+=("aws")
    command -v eksctl &>/dev/null || missing+=("eksctl")
    command -v kubectl &>/dev/null || missing+=("kubectl")

    if [ ${#missing[@]} -gt 0 ]; then
        print_error "Missing required tools: ${missing[*]}"
        exit 1
    fi

    if ! aws sts get-caller-identity &>/dev/null; then
        print_error "AWS credentials not configured"
        exit 1
    fi

    if ! aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" &>/dev/null; then
        print_error "Cluster '$CLUSTER_NAME' not found in $REGION. Create it first with deploy_cluster.sh"
        exit 1
    fi
    print_success "Prerequisites satisfied"
}

# ─── Cleanup ─────────────────────────────────────────────────────────────────
cleanup_nodegroup() {
    echo -e "${BLUE}"
    echo "=================================================="
    echo "  GPU Node Group Deletion"
    echo "=================================================="
    echo -e "${NC}"
    echo "  Cluster:    $CLUSTER_NAME"
    echo "  Node Group: $GPU_NODEGROUP_NAME"
    echo "  Region:     $REGION"
    echo ""

    read -p "Are you sure you want to delete the GPU node group? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi

    print_section "Deleting GPU Node Group"
    eksctl delete nodegroup \
        --cluster="$CLUSTER_NAME" \
        --region="$REGION" \
        --name="$GPU_NODEGROUP_NAME"

    print_success "GPU node group '$GPU_NODEGROUP_NAME' deleted"

    ELAPSED_MIN=$((SECONDS / 60))
    ELAPSED_SEC=$((SECONDS % 60))
    echo -e "\n${BLUE}⏱ Elapsed: ${ELAPSED_MIN}m ${ELAPSED_SEC}s${NC}"
}

# ─── Main: route subcommand ─────────────────────────────────────────────────
COMMAND=${1:-"deploy"}
if [ "$COMMAND" = "cleanup" ]; then
    check_prerequisites
    cleanup_nodegroup
    exit 0
fi

# ─── Deploy ──────────────────────────────────────────────────────────────────
echo -e "${BLUE}"
echo "=================================================="
echo "  Create GPU Node Group"
echo "=================================================="
echo -e "${NC}"
echo "  Cluster:    $CLUSTER_NAME"
echo "  Region:     $REGION"
echo "  Node Group: $GPU_NODEGROUP_NAME"
echo "  Instance:   $GPU_NODE_TYPE"
echo "  Count:      $GPU_NODE_COUNT"
echo

read -p "Proceed? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

check_prerequisites

# ─── Check for Existing Node Group ──────────────────────────────────────────
print_section "Checking for Existing GPU Node Group"

if eksctl get nodegroup --cluster="$CLUSTER_NAME" --region="$REGION" --name="$GPU_NODEGROUP_NAME" &>/dev/null; then
    print_success "GPU node group '$GPU_NODEGROUP_NAME' already exists"
    kubectl get nodes -l role=gpu-worker
    ELAPSED_MIN=$((SECONDS / 60))
    ELAPSED_SEC=$((SECONDS % 60))
    echo -e "\n${BLUE}⏱ Elapsed: ${ELAPSED_MIN}m ${ELAPSED_SEC}s${NC}"
    exit 0
fi

# ─── Create GPU Node Group ──────────────────────────────────────────────────
print_section "Creating GPU Node Group"
echo "Creating ${GPU_NODE_COUNT}x ${GPU_NODE_TYPE} nodes..."

eksctl create nodegroup \
    --cluster="$CLUSTER_NAME" \
    --region="$REGION" \
    --name="$GPU_NODEGROUP_NAME" \
    --node-type="$GPU_NODE_TYPE" \
    --nodes="$GPU_NODE_COUNT" \
    --nodes-min="$GPU_NODE_COUNT" \
    --nodes-max="$GPU_NODE_COUNT" \
    --node-labels="role=gpu-worker" \
    --managed

print_success "GPU node group created"

# ─── Verify ─────────────────────────────────────────────────────────────────
print_section "Verifying GPU Nodes"
kubectl get nodes -l role=gpu-worker

ELAPSED_MIN=$((SECONDS / 60))
ELAPSED_SEC=$((SECONDS % 60))
echo -e "\n${BLUE}⏱ Elapsed: ${ELAPSED_MIN}m ${ELAPSED_SEC}s${NC}"
