#!/bin/bash
# =============================================================================
# delete_node_group.sh — Delete the GPU node group.
#
# Usage:
#   bash delete_node_group.sh
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

echo -e "${BLUE}"
echo "=================================================="
echo "  Delete GPU Node Group"
echo "=================================================="
echo -e "${NC}"
echo "  Cluster:    $CLUSTER_NAME"
echo "  Region:     $REGION"
echo "  Node Group: $GPU_NODEGROUP_NAME"
echo

read -p "Are you sure you want to delete the GPU node group? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# ─── Delete GPU Node Group ───────────────────────────────────────────────────
print_section "Deleting GPU Node Group"

if ! eksctl get nodegroup --cluster="$CLUSTER_NAME" --region="$REGION" --name="$GPU_NODEGROUP_NAME" &>/dev/null; then
    print_warning "Node group '$GPU_NODEGROUP_NAME' not found in cluster '$CLUSTER_NAME'"
    exit 0
fi

eksctl delete nodegroup \
    --cluster="$CLUSTER_NAME" \
    --region="$REGION" \
    --name="$GPU_NODEGROUP_NAME"

print_success "GPU node group '$GPU_NODEGROUP_NAME' deleted"

ELAPSED_MIN=$((SECONDS / 60))
ELAPSED_SEC=$((SECONDS % 60))
echo -e "\n${BLUE}⏱ Elapsed: ${ELAPSED_MIN}m ${ELAPSED_SEC}s${NC}"
