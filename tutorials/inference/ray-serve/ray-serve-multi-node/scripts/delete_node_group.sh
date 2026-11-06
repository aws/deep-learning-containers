#!/bin/bash
# delete_node_group.sh — Delete the GPU node group.
# Usage: bash delete_node_group.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
source "$SCRIPT_DIR/_lib.sh"

SECONDS=0

for cmd in aws eksctl; do
    command -v "$cmd" &>/dev/null || { print_error "Missing required tool: $cmd"; exit 1; }
done
check_credentials

print_section "Checking GPU Node Group Status"
NODEGROUP_STATUS=$(get_nodegroup_status "$GPU_NODEGROUP_NAME")

case "$NODEGROUP_STATUS" in
    NOT_FOUND)
        print_success "Node group '$GPU_NODEGROUP_NAME' does not exist in cluster '$CLUSTER_NAME'. Nothing to delete."
        exit 0
        ;;
    UNKNOWN)
        print_error "Could not determine status of node group '$GPU_NODEGROUP_NAME' (repeated API errors -- check your AWS session/credentials and re-run)."
        exit 1
        ;;
    *)
        print_success "Node group '$GPU_NODEGROUP_NAME' found (status: $NODEGROUP_STATUS)"
        ;;
esac

echo -e "${BLUE}"
echo "=================================================="
echo "  GPU Node Group Deletion"
echo "=================================================="
echo -e "${NC}"
echo "  Cluster:    $CLUSTER_NAME"
echo "  Node Group: $GPU_NODEGROUP_NAME"
echo "  Region:     $REGION"
echo

read -p "Are you sure you want to delete the GPU node group? (y/N): " -n 1 -r
echo
[[ $REPLY =~ ^[Yy]$ ]] || { echo "Cancelled."; exit 0; }

print_section "Deleting GPU Node Group"
delete_nodegroup_and_wait "$GPU_NODEGROUP_NAME"

print_success "GPU node group '$GPU_NODEGROUP_NAME' deleted"
print_elapsed