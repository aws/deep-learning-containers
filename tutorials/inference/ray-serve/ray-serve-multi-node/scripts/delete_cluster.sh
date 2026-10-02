#!/bin/bash
# delete_cluster.sh — Delete the EKS cluster and all associated resources.
# Usage: bash delete_cluster.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
source "$SCRIPT_DIR/_lib.sh"

SECONDS=0

for cmd in aws eksctl; do
    command -v "$cmd" &>/dev/null || { print_error "Missing required tool: $cmd"; exit 1; }
done
check_credentials

print_section "Checking Cluster Status"
CLUSTER_STATUS=$(get_cluster_status)

case "$CLUSTER_STATUS" in
    NOT_FOUND)
        print_success "Cluster '$CLUSTER_NAME' does not exist in $REGION. Nothing to delete."
        exit 0
        ;;
    UNKNOWN)
        print_error "Could not determine whether cluster '$CLUSTER_NAME' exists (repeated API errors -- check your AWS session/credentials and re-run)."
        exit 1
        ;;
    DELETING)
        print_warning "Cluster '$CLUSTER_NAME' is already being deleted. Waiting for it to finish..."
        ;;
    *)
        print_success "Cluster '$CLUSTER_NAME' found (status: $CLUSTER_STATUS)"
        echo -e "${BLUE}"
        echo "=================================================="
        echo "  Delete EKS Cluster"
        echo "=================================================="
        echo -e "${NC}"
        echo "  Cluster: $CLUSTER_NAME"
        echo "  Region:  $REGION"
        echo
        read -p "Are you sure you want to delete this cluster? (y/N): " -n 1 -r
        echo
        [[ $REPLY =~ ^[Yy]$ ]] || { echo "Cancelled."; exit 0; }

        echo "Deleting EKS cluster '$CLUSTER_NAME'... (this takes 10-15 minutes)"
        wait_for_no_active_update
        eksctl delete cluster --name "$CLUSTER_NAME" --region "$REGION"
        ;;
esac

delete_cf_stack_and_wait "eksctl-${CLUSTER_NAME}-cluster"

FINAL_STATUS=$(get_cluster_status)
if [ "$FINAL_STATUS" != "NOT_FOUND" ]; then
    print_error "Cluster '$CLUSTER_NAME' is still in state '$FINAL_STATUS' after deletion. It may still be tearing down -- check the EKS console, or re-run this script in a few minutes."
    exit 1
fi

print_success "EKS cluster '$CLUSTER_NAME' deleted"
print_elapsed