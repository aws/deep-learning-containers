#!/bin/bash
# =============================================================================
# delete_ray_cluster.sh — Delete the Ray Serve deployment.
#
# Usage:
#   bash delete_ray_cluster.sh
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
echo "  Delete Ray Serve Deployment"
echo "=================================================="
echo -e "${NC}"
echo "  Cluster:     $CLUSTER_NAME"
echo "  Namespace:   $NAMESPACE"
echo "  Deployment:  $RAY_CLUSTER_NAME"
echo

read -p "Are you sure you want to delete the deployment? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# ─── Step 1: Delete Deployment ───────────────────────────────────────────────
print_section "Step 1: Deleting Deployment"

kubectl delete deployment "$RAY_CLUSTER_NAME" -n "$NAMESPACE" --ignore-not-found
kubectl delete configmap qwen-serve-code -n "$NAMESPACE" --ignore-not-found
print_success "Deployment '$RAY_CLUSTER_NAME' deleted"

echo "Waiting for pods to terminate..."
for i in $(seq 1 24); do
    if ! kubectl get pods -n "$NAMESPACE" -l app=ray-serve -o name 2>/dev/null | grep -q .; then
        break
    fi
    echo "  Pods still terminating... ($((i * 5))s)"
    sleep 5
done
print_success "All pods terminated"

# ─── Step 2: Clean Up Namespace (if empty) ───────────────────────────────────
print_section "Step 2: Cleaning Up Namespace"

REMAINING_PODS=$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l | tr -d ' ')
if [ "$REMAINING_PODS" = "0" ]; then
    kubectl delete namespace "$NAMESPACE" --ignore-not-found 2>/dev/null || true
    print_success "Namespace '$NAMESPACE' deleted (was empty)"
else
    print_warning "Namespace '$NAMESPACE' still has $REMAINING_PODS pod(s), keeping it"
fi

print_section "Deletion Complete"

ELAPSED_MIN=$((SECONDS / 60))
ELAPSED_SEC=$((SECONDS % 60))
echo -e "\n${BLUE}⏱ Elapsed: ${ELAPSED_MIN}m ${ELAPSED_SEC}s${NC}"
