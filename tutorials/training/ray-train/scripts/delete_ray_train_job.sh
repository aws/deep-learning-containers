#!/bin/bash
# delete_ray_train_job.sh - Delete the multi-node RayCluster.
# Usage: bash delete_ray_train_job.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
source "$SCRIPT_DIR/_lib.sh"

SECONDS=0

check_kubectl_prerequisites

if ! kubectl get raycluster "$RAY_CLUSTER_NAME" -n "$NAMESPACE" &>/dev/null; then
    print_success "RayCluster '$RAY_CLUSTER_NAME' not found in namespace '$NAMESPACE'. Nothing to delete."
    exit 0
fi

echo -e "${BLUE}"
echo "=================================================="
echo "  RayCluster Deletion"
echo "=================================================="
echo -e "${NC}"
echo "  RayCluster: $RAY_CLUSTER_NAME"
echo "  Namespace:  $NAMESPACE"
echo

read -p "Are you sure you want to delete the RayCluster? (y/N): " -n 1 -r
echo
[[ $REPLY =~ ^[Yy]$ ]] || { echo "Cancelled."; exit 0; }

print_section "Deleting RayCluster"
kubectl delete raycluster "$RAY_CLUSTER_NAME" -n "$NAMESPACE" --ignore-not-found --timeout=180s
kubectl wait --for=delete pod -l "ray.io/cluster=${RAY_CLUSTER_NAME}" -n "$NAMESPACE" --timeout=240s 2>/dev/null || true

if kubectl get raycluster "$RAY_CLUSTER_NAME" -n "$NAMESPACE" &>/dev/null; then
    print_error "RayCluster '$RAY_CLUSTER_NAME' still exists after delete. Check 'kubectl describe raycluster $RAY_CLUSTER_NAME -n $NAMESPACE'."
    exit 1
fi
print_success "RayCluster '$RAY_CLUSTER_NAME' deleted"

REMAINING_PODS=$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l | tr -d ' ')
if [ "$REMAINING_PODS" = "0" ]; then
    kubectl delete namespace "$NAMESPACE" --ignore-not-found 2>/dev/null || true
    print_success "Namespace '$NAMESPACE' deleted (was empty)"
else
    print_warning "Namespace '$NAMESPACE' left in place ($REMAINING_PODS pod(s) still present)."
fi

print_elapsed
