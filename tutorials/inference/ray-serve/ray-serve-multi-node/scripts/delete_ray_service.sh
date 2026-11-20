#!/bin/bash
# delete_ray_service.sh - Delete the multi-node RayService.
# Usage: bash delete_ray_service.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
source "$SCRIPT_DIR/_lib.sh"

SECONDS=0

check_kubectl_prerequisites

if ! kubectl get rayservice "$RAY_SERVICE_NAME" -n "$NAMESPACE" &>/dev/null; then
    print_success "RayService '$RAY_SERVICE_NAME' not found in namespace '$NAMESPACE'. Nothing to delete."
    exit 0
fi

echo -e "${BLUE}"
echo "=================================================="
echo "  RayService Deletion"
echo "=================================================="
echo -e "${NC}"
echo "  RayService: $RAY_SERVICE_NAME"
echo "  Namespace:  $NAMESPACE"
echo

read -p "Are you sure you want to delete the RayService? (y/N): " -n 1 -r
echo
[[ $REPLY =~ ^[Yy]$ ]] || { echo "Cancelled."; exit 0; }

print_section "Deleting RayService"
kubectl delete rayservice "$RAY_SERVICE_NAME" -n "$NAMESPACE" --ignore-not-found --timeout=180s
kubectl wait --for=delete pod -l ray.io/cluster -n "$NAMESPACE" --timeout=240s 2>/dev/null || true

if kubectl get rayservice "$RAY_SERVICE_NAME" -n "$NAMESPACE" &>/dev/null; then
    print_error "RayService '$RAY_SERVICE_NAME' still exists after delete. Check 'kubectl describe rayservice $RAY_SERVICE_NAME -n $NAMESPACE'."
    exit 1
fi
print_success "RayService '$RAY_SERVICE_NAME' deleted"

REMAINING_PODS=$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l | tr -d ' ')
if [ "$REMAINING_PODS" = "0" ]; then
    kubectl delete namespace "$NAMESPACE" --ignore-not-found 2>/dev/null || true
    print_success "Namespace '$NAMESPACE' deleted (was empty)"
else
    print_warning "Namespace '$NAMESPACE' left in place ($REMAINING_PODS pod(s) still present)."
fi

print_elapsed