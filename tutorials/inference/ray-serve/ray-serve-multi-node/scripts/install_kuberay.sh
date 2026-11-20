#!/bin/bash
# install_kuberay.sh - Install the KubeRay operator via Helm.
#
# Usage:
#   bash install_kuberay.sh            # Install operator
#   bash install_kuberay.sh cleanup    # Uninstall operator
#
# Prerequisites: EKS cluster running (deploy_cluster.sh), helm installed.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
source "$SCRIPT_DIR/_lib.sh"

SECONDS=0
KUBERAY_NAMESPACE="kuberay-operator"

check_prerequisites() {
    command -v helm &>/dev/null || { print_error "helm not found. Install: https://helm.sh/docs/intro/install/"; exit 1; }
    check_kubectl_prerequisites
    print_success "Prerequisites satisfied (kubectl, helm)"
}

cleanup() {
    print_section "Uninstalling KubeRay Operator"
    helm uninstall kuberay-operator -n "$KUBERAY_NAMESPACE" 2>/dev/null || true
    kubectl delete namespace "$KUBERAY_NAMESPACE" --ignore-not-found 2>/dev/null || true
    print_success "KubeRay operator uninstalled"
}

if [ "${1:-install}" = "cleanup" ]; then
    check_prerequisites
    cleanup
    exit 0
fi

echo -e "${BLUE}"
echo "=================================================="
echo "  Install KubeRay Operator"
echo "=================================================="
echo -e "${NC}"
echo "  KubeRay version: $KUBERAY_VERSION"
echo "  Namespace:       $KUBERAY_NAMESPACE"
echo

check_prerequisites

if helm status kuberay-operator -n "$KUBERAY_NAMESPACE" &>/dev/null; then
    print_success "KubeRay operator already installed"
    kubectl get pods -n "$KUBERAY_NAMESPACE"
    exit 0
fi

print_section "Adding KubeRay Helm repo"
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

# Pinned to the system node group to keep the GPU nodes free for inference.
helm install kuberay-operator kuberay/kuberay-operator \
    --version "$KUBERAY_VERSION" \
    --namespace "$KUBERAY_NAMESPACE" \
    --create-namespace \
    --set nodeSelector.role=system

print_section "Waiting for operator to be Ready"
kubectl wait --for=condition=Available deployment/kuberay-operator \
    -n "$KUBERAY_NAMESPACE" --timeout=180s

print_success "KubeRay operator installed and ready"
kubectl get pods -n "$KUBERAY_NAMESPACE"
print_elapsed
