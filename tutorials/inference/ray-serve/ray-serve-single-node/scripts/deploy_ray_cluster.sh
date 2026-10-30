#!/bin/bash
# =============================================================================
# deploy_ray_cluster.sh — Deploy Ray Serve as a plain Deployment and wait for
# the pod to become ready.
#
# Usage:
#   bash deploy_ray_cluster.sh            # Deploy
#   bash deploy_ray_cluster.sh cleanup    # Delete Deployment
#   bash deploy_ray_cluster.sh status     # Check status
#
# Prerequisites:
#   - EKS cluster running (deploy_cluster.sh)
#   - GPU node group created (deploy_node_group.sh)
# =============================================================================

set -eo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_DIR="$(dirname "$SCRIPT_DIR")/manifest"

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
    if ! command -v kubectl &>/dev/null; then
        print_error "kubectl not found"
        exit 1
    fi

    if ! kubectl cluster-info &>/dev/null; then
        print_error "Cannot connect to Kubernetes cluster. Check kubeconfig."
        exit 1
    fi
    print_success "Prerequisites satisfied"
}

# ─── Cleanup ─────────────────────────────────────────────────────────────────
cleanup() {
    echo -e "${BLUE}"
    echo "=================================================="
    echo "  Ray Serve Deployment Deletion"
    echo "=================================================="
    echo -e "${NC}"
    echo "  Deployment: $RAY_CLUSTER_NAME"
    echo "  Namespace:  $NAMESPACE"
    echo ""

    read -p "Are you sure you want to delete the deployment? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi

    print_section "Deleting Deployment"
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

    ELAPSED_MIN=$((SECONDS / 60))
    ELAPSED_SEC=$((SECONDS % 60))
    echo -e "\n${BLUE}⏱ Elapsed: ${ELAPSED_MIN}m ${ELAPSED_SEC}s${NC}"
}

# ─── Status ──────────────────────────────────────────────────────────────────
status() {
    print_section "Ray Serve Deployment Status"

    echo "Deployment:"
    kubectl get deployment "$RAY_CLUSTER_NAME" -n "$NAMESPACE" 2>/dev/null || echo "  Not found"

    echo
    echo "Pods:"
    kubectl get pods -n "$NAMESPACE" -l app=ray-serve -o wide 2>/dev/null || echo "  No pods"

    echo
    echo "GPU Nodes:"
    kubectl get nodes -l role=gpu-worker -o custom-columns=\
'NAME:.metadata.name,STATUS:.status.conditions[-1:].type,GPU:.status.capacity.nvidia\.com/gpu,AGE:.metadata.creationTimestamp' 2>/dev/null || echo "  No GPU nodes"
}

# ─── Main: route subcommand ─────────────────────────────────────────────────
COMMAND=${1:-"deploy"}

case "$COMMAND" in
    cleanup)
        check_prerequisites
        cleanup
        exit 0
        ;;
    status)
        check_prerequisites
        status
        exit 0
        ;;
esac

# ─── Deploy ──────────────────────────────────────────────────────────────────
echo -e "${BLUE}"
echo "=================================================="
echo "  Deploy Ray Serve"
echo "=================================================="
echo -e "${NC}"
echo "  Cluster:     $CLUSTER_NAME"
echo "  Namespace:   $NAMESPACE"
echo "  Deployment:  $RAY_CLUSTER_NAME"
echo "  DLC Image:   $DLC_IMAGE"
echo

read -p "Proceed? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

check_prerequisites

# ─── Step 1: Create Namespace ────────────────────────────────────────────────
print_section "Step 1: Ensuring Namespace Exists"
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
print_success "Namespace '$NAMESPACE' ready"

# ─── Step 2: Create ConfigMap from qwen_serve.py ────────────────────────────
print_section "Step 2: Creating qwen-serve-code ConfigMap"
CODE_DIR="$(dirname "$SCRIPT_DIR")/code"
if [ ! -f "${CODE_DIR}/qwen_serve.py" ]; then
    print_error "qwen_serve.py not found at ${CODE_DIR}/qwen_serve.py"
    exit 1
fi
kubectl create configmap qwen-serve-code -n "$NAMESPACE" \
    --from-file=qwen_serve.py="${CODE_DIR}/qwen_serve.py" \
    --dry-run=client -o yaml | kubectl apply -f -
print_success "ConfigMap 'qwen-serve-code' ready"

# ─── Step 3: Deploy Manifest ────────────────────────────────────────────────
print_section "Step 3: Deploying Ray Serve"

if [ ! -f "${MANIFEST_DIR}/ray-cluster.yaml" ]; then
    print_error "Manifest not found: ${MANIFEST_DIR}/ray-cluster.yaml"
    exit 1
fi

sed -e "s|\${NAMESPACE}|${NAMESPACE}|g" \
    -e "s|\${DLC_IMAGE}|${DLC_IMAGE}|g" \
    -e "s|\${RAY_CLUSTER_NAME}|${RAY_CLUSTER_NAME}|g" \
    "${MANIFEST_DIR}/ray-cluster.yaml" \
    | kubectl apply -f -

print_success "Deployment manifest applied"

# ─── Step 4: Wait for Pod Ready ──────────────────────────────────────────────
print_section "Step 4: Waiting for Pod to be Ready"
echo "Waiting for the GPU-scheduled pod to start..."

POD_READY=false
for i in $(seq 1 60); do
    POD=$(kubectl get pods -n "$NAMESPACE" -l app=ray-serve \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    if [[ -n "$POD" ]]; then
        READY=$(kubectl get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null || echo "false")
        POD_STATUS=$(kubectl get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
        echo "  Pod: $POD ($POD_STATUS, ready=$READY)"
        if [[ "$READY" == "true" ]]; then
            POD_READY=true
            print_success "Pod is ready"
            break
        fi
    else
        echo "  Waiting for pod to be created... ($i/60)"
    fi
    sleep 15
done

if [[ "$POD_READY" != "true" ]]; then
    print_error "Pod did not become ready within 15 minutes"
    [[ -n "$POD" ]] && kubectl describe pod "$POD" -n "$NAMESPACE" | tail -30
    [[ -n "$POD" ]] && kubectl logs "$POD" -n "$NAMESPACE" --tail=50 2>/dev/null || true
    exit 1
fi

# ─── Step 5: Verify ─────────────────────────────────────────────────────────
print_section "Step 5: Verifying Deployment"

kubectl get pods -n "$NAMESPACE" -l app=ray-serve -o wide

print_section "Deployment Complete"
echo
echo "  Deployment: $RAY_CLUSTER_NAME"
echo "  Namespace:  $NAMESPACE"
echo "  Pod:        $POD"
echo
echo "Access the endpoint:"
echo "  kubectl port-forward -n $NAMESPACE deploy/$RAY_CLUSTER_NAME 8000:8000"
echo
echo "Other commands:"
echo "  bash deploy_ray_cluster.sh status     # Check status"
echo "  bash delete_ray_cluster.sh            # Tear down"

ELAPSED_MIN=$((SECONDS / 60))
ELAPSED_SEC=$((SECONDS % 60))
echo -e "\n${BLUE}⏱ Elapsed: ${ELAPSED_MIN}m ${ELAPSED_SEC}s${NC}"
