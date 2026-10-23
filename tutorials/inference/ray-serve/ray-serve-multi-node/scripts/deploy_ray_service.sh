#!/bin/bash
# deploy_ray_service.sh - Deploy the multi-node RayService and wait for the
# OpenAI-compatible Serve app to become RUNNING. To delete, use
# delete_ray_service.sh.
#
# Usage: bash deploy_ray_service.sh [status]
# Prerequisites: EKS cluster, GPU node group, and KubeRay operator installed.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
source "$SCRIPT_DIR/_lib.sh"

MANIFEST="$(dirname "$SCRIPT_DIR")/manifest/rayservice.yaml"

SECONDS=0
TIMEOUT_READY=900
TIMEOUT_SERVE=1800

check_prerequisites() {
    check_kubectl_prerequisites rayservices.ray.io "Install KubeRay first: bash install_kuberay.sh"
    print_success "Prerequisites satisfied"
}

raycluster_name() {
    kubectl get raycluster -n "$NAMESPACE" \
        -l "ray.io/originated-from-cr-name=${RAY_SERVICE_NAME}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo ""
}

status() {
    print_section "RayService Status"
    kubectl get rayservice "$RAY_SERVICE_NAME" -n "$NAMESPACE" 2>/dev/null || echo "  Not found"

    echo
    echo "Pods (head + workers, with node placement):"
    local ray_cluster
    ray_cluster=$(raycluster_name)
    if [ -n "$ray_cluster" ]; then
        kubectl get pods -l "ray.io/cluster=${ray_cluster}" -n "$NAMESPACE" -o wide 2>/dev/null
    else
        echo "  No Ray pods yet"
    fi

    echo
    echo "GPU Nodes:"
    kubectl get nodes -l role=gpu-worker \
        -o custom-columns='NAME:.metadata.name,GPU:.status.capacity.nvidia\.com/gpu' 2>/dev/null || echo "  No GPU nodes"
}

if [ "${1:-deploy}" = "status" ]; then
    check_prerequisites
    status
    exit 0
fi

echo -e "${BLUE}"
echo "=================================================="
echo "  Deploy Multi-Node RayService"
echo "=================================================="
echo -e "${NC}"
echo "  Cluster:      $CLUSTER_NAME"
echo "  Namespace:    $NAMESPACE"
echo "  RayService:   $RAY_SERVICE_NAME"
echo "  DLC Image:    $DLC_IMAGE"
echo "  Model:        $MODEL_SOURCE (id: $MODEL_ID)"
echo "  Serving mode: prefill/decode disaggregation (across $GPU_NODE_COUNT x $GPU_NODE_TYPE)"
echo

read -p "Proceed? (y/N): " -n 1 -r
echo
[[ $REPLY =~ ^[Yy]$ ]] || { echo "Cancelled."; exit 0; }

check_prerequisites

print_section "Step 1: Ensuring Namespace Exists"
retry 3 8 bash -c "kubectl create namespace '$NAMESPACE' --dry-run=client -o yaml | kubectl apply -f -" >/dev/null \
    || { print_error "Could not create/verify namespace '$NAMESPACE' after retries."; exit 1; }
print_success "Namespace '$NAMESPACE' ready"

print_section "Step 2: Applying RayService Manifest"
if [ ! -f "$MANIFEST" ]; then
    print_error "Manifest not found: $MANIFEST"
    exit 1
fi

export DLC_IMAGE RAY_VERSION NAMESPACE RAY_SERVICE_NAME MODEL_ID MODEL_SOURCE
envsubst '${DLC_IMAGE} ${RAY_VERSION} ${NAMESPACE} ${RAY_SERVICE_NAME} ${MODEL_ID} ${MODEL_SOURCE}' \
    < "$MANIFEST" | kubectl apply -f -
print_success "RayService manifest applied"

print_section "Step 3: Waiting for RayCluster"
RAY_CLUSTER=""
for _ in $(seq 1 60); do
    RAY_CLUSTER=$(raycluster_name)
    [ -n "$RAY_CLUSTER" ] && break
    sleep 5
done
if [ -z "$RAY_CLUSTER" ]; then
    print_error "RayCluster for RayService '$RAY_SERVICE_NAME' did not appear within 5 min"
    exit 1
fi
print_success "RayCluster: $RAY_CLUSTER"

print_section "Step 4: Waiting for Head + Worker Pods (timeout ${TIMEOUT_READY}s)"
echo "Head pod becoming Ready..."
kubectl wait --for=condition=Ready pod \
    -l "ray.io/cluster=${RAY_CLUSTER},ray.io/node-type=head" \
    -n "$NAMESPACE" --timeout="${TIMEOUT_READY}s"

echo "Worker pods reaching Running (one per GPU node)..."
kubectl wait --for=jsonpath='{.status.phase}=Running' pod \
    -l "ray.io/cluster=${RAY_CLUSTER},ray.io/node-type=worker" \
    -n "$NAMESPACE" --timeout="${TIMEOUT_READY}s"

kubectl get pods -l "ray.io/cluster=${RAY_CLUSTER}" -n "$NAMESPACE" -o wide

print_section "Step 5: Waiting for Serve App RUNNING (timeout ${TIMEOUT_SERVE}s)"
echo "Each phase loads the model on its node (prefill on one, decode on the other). This can take several minutes."
SECONDS_WAITED=0
while [ "$SECONDS_WAITED" -lt "$TIMEOUT_SERVE" ]; do
    SVC=$(kubectl get rayservice "$RAY_SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.status.serviceStatus}' 2>/dev/null || echo "")
    APP=$(kubectl get rayservice "$RAY_SERVICE_NAME" -n "$NAMESPACE" -o jsonpath="{.status.activeServiceStatus.applicationStatuses.qwen.status}" 2>/dev/null || echo "")
    if [ "$SVC" = "Running" ] && [ "$APP" = "RUNNING" ]; then
        print_success "Serve app RUNNING after ${SECONDS_WAITED}s"
        break
    fi
    echo "  ...rayservice=${SVC:-?} app=${APP:-?} (${SECONDS_WAITED}s)"
    sleep 15
    SECONDS_WAITED=$((SECONDS_WAITED + 15))
done
if [ "$SVC" != "Running" ] || [ "$APP" != "RUNNING" ]; then
    print_error "RayService did not reach RUNNING within ${TIMEOUT_SERVE}s (rayservice=$SVC app=$APP)"
    kubectl describe rayservice "$RAY_SERVICE_NAME" -n "$NAMESPACE" | tail -40
    exit 1
fi

print_section "Deployment Complete"
echo "The model is served across ${GPU_NODE_COUNT} nodes (prefill + decode disaggregation)."
echo "Invoke it by port-forwarding the serve port (see the README)."
print_elapsed
