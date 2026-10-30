#!/bin/bash
# deploy_ray_train_job.sh - The multi-node smoke test for this sample. Deploys
# the RayCluster, waits for the head + GPU worker pods, copies ../code onto the
# head pod, and submits an FSDP fine-tuning job with `ray job submit` across
# both nodes. The pass/fail bar is deliberately infrastructure-level, not
# training quality: EFA (not TCP) carries the job's NCCL collectives, EFA's
# network RDMA capability is present, and the job starts and completes.
# Exits non-zero if the job fails or EFA did not carry it, so this doubles as
# a CI check. To delete, use delete_ray_train_job.sh.
#
# Usage: bash deploy_ray_train_job.sh [status]
# Prerequisites: EKS cluster, GPU node group, KubeRay operator, and the GPU
# device plugins (install_gpu_plugins.sh) all installed.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
source "$SCRIPT_DIR/_lib.sh"

MANIFEST="$(dirname "$SCRIPT_DIR")/manifest/raycluster.yaml"
CODE_DIR="$(dirname "$SCRIPT_DIR")/code"
REMOTE_CODE_DIR="/tmp/ray-train-code"

SECONDS=0
TIMEOUT_READY=900

check_prerequisites() {
    check_kubectl_prerequisites rayclusters.ray.io "Install KubeRay first: bash install_kuberay.sh"
    command -v envsubst &>/dev/null || { print_error "envsubst not found (part of gettext)"; exit 1; }
    print_success "Prerequisites satisfied"
}

status() {
    print_section "RayCluster Status"
    kubectl get raycluster "$RAY_CLUSTER_NAME" -n "$NAMESPACE" 2>/dev/null || echo "  Not found"

    echo
    echo "Pods (head + workers, with node placement):"
    kubectl get pods -l "ray.io/cluster=${RAY_CLUSTER_NAME}" -n "$NAMESPACE" -o wide 2>/dev/null || echo "  No Ray pods yet"

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
echo "  Deploy Multi-Node Ray Train Job"
echo "=================================================="
echo -e "${NC}"
echo "  Cluster:      $CLUSTER_NAME"
echo "  Namespace:    $NAMESPACE"
echo "  RayCluster:   $RAY_CLUSTER_NAME"
echo "  DLC Image:    $DLC_IMAGE"
echo "  Job:          FSDP fine-tune of $MODEL_ID, ${STEPS} step(s)"
echo "  Workers:      $GPU_NODE_COUNT x $GPU_NODE_TYPE (${GPUS_PER_NODE} GPU + 1 EFA interface each)"
echo

confirm

check_prerequisites

print_section "Step 1: Ensuring Namespace Exists"
retry 3 8 bash -c "kubectl create namespace '$NAMESPACE' --dry-run=client -o yaml | kubectl apply -f -" >/dev/null \
    || { print_error "Could not create/verify namespace '$NAMESPACE' after retries."; exit 1; }
print_success "Namespace '$NAMESPACE' ready"

print_section "Step 2: Applying RayCluster Manifest"
if [ ! -f "$MANIFEST" ]; then
    print_error "Manifest not found: $MANIFEST"
    exit 1
fi

export DLC_IMAGE RAY_VERSION NAMESPACE RAY_CLUSTER_NAME GPUS_PER_NODE
envsubst '${DLC_IMAGE} ${RAY_VERSION} ${NAMESPACE} ${RAY_CLUSTER_NAME} ${GPUS_PER_NODE}' \
    < "$MANIFEST" | kubectl apply -f -
print_success "RayCluster manifest applied"

print_section "Step 3: Confirming GPU + EFA Are Allocatable on Both Nodes"
echo "Checked here, before waiting on pod scheduling, so a missing device plugin fails fast"
echo "instead of surfacing as an opaque Pending timeout."
kubectl get nodes -l role=gpu-worker \
    -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu,EFA:.status.allocatable.vpc\.amazonaws\.com/efa'
GPU_READY_NODES=$(kubectl get nodes -l role=gpu-worker -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' 2>/dev/null | grep -c "^${GPUS_PER_NODE}$" || true)
EFA_READY_NODES=$(kubectl get nodes -l role=gpu-worker -o jsonpath='{range .items[*]}{.status.allocatable.vpc\.amazonaws\.com/efa}{"\n"}{end}' 2>/dev/null | grep -c "^1$" || true)
if [ "$GPU_READY_NODES" -lt "$GPU_NODE_COUNT" ] || [ "$EFA_READY_NODES" -lt "$GPU_NODE_COUNT" ]; then
    print_error "Not all $GPU_NODE_COUNT GPU node(s) advertise nvidia.com/gpu=${GPUS_PER_NODE} and vpc.amazonaws.com/efa=1 (gpu_ready=$GPU_READY_NODES efa_ready=$EFA_READY_NODES). Run install_gpu_plugins.sh first."
    exit 1
fi
print_success "GPU and EFA allocatable on all $GPU_NODE_COUNT node(s)"

print_section "Step 4: Waiting for Head + Worker Pods (timeout ${TIMEOUT_READY}s)"
echo "Head pod becoming Ready..."
kubectl wait --for=condition=Ready pod \
    -l "ray.io/cluster=${RAY_CLUSTER_NAME},ray.io/node-type=head" \
    -n "$NAMESPACE" --timeout="${TIMEOUT_READY}s"

echo "Worker pods becoming Ready (one per GPU node)..."
kubectl wait --for=condition=Ready pod \
    -l "ray.io/cluster=${RAY_CLUSTER_NAME},ray.io/node-type=worker" \
    -n "$NAMESPACE" --timeout="${TIMEOUT_READY}s"

kubectl get pods -l "ray.io/cluster=${RAY_CLUSTER_NAME}" -n "$NAMESPACE" -o wide

HEAD_POD=$(kubectl get pod -n "$NAMESPACE" \
    -l "ray.io/cluster=${RAY_CLUSTER_NAME},ray.io/node-type=head" \
    -o jsonpath='{.items[0].metadata.name}')
if [ -z "$HEAD_POD" ]; then
    print_error "Could not find the head pod for RayCluster '$RAY_CLUSTER_NAME'."
    exit 1
fi
print_success "Head pod: $HEAD_POD"

print_section "Step 5: Copying Training Code to Head Pod"
kubectl exec "$HEAD_POD" -n "$NAMESPACE" -c ray-head -- rm -rf "$REMOTE_CODE_DIR"
kubectl cp "$CODE_DIR" "${NAMESPACE}/${HEAD_POD}:${REMOTE_CODE_DIR}" -c ray-head
print_success "Code copied to ${HEAD_POD}:${REMOTE_CODE_DIR}"

print_section "Step 6: Submitting Training Job"
TOTAL_GPUS=$((GPU_NODE_COUNT * GPUS_PER_NODE))
echo "FSDP shards $MODEL_ID's parameters, gradients, and optimizer state across all $TOTAL_GPUS ranks;"
echo "any all-gather/reduce-scatter between a rank on one node and a rank on the other crosses over EFA."
set +e
kubectl exec "$HEAD_POD" -n "$NAMESPACE" -c ray-head -- \
    ray job submit --address http://localhost:8265 --working-dir "$REMOTE_CODE_DIR" -- \
    python3 train.py \
        --model_id "$MODEL_ID" \
        --steps "$STEPS" \
        --seq_len "$SEQ_LEN" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --num_workers "$([ "$NUM_WORKERS" -gt 0 ] && echo "$NUM_WORKERS" || echo "$TOTAL_GPUS")"
JOB_EXIT_CODE=$?
set -e

if [ "$JOB_EXIT_CODE" -ne 0 ]; then
    print_error "Ray Train job failed (exit code $JOB_EXIT_CODE). Recent worker logs:"
    kubectl logs -n "$NAMESPACE" -l "ray.io/cluster=${RAY_CLUSTER_NAME},ray.io/node-type=worker" -c ray-worker --tail=60 2>/dev/null || true
    exit "$JOB_EXIT_CODE"
fi
print_success "Ray Train job completed: $TOTAL_GPUS ranks across $GPU_NODE_COUNT nodes"

print_section "Step 7: Confirming EFA (Not TCP) Carried the Job"
# A job over TCP looks identical to a healthy one in 'kubectl get pods', so
# this has to be checked in NCCL's logs. NCCL runs inside the Ray Train worker
# actors, which write to Ray's per-worker session logs (/tmp/ray/session_*/
# logs/) -- not to the pod's container stdout -- so grep there, on every
# worker pod. Both the positive signal and the absence of the socket fallback
# matter: a job can complete and print success while quietly using sockets.
WORKER_LOGS=""
for wpod in $(kubectl get pods -n "$NAMESPACE" -l "ray.io/cluster=${RAY_CLUSTER_NAME},ray.io/node-type=worker" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null); do
    WORKER_LOGS+=$(kubectl exec "$wpod" -n "$NAMESPACE" -c ray-worker -- \
        bash -c 'grep -rhiE "NET/OFI Selected provider|Using network Socket|Selected provider is sockets" /tmp/ray/session_*/logs/ 2>/dev/null' 2>/dev/null || true)
    WORKER_LOGS+=$'\n'
done
EFA_LOG=$(echo "$WORKER_LOGS" | grep -iE "NET/OFI Selected provider is efa" || true)
SOCKET_FALLBACK=$(echo "$WORKER_LOGS" | grep -iE "Using network Socket|Selected provider is sockets" || true)

if [ -n "$SOCKET_FALLBACK" ]; then
    print_error "NCCL fell back to sockets/TCP -- EFA is not carrying this job's collectives:"
    echo "$SOCKET_FALLBACK" | head -3
    EFA_CONFIRMED=false
elif [ -n "$EFA_LOG" ]; then
    print_success "EFA confirmed: $(echo "$EFA_LOG" | head -1)"
    EFA_CONFIRMED=true
else
    print_warning "Found neither the EFA line nor a socket fallback in worker logs -- inconclusive, re-check manually."
    EFA_CONFIRMED=false
fi

print_section "Step 8: Confirming EFA's RDMA Capability"
# g6.12xlarge is EFA-capable with network-level RDMA (libfabric RMA read/
# write between hosts), but it does NOT have GPUDirect RDMA -- that needs
# p4d/p4de/p5/p5e/p5en/p6/trn1. This checks the former; it is not evidence
# of the latter.
WORKER_POD=$(kubectl get pod -n "$NAMESPACE" -l "ray.io/cluster=${RAY_CLUSTER_NAME},ray.io/node-type=worker" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
if [ -n "$WORKER_POD" ]; then
    FI_INFO=$(kubectl exec "$WORKER_POD" -n "$NAMESPACE" -c ray-worker -- fi_info -p efa 2>/dev/null || true)
    if echo "$FI_INFO" | grep -qE "FI_RMA|FI_EP_RDM"; then
        print_success "EFA RMA/RDM capability confirmed via fi_info on $WORKER_POD (network-level RDMA, not GPUDirect RDMA)"
        RDMA_CONFIRMED=true
    else
        print_warning "Could not confirm RMA/RDM capability via fi_info on $WORKER_POD. Run manually: kubectl exec $WORKER_POD -n $NAMESPACE -c ray-worker -- fi_info -p efa"
        RDMA_CONFIRMED=false
    fi
else
    print_warning "Could not find a worker pod to run fi_info against."
    RDMA_CONFIRMED=false
fi

print_section "Summary"
echo "  EFA enabled and carrying the job:     ${EFA_CONFIRMED}"
echo "  EFA network RDMA capability:          ${RDMA_CONFIRMED}  (network-level, not GPUDirect RDMA -- g6 does not support GPUDirect)"
echo "  Job started and completed:            true"
print_elapsed

if [ "$EFA_CONFIRMED" != "true" ] || [ "$RDMA_CONFIRMED" != "true" ]; then
    print_error "One or more infrastructure criteria did not pass -- see Step 7/8 above."
    exit 1
fi
print_success "All criteria met: EFA enabled, network RDMA confirmed, job started and completed."
