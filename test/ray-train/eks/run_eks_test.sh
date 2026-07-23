#!/bin/bash
# EKS integration test orchestrator for Ray Train DLC.
# Runs on the CI runner (not inside the container). Applies a KubeRay RayCluster,
# waits for readiness, submits a training job, asserts success, and cleans up.
#
# Required env vars:
#   IMAGE_URI       - ECR image URI for the ray-train DLC
#   RAY_VERSION     - Ray version (e.g., 2.56.0) for the RayCluster spec
#   EKS_CLUSTER     - EKS cluster name (default: dlc-shared-cluster)
#   AWS_REGION      - AWS region (default: us-west-2)
#   NAMESPACE       - K8s namespace (default: ray-train)
set -euo pipefail

: "${IMAGE_URI:?IMAGE_URI is required}"
: "${RAY_VERSION:?RAY_VERSION is required}"
EKS_CLUSTER="${EKS_CLUSTER:-dlc-shared-cluster}"
AWS_REGION="${AWS_REGION:-us-west-2}"
NAMESPACE="${NAMESPACE:-ray-train}"
CLUSTER_NAME="ray-train-test"
TIMEOUT_READY=600
TIMEOUT_JOB=1800

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cleanup() {
    echo "=== Cleanup: deleting RayCluster ${CLUSTER_NAME} ==="
    kubectl delete raycluster "${CLUSTER_NAME}" -n "${NAMESPACE}" --ignore-not-found=true --timeout=120s || true
    echo "=== Cleanup: waiting for pods to terminate ==="
    kubectl wait --for=delete pod -l "ray.io/cluster=${CLUSTER_NAME}" -n "${NAMESPACE}" --timeout=180s 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Configuring kubectl for ${EKS_CLUSTER} ==="
aws eks update-kubeconfig --name "${EKS_CLUSTER}" --region "${AWS_REGION}"

echo "=== Applying RayCluster manifest ==="
envsubst '${IMAGE_URI} ${RAY_VERSION}' < "${SCRIPT_DIR}/raycluster.yml" | kubectl apply -f -

echo "=== Waiting for head pod Ready (timeout ${TIMEOUT_READY}s) ==="
kubectl wait --for=condition=Ready pod \
    -l "ray.io/cluster=${CLUSTER_NAME},ray.io/node-type=head" \
    -n "${NAMESPACE}" --timeout="${TIMEOUT_READY}s"

echo "=== Waiting for worker pods Ready (timeout ${TIMEOUT_READY}s) ==="
kubectl wait --for=condition=Ready pod \
    -l "ray.io/cluster=${CLUSTER_NAME},ray.io/node-type=worker" \
    -n "${NAMESPACE}" --timeout="${TIMEOUT_READY}s"

echo "=== Cluster status ==="
kubectl get pods -l "ray.io/cluster=${CLUSTER_NAME}" -n "${NAMESPACE}" -o wide

HEAD_POD=$(kubectl get pod -l "ray.io/cluster=${CLUSTER_NAME},ray.io/node-type=head" \
    -n "${NAMESPACE}" -o jsonpath='{.items[0].metadata.name}')
echo "Head pod: ${HEAD_POD}"

echo "=== Copying training script to head pod ==="
kubectl exec "${HEAD_POD}" -n "${NAMESPACE}" -- mkdir -p /workspace
kubectl cp "${SCRIPT_DIR}/scripts/fsdp_ray.py" "${NAMESPACE}/${HEAD_POD}:/workspace/fsdp_ray.py"

echo "=== Submitting Ray job (timeout ${TIMEOUT_JOB}s) ==="
JOB_OUTPUT=$(kubectl exec "${HEAD_POD}" -n "${NAMESPACE}" -- \
    timeout "${TIMEOUT_JOB}" ray job submit \
        --address http://localhost:8265 \
        --working-dir /workspace \
        -- python3 fsdp_ray.py 2>&1) || {
    echo "=== Job failed ==="
    echo "${JOB_OUTPUT}"
    exit 1
}

echo "${JOB_OUTPUT}"

echo "=== Validating results ==="
FAILED=false

if ! echo "${JOB_OUTPUT}" | grep -q "EKS_TEST_RESULT: SUCCESS"; then
    echo "FAIL: SUCCESS marker not found"
    FAILED=true
fi

if echo "${JOB_OUTPUT}" | grep -q "nranks 8"; then
    echo "PASS: multi-node confirmed (nranks=8)"
else
    echo "FAIL: nranks 8 not found in NCCL logs"
    FAILED=true
fi

if echo "${JOB_OUTPUT}" | grep -q "NET/OFI.*Libfabric\|Selected provider is efa"; then
    echo "PASS: EFA/OFI provider engaged"
else
    echo "FAIL: EFA/Libfabric not found in NCCL logs"
    FAILED=true
fi

if [ "${FAILED}" = "true" ]; then
    exit 1
fi
echo "PASS: EKS integration test completed successfully"
