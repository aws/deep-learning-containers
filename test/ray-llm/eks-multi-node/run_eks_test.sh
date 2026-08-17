#!/bin/bash
# EKS integration test orchestrator for Ray LLM DLC.
# Applies a KubeRay RayService, waits for readiness, queries the OpenAI-compatible
# endpoint, verifies TP=2 sharding across nodes and EFA usage, and cleans up.
#
# Required env vars:
#   IMAGE_URI       - ECR image URI for the ray-llm DLC
#   RAY_VERSION     - Ray version (e.g., 2.56.1)
#   EKS_CLUSTER     - EKS cluster name (default: dlc-shared-cluster)
#   AWS_REGION      - AWS region (default: us-west-2)
#   NAMESPACE       - K8s namespace (default: ray-llm)
set -euo pipefail

: "${IMAGE_URI:?IMAGE_URI is required}"
: "${RAY_VERSION:?RAY_VERSION is required}"
EKS_CLUSTER="${EKS_CLUSTER:-dlc-shared-cluster}"
AWS_REGION="${AWS_REGION:-us-west-2}"
NAMESPACE="${NAMESPACE:-ray-llm}"
TIMEOUT_READY=600
TIMEOUT_SERVE=1800
MODEL_ID="qwen-14b"
EXPECTED_TP=2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME=$(yq '.metadata.name' "${SCRIPT_DIR}/rayservice.yml")

cleanup() {
    echo "=== Cleanup: killing port-forward ==="
    if [[ -n "${PORT_FORWARD_PID:-}" ]]; then
        kill "${PORT_FORWARD_PID}" 2>/dev/null || true
    fi
    echo "=== Cleanup: deleting RayService ${SERVICE_NAME} ==="
    kubectl delete rayservice "${SERVICE_NAME}" -n "${NAMESPACE}" --ignore-not-found=true --timeout=180s || true
    kubectl wait --for=delete pod -l "ray.io/cluster" -n "${NAMESPACE}" --timeout=240s 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Configuring kubectl for ${EKS_CLUSTER} ==="
aws eks update-kubeconfig --name "${EKS_CLUSTER}" --region "${AWS_REGION}"

echo "=== Applying RayService manifest ==="
envsubst '${IMAGE_URI} ${RAY_VERSION}' < "${SCRIPT_DIR}/rayservice.yml" | kubectl apply -f -

CLUSTER_NAME=""
for _ in $(seq 1 60); do
    CLUSTER_NAME=$(kubectl get raycluster -n "${NAMESPACE}" \
        -l "ray.io/originated-from-cr-name=${SERVICE_NAME}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    [[ -n "${CLUSTER_NAME}" ]] && break
    sleep 5
done
if [[ -z "${CLUSTER_NAME}" ]]; then
    echo "FAIL: RayCluster for RayService ${SERVICE_NAME} did not appear within 5 min"
    exit 1
fi
echo "RayCluster: ${CLUSTER_NAME}"

echo "=== Waiting for head pod Ready (timeout ${TIMEOUT_READY}s) ==="
kubectl wait --for=condition=Ready pod \
    -l "ray.io/cluster=${CLUSTER_NAME},ray.io/node-type=head" \
    -n "${NAMESPACE}" --timeout="${TIMEOUT_READY}s"

echo "=== Waiting for worker pods to reach Running (timeout ${TIMEOUT_READY}s) ==="
kubectl wait --for=jsonpath='{.status.phase}=Running' pod \
    -l "ray.io/cluster=${CLUSTER_NAME},ray.io/node-type=worker" \
    -n "${NAMESPACE}" --timeout="${TIMEOUT_READY}s"

echo "=== Cluster status ==="
kubectl get pods -l "ray.io/cluster=${CLUSTER_NAME}" -n "${NAMESPACE}" -o wide

echo "=== Waiting for Serve app RUNNING (timeout ${TIMEOUT_SERVE}s) ==="
SECONDS_WAITED=0
while [[ "${SECONDS_WAITED}" -lt "${TIMEOUT_SERVE}" ]]; do
    STATUS=$(kubectl get rayservice "${SERVICE_NAME}" -n "${NAMESPACE}" -o jsonpath='{.status.serviceStatus}' 2>/dev/null || echo "")
    APP=$(kubectl get rayservice "${SERVICE_NAME}" -n "${NAMESPACE}" -o jsonpath="{.status.activeServiceStatus.applicationStatuses.qwen.status}" 2>/dev/null || echo "")
    if [[ "${STATUS}" == "Running" && "${APP}" == "RUNNING" ]]; then
        echo "Serve running after ${SECONDS_WAITED}s"
        break
    fi
    sleep 15
    SECONDS_WAITED=$((SECONDS_WAITED + 15))
done
if [[ "${STATUS}" != "Running" || "${APP}" != "RUNNING" ]]; then
    echo "FAIL: RayService did not reach Running/RUNNING within ${TIMEOUT_SERVE}s (rayservice=${STATUS} app=${APP})"
    kubectl describe rayservice "${SERVICE_NAME}" -n "${NAMESPACE}"
    exit 1
fi

HEAD_POD=$(kubectl get pod -l "ray.io/cluster=${CLUSTER_NAME},ray.io/node-type=head" \
    -n "${NAMESPACE}" -o jsonpath='{.items[0].metadata.name}')
WORKER_PODS=($(kubectl get pods -l "ray.io/cluster=${CLUSTER_NAME},ray.io/node-type=worker" \
    -n "${NAMESPACE}" -o jsonpath='{.items[*].metadata.name}'))
echo "Head: ${HEAD_POD}"
echo "Workers: ${WORKER_PODS[*]}"

echo "=== Port-forwarding head :8000 (Serve) + :8265 (dashboard) ==="
kubectl port-forward -n "${NAMESPACE}" "pod/${HEAD_POD}" 8000:8000 8265:8265 >/tmp/pf.log 2>&1 &
PORT_FORWARD_PID=$!
sleep 5

echo "=== Sharding check A: Serve config API reports TP=${EXPECTED_TP} + STRICT_SPREAD ==="
TP=$(curl -sf --max-time 10 http://127.0.0.1:8265/api/serve/applications/ \
    | jq -r '.applications.qwen.deployed_app_config.args.llm_configs[0].engine_kwargs.tensor_parallel_size')
STRATEGY=$(curl -sf --max-time 10 http://127.0.0.1:8265/api/serve/applications/ \
    | jq -r '.applications.qwen.deployed_app_config.args.llm_configs[0].placement_group_config.strategy')
echo "tensor_parallel_size=${TP} strategy=${STRATEGY}"
[[ "${TP}" == "${EXPECTED_TP}" && "${STRATEGY}" == "STRICT_SPREAD" ]] || {
    echo "FAIL: Serve config API did not report TP=${EXPECTED_TP} + STRICT_SPREAD"; exit 1;
}

echo "=== Sharding check B: workers on ${EXPECTED_TP} distinct nodes ==="
DISTINCT_NODES=$(kubectl get pods -l "ray.io/cluster=${CLUSTER_NAME},ray.io/node-type=worker" \
    -n "${NAMESPACE}" -o jsonpath='{.items[*].spec.nodeName}' | tr ' ' '\n' | sort -u | wc -l | tr -d ' ')
echo "distinct nodes: ${DISTINCT_NODES}"
[[ "${DISTINCT_NODES}" == "${EXPECTED_TP}" ]] || {
    echo "FAIL: expected ${EXPECTED_TP} distinct worker nodes, got ${DISTINCT_NODES}"; exit 1;
}

echo "=== Sharding check C: each worker joined a size-${EXPECTED_TP} TP group with a distinct rank ==="
RANKS=""
for w in "${WORKER_PODS[@]}"; do
    RANK=$(kubectl exec -n "${NAMESPACE}" "${w}" -c ray-worker -- bash -c "
        grep -rhE 'world_size=${EXPECTED_TP} rank=[0-9]+' /tmp/ray/session_*/logs 2>/dev/null \
            | grep -oE 'rank=[0-9]+' | head -1
    " 2>/dev/null || echo "")
    echo "  ${w}: ${RANK}"
    [[ -n "${RANK}" ]] || { echo "FAIL: ${w} did not join a size-${EXPECTED_TP} distributed group"; exit 1; }
    RANKS="${RANKS}${RANK}\n"
done
DISTINCT_RANKS=$(printf "${RANKS}" | sort -u | grep -c 'rank=')
[[ "${DISTINCT_RANKS}" == "${EXPECTED_TP}" ]] || {
    echo "FAIL: expected ${EXPECTED_TP} distinct TP ranks across workers, got ${DISTINCT_RANKS}"; exit 1;
}

echo "=== EFA check: 3 canonical strings on every worker + no socket fallback ==="
for w in "${WORKER_PODS[@]}"; do
    echo "--- ${w} ---"
    kubectl exec -n "${NAMESPACE}" "${w}" -c ray-worker -- bash -c '
        LOGS=/tmp/ray/session_*/logs
        grep -qrE "NET/OFI Selected provider is efa" $LOGS 2>/dev/null || { echo "MISS: NET/OFI Selected provider is efa"; exit 1; }
        grep -qrE "Using network (AWS )?Libfabric"   $LOGS 2>/dev/null || { echo "MISS: Using network Libfabric"; exit 1; }
        grep -qr  "aws-ofi-nccl"                     $LOGS 2>/dev/null || { echo "MISS: aws-ofi-nccl"; exit 1; }
        if grep -qrE "NET/Socket: Using" $LOGS 2>/dev/null; then
            echo "FAIL: socket fallback detected"; exit 1
        fi
        echo "EFA strings OK, no socket fallback"
    '
done

echo "=== GET /v1/models ==="
MODELS=$(curl -sf --max-time 10 http://127.0.0.1:8000/v1/models)
echo "${MODELS}"
echo "${MODELS}" | grep -q "${MODEL_ID}" || { echo "FAIL: ${MODEL_ID} not in /v1/models"; exit 1; }

validate_response() {
    local resp="$1"
    local label="$2"
    echo "${resp}" | jq -e 'has("id") and has("object") and has("choices") and has("usage")' >/dev/null \
        || { echo "FAIL: ${label} response missing schema fields (id/object/choices/usage)"; exit 1; }
    local tokens
    tokens=$(echo "${resp}" | jq -r '.usage.completion_tokens // 0')
    [[ "${tokens}" -gt 0 ]] || { echo "FAIL: ${label} completion_tokens=${tokens}"; exit 1; }
    local fp
    fp=$(echo "${resp}" | jq -r '.system_fingerprint // empty')
    [[ "${fp}" == *"tp${EXPECTED_TP}"* ]] || { echo "FAIL: ${label} system_fingerprint ${fp} missing tp${EXPECTED_TP}"; exit 1; }
}

echo "=== POST /v1/completions ==="
RESPONSE=$(curl -sf --max-time 60 -X POST http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL_ID}\",\"prompt\":\"Hello, how are you?\",\"max_tokens\":100,\"temperature\":0.7}")
echo "${RESPONSE}"
validate_response "${RESPONSE}" "/v1/completions"

echo "=== POST /v1/chat/completions ==="
RESPONSE=$(curl -sf --max-time 60 -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"What are the benefits of using FSx Lustre with EKS?\"}],\"max_tokens\":100,\"temperature\":0.7}")
echo "${RESPONSE}"
validate_response "${RESPONSE}" "/v1/chat/completions"

echo "PASS: EKS integration test completed successfully"
