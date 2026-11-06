#!/bin/bash
# deploy_node_group.sh — Create the GPU node group for Ray workers.
# Idempotent: safe to re-run if interrupted. To delete, use delete_node_group.sh.
#
# Usage: bash deploy_node_group.sh
# Override: GPU_NODE_TYPE=g6.12xlarge GPU_AZ=sa-east-1b bash deploy_node_group.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
source "$SCRIPT_DIR/_lib.sh"

SECONDS=0

check_prerequisites() {
    local missing=()
    command -v aws &>/dev/null || missing+=("aws")
    command -v eksctl &>/dev/null || missing+=("eksctl")
    command -v kubectl &>/dev/null || missing+=("kubectl")
    if [ ${#missing[@]} -gt 0 ]; then
        print_error "Missing required tools: ${missing[*]}"
        exit 1
    fi
    check_credentials

    local cluster_status
    cluster_status=$(get_cluster_status)
    if [ "$cluster_status" != "ACTIVE" ]; then
        print_error "Cluster '$CLUSTER_NAME' is not ACTIVE in $REGION (status: $cluster_status). Create it first with deploy_cluster.sh."
        exit 1
    fi
    print_success "Prerequisites satisfied"
}

echo -e "${BLUE}"
echo "=================================================="
echo "  Create GPU Node Group"
echo "=================================================="
echo -e "${NC}"
echo "  Cluster:    $CLUSTER_NAME"
echo "  Region:     $REGION"
echo "  Node Group: $GPU_NODEGROUP_NAME"
echo "  Instance:   $GPU_NODE_TYPE"
echo "  Count:      $GPU_NODE_COUNT"
echo "  EFA:        enabled"
echo "  AZ:         ${GPU_AZ:-auto-discover}"
echo

read -p "Proceed? (y/N): " -n 1 -r
echo
[[ $REPLY =~ ^[Yy]$ ]] || { echo "Cancelled."; exit 0; }

check_prerequisites

GPU_AZ=$(resolve_gpu_az)
print_success "GPU AZ: $GPU_AZ"

print_section "Checking for Existing GPU Node Group"
NODEGROUP_STATUS=$(get_nodegroup_status "$GPU_NODEGROUP_NAME")

case "$NODEGROUP_STATUS" in
    NOT_FOUND)
        print_section "Creating GPU Node Group"
        echo "Creating ${GPU_NODE_COUNT}x ${GPU_NODE_TYPE} node(s) in ${GPU_AZ}..."
        wait_for_no_active_update

        # efaEnabled has no eksctl CLI flag, so this needs a config file.
        NODEGROUP_CONFIG=$(mktemp)
        cat > "$NODEGROUP_CONFIG" << EOF
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: ${CLUSTER_NAME}
  region: ${REGION}

managedNodeGroups:
  - name: ${GPU_NODEGROUP_NAME}
    instanceType: ${GPU_NODE_TYPE}
    desiredCapacity: ${GPU_NODE_COUNT}
    minSize: ${GPU_NODE_COUNT}
    maxSize: ${GPU_NODE_COUNT}
    availabilityZones: ["${GPU_AZ}"]
    privateNetworking: true
    efaEnabled: true
    # Model weights land on the node's EBS via the container overlay fs.
    volumeSize: 200
    labels:
      role: gpu-worker
EOF
        eksctl create nodegroup -f "$NODEGROUP_CONFIG"
        rm -f "$NODEGROUP_CONFIG"
        print_success "GPU node group created"

        print_section "Verifying GPU Nodes + EFA"
        # The NVIDIA and EFA device plugins register shortly after the nodes
        # join, so poll until every node advertises both before printing.
        for _ in $(seq 1 30); do
            advertised=$(kubectl get nodes -l role=gpu-worker \
                -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}/{.status.allocatable.vpc\.amazonaws\.com/efa}{"\n"}{end}' \
                2>/dev/null | grep -c '^[0-9][0-9]*/[0-9][0-9]*$' || true)
            [ "${advertised:-0}" -ge "$GPU_NODE_COUNT" ] && break
            sleep 10
        done
        kubectl get nodes -l role=gpu-worker \
            -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu,EFA:.status.allocatable.vpc\.amazonaws\.com/efa' \
            || print_warning "Could not fetch nodes right now (transient?). The node group was created successfully above."
        ;;
    ACTIVE)
        print_success "GPU node group '$GPU_NODEGROUP_NAME' already exists (ACTIVE)"
        kubectl get nodes -l role=gpu-worker
        ;;
    UNKNOWN)
        print_error "Could not determine status of node group '$GPU_NODEGROUP_NAME' (repeated API errors). Check your AWS session and re-run."
        exit 1
        ;;
    CREATING|UPDATING)
        print_warning "GPU node group is already '$NODEGROUP_STATUS'. Not creating a duplicate; re-run later to verify it finished."
        ;;
    *)
        print_warning "GPU node group is in unexpected state '$NODEGROUP_STATUS'. Inspect it in the EKS console before re-running."
        ;;
esac

print_elapsed