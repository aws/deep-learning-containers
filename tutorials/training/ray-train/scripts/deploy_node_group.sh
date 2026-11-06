#!/bin/bash
# deploy_node_group.sh — Create the GPU node group for Ray Train workers.
# Idempotent: safe to re-run if interrupted. To delete, use delete_node_group.sh.
#
# GPU_AZ is auto-picked from the intersection of "AZs that offer
# $GPU_NODE_TYPE" and "AZs the cluster already has a private subnet in" --
# the node group always lands in the SAME subnet the cluster (and system
# node group) already uses. If you set GPU_AZ explicitly, it is validated
# against that same intersection and rejected with a clear reason otherwise
# (e.g. the type isn't offered there, or the cluster has no subnet there).
#
# Usage: bash deploy_node_group.sh
# Override: GPU_NODE_TYPE=g6.8xlarge GPUS_PER_NODE=1 GPU_AZ=us-east-1b bash deploy_node_group.sh

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

confirm

check_prerequisites

if [ -n "$CAPACITY_RESERVATION_ID" ]; then
    GPU_AZ=$(resolve_capacity_reservation "$CAPACITY_RESERVATION_ID")
    print_success "Capacity reservation $CAPACITY_RESERVATION_ID: ${GPU_NODE_TYPE} in ${GPU_AZ}"
fi

AZ_WAS_SET=${GPU_AZ:+yes}
GPU_AZ=$(resolve_gpu_az)
# Resolved from the cluster's own VPC, not re-derived from the AZ name --
# this is what guarantees the node group lands in the same subnet the
# cluster (and system node group) already uses, never a different one.
GPU_SUBNET=$(resolve_gpu_subnet "$GPU_AZ")
print_success "GPU AZ: $GPU_AZ (cluster subnet $GPU_SUBNET)"
if [ -z "$AZ_WAS_SET" ]; then
    # resolve_gpu_az only knows where the type is OFFERED and where the
    # cluster has a subnet, not where there is quota or capacity right now;
    # find_gpu_capacity.sh checks both of those too.
    print_warning "GPU_AZ was auto-picked (and cross-checked against the cluster's own subnets). If launches fail with InsufficientInstanceCapacity, run 'bash find_gpu_capacity.sh' and re-run with GPU_AZ=<az>."
fi

print_section "Checking for Existing GPU Node Group"
NODEGROUP_STATUS=$(get_nodegroup_status "$GPU_NODEGROUP_NAME")

case "$NODEGROUP_STATUS" in
    NOT_FOUND)
        # A previous attempt that failed (e.g. no capacity in its AZ) leaves
        # eksctl's node group stack in ROLLBACK_COMPLETE. It holds no resources,
        # but eksctl refuses to create a stack with the same name, so remove it.
        NG_STACK="eksctl-${CLUSTER_NAME}-nodegroup-${GPU_NODEGROUP_NAME}"
        NG_STACK_STATUS=$(get_cf_stack_status "$NG_STACK")
        if [ "$NG_STACK_STATUS" = "ROLLBACK_COMPLETE" ]; then
            print_warning "Removing failed stack $NG_STACK (ROLLBACK_COMPLETE, no resources) left by a previous attempt..."
            disable_eksctl_stack_protection "$NG_STACK"
            aws cloudformation delete-stack --stack-name "$NG_STACK" --region "$REGION"
            aws cloudformation wait stack-delete-complete --stack-name "$NG_STACK" --region "$REGION"
            print_success "Stale stack removed"
        fi

        print_section "Creating GPU Node Group"
        echo "Creating ${GPU_NODE_COUNT}x ${GPU_NODE_TYPE} node(s) in ${GPU_AZ} (subnet ${GPU_SUBNET})..."
        wait_for_no_active_update

        CR_BLOCK=""
        [ -n "$CAPACITY_RESERVATION_ID" ] && CR_BLOCK="    capacityReservation:
      capacityReservationTarget:
        capacityReservationID: ${CAPACITY_RESERVATION_ID}"

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
    # An explicit subnet id, not an AZ name -- eksctl would otherwise
    # re-resolve the AZ to a subnet itself, which is the exact step that let
    # a node group drift to a different subnet than the cluster's.
    subnets: ["${GPU_SUBNET}"]
    privateNetworking: true
    efaEnabled: true
    amiFamily: ${NODE_AMI_FAMILY}
    # Training checkpoints/data land on the node's EBS via emptyDir.
    volumeSize: 200
    labels:
      role: gpu-worker
      # The NVIDIA device plugin chart's default affinity requires this (or
      # Node Feature Discovery, which we don't run) -- without it the
      # DaemonSet sits at DESIRED 0 with no error anywhere, and
      # nvidia.com/gpu never becomes allocatable.
      nvidia.com/gpu.present: "true"
${CR_BLOCK}
EOF
        eksctl create nodegroup -f "$NODEGROUP_CONFIG"
        rm -f "$NODEGROUP_CONFIG"
        print_success "GPU node group created"

        print_section "Verifying GPU Nodes + EFA"
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
