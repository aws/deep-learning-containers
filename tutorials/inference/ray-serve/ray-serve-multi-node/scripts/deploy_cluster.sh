#!/bin/bash
# deploy_cluster.sh — Deploy an EKS cluster with GPU-ready infrastructure.
# Idempotent: safe to re-run if interrupted. To delete, use delete_cluster.sh.
#
# Usage: bash deploy_cluster.sh
# Override: CLUSTER_NAME=my-cluster REGION=us-east-1 bash deploy_cluster.sh

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
    print_success "Prerequisites satisfied (aws, eksctl, kubectl)"
}

ensure_core_addons_and_oidc() {
    print_section "Verifying OIDC Provider + Core Addons"

    if retry 3 8 eksctl utils associate-iam-oidc-provider --cluster "$CLUSTER_NAME" --region "$REGION" --approve &>/dev/null; then
        print_success "IAM OIDC provider associated"
    else
        print_warning "Could not confirm/associate IAM OIDC provider (may already be associated)"
    fi

    local addon addon_status
    for addon in vpc-cni coredns kube-proxy; do
        addon_status=$(get_addon_status "$addon")
        case "$addon_status" in
            NOT_FOUND)
                print_warning "Addon '$addon' missing. Installing..."
                wait_for_no_active_update
                retry 3 8 aws eks create-addon --cluster-name "$CLUSTER_NAME" --region "$REGION" \
                    --addon-name "$addon" --resolve-conflicts OVERWRITE >/dev/null \
                    || { print_error "Failed to create addon '$addon' after retries."; exit 1; }
                print_success "Addon '$addon' creation started"
                ;;
            UNKNOWN)
                print_error "Could not determine status of addon '$addon' (repeated API errors). Check your AWS session and re-run."
                exit 1
                ;;
            *)
                print_success "Addon '$addon' present ($addon_status)"
                ;;
        esac
    done

    if ! kubectl get daemonset aws-node -n kube-system &>/dev/null; then
        echo "Waiting for VPC CNI to roll out..."
        for _ in $(seq 1 24); do
            kubectl get daemonset aws-node -n kube-system &>/dev/null && break
            sleep 5
        done
    fi
}

create_system_nodegroup() {
    eksctl create nodegroup \
        --cluster="$CLUSTER_NAME" \
        --region="$REGION" \
        --name=system-nodes \
        --node-type="$SYSTEM_NODE_TYPE" \
        --nodes="$SYSTEM_NODE_COUNT" \
        --nodes-min="$SYSTEM_NODE_COUNT" \
        --nodes-max="$SYSTEM_NODE_COUNT" \
        --node-labels="role=system" \
        --node-private-networking \
        --managed
}

resume_existing_cluster() {
    print_success "Cluster '$CLUSTER_NAME' already exists in $REGION (ACTIVE)"
    echo "Updating kubeconfig..."
    retry 3 8 aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$REGION"
    print_success "kubeconfig updated"

    ensure_core_addons_and_oidc

    local nodegroup_status
    nodegroup_status=$(get_nodegroup_status system-nodes)
    case "$nodegroup_status" in
        NOT_FOUND)
            print_warning "System node group not found. Creating..."
            create_system_nodegroup
            print_success "System node group created"
            ;;
        UNKNOWN)
            print_error "Could not determine status of node group 'system-nodes' (repeated API errors). Check your AWS session and re-run."
            exit 1
            ;;
        ACTIVE)
            print_success "System node group already exists (ACTIVE)"
            ;;
        CREATING|UPDATING)
            print_warning "System node group is already '$nodegroup_status'. Not creating a duplicate; re-run later to verify it finished."
            ;;
        *)
            print_warning "System node group is in unexpected state '$nodegroup_status'. Inspect it in the EKS console before re-running."
            ;;
    esac

    print_elapsed
}

create_new_cluster() {
    local cf_stack_name="eksctl-${CLUSTER_NAME}-cluster"
    delete_cf_stack_and_wait "$cf_stack_name"

    print_section "Step 1: Creating EKS Cluster"
    echo "This will take 15-20 minutes..."

    local gpu_azs all_azs cluster_azs az_json
    gpu_azs=$(get_gpu_capable_azs)
    if [ -z "$gpu_azs" ]; then
        print_error "No AZ in $REGION offers $GPU_NODE_TYPE. Choose a different region or instance type."
        exit 1
    fi
    all_azs=$(retry 5 8 aws ec2 describe-availability-zones --region "$REGION" \
        --query 'AvailabilityZones[?State==`available`].ZoneName' --output text | tr '\t' '\n' | sort -u)
    if [ -z "$all_azs" ]; then
        print_error "Could not list availability zones in $REGION after retries. Check your AWS session and re-run."
        exit 1
    fi
    # Prefer GPU-capable AZs so the GPU node group always has a usable subnet.
    cluster_azs="$gpu_azs"
    if [ "$(echo "$cluster_azs" | wc -l | tr -d ' ')" -lt 2 ]; then
        cluster_azs=$(printf '%s\n%s\n' "$gpu_azs" "$all_azs" | awk 'NF && !seen[$0]++' | head -2)
    fi
    if [ "$(echo "$cluster_azs" | wc -l | tr -d ' ')" -lt 2 ]; then
        print_error "EKS requires at least 2 AZs in $REGION; found: $(echo $cluster_azs | tr '\n' ' ')"
        exit 1
    fi
    az_json=$(echo "$cluster_azs" | awk 'NF{printf "%s\"%s\"", (c++?",":""), $0}')
    print_success "GPU-capable AZs: $(echo $gpu_azs | tr '\n' ' ')"
    print_success "Cluster AZs: $(echo $cluster_azs | tr '\n' ' ')"

    local cluster_config
    cluster_config=$(mktemp)
    cat > "$cluster_config" << EOF
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: $CLUSTER_NAME
  region: $REGION
  version: "${K8S_VERSION}"

availabilityZones: [${az_json}]

vpc:
  clusterEndpoints:
    privateAccess: true
    publicAccess: true

iam:
  withOIDC: true

# No aws-ebs-csi-driver: this sample uses only emptyDir volumes (no PVCs).
addons:
  - name: vpc-cni
  - name: coredns
  - name: kube-proxy
EOF

    eksctl create cluster -f "$cluster_config"
    rm -f "$cluster_config"

    local post_create_status
    post_create_status=$(get_cluster_status)
    if [ "$post_create_status" != "ACTIVE" ]; then
        print_error "Cluster status is '$post_create_status' after creation (expected ACTIVE). Check the EKS console -- eksctl itself may have still succeeded even if this check couldn't confirm it."
        exit 1
    fi
    print_success "EKS cluster created"

    print_section "Step 2: Creating System Node Group"
    echo "Creating ${SYSTEM_NODE_COUNT} x ${SYSTEM_NODE_TYPE} node(s)..."
    create_system_nodegroup
    print_success "System node group created"

    print_section "Step 3: Verifying Cluster"
    kubectl get nodes || print_warning "Could not fetch nodes right now (transient?). The cluster and node group were created successfully above."
    print_success "Cluster is ready"
    echo "Cluster endpoint:"
    retry 3 8 aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" \
        --query "cluster.endpoint" --output text \
        || print_warning "Could not fetch the cluster endpoint right now (transient?)."

    print_elapsed
}

echo -e "${BLUE}"
echo "=================================================="
echo "  Deploy EKS Cluster"
echo "=================================================="
echo -e "${NC}"
echo "  Region:       $REGION"
echo "  Cluster:      $CLUSTER_NAME"
echo "  K8s Version:  $K8S_VERSION"
echo "  System Nodes: ${SYSTEM_NODE_COUNT} x ${SYSTEM_NODE_TYPE}"
echo "  AWS Auth:     ${AWS_PROFILE:-environment credentials}"
echo

read -p "Proceed? (y/N): " -n 1 -r
echo
[[ $REPLY =~ ^[Yy]$ ]] || { echo "Cancelled."; exit 0; }

check_prerequisites

print_section "Checking for Existing Cluster"
CLUSTER_STATUS=$(get_cluster_status)

case "$CLUSTER_STATUS" in
    NOT_FOUND)  create_new_cluster ;;
    ACTIVE)     resume_existing_cluster ;;
    UNKNOWN)
        print_error "Could not determine whether cluster '$CLUSTER_NAME' exists (repeated API errors -- check your AWS session/credentials and re-run)."
        exit 1
        ;;
    CREATING)
        print_error "Cluster '$CLUSTER_NAME' is still being created by a previous run. Wait for it to finish, then re-run this script."
        exit 1
        ;;
    DELETING)
        print_error "Cluster '$CLUSTER_NAME' is currently being deleted. Wait for deletion to finish, then re-run this script."
        exit 1
        ;;
    *)
        print_error "Cluster '$CLUSTER_NAME' exists in an unexpected state ($CLUSTER_STATUS). Inspect it in the EKS console before re-running."
        exit 1
        ;;
esac