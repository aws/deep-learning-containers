#!/bin/bash
# =============================================================================
# deploy_cluster.sh — Deploy an EKS cluster with GPU-ready infrastructure.
#
# Standalone script. Configurable via environment variables or defaults below.
#
# Usage:
#   bash deploy_cluster.sh            # Create cluster
#   bash deploy_cluster.sh cleanup    # Delete cluster and all resources
#
# Override defaults:
#   CLUSTER_NAME=my-cluster REGION=us-east-1 bash deploy_cluster.sh
# =============================================================================

set -eo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

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
    local missing=()
    command -v aws &>/dev/null || missing+=("aws")
    command -v eksctl &>/dev/null || missing+=("eksctl")
    command -v kubectl &>/dev/null || missing+=("kubectl")

    if [ ${#missing[@]} -gt 0 ]; then
        print_error "Missing required tools: ${missing[*]}"
        echo "Install them before running this script."
        exit 1
    fi

    if ! aws sts get-caller-identity &>/dev/null; then
        print_error "AWS credentials not configured. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE."
        exit 1
    fi
    print_success "Prerequisites satisfied (aws, eksctl, kubectl)"
}

# ─── Cleanup ─────────────────────────────────────────────────────────────────
cleanup_cluster() {
    echo -e "${BLUE}"
    echo "=================================================="
    echo "  EKS Cluster Deletion"
    echo "=================================================="
    echo -e "${NC}"
    echo "  Cluster: $CLUSTER_NAME"
    echo "  Region:  $REGION"
    echo ""

    read -p "Are you sure you want to delete the cluster? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi

    print_section "Deleting EKS Cluster"
    echo "This will take 10-15 minutes..."
    eksctl delete cluster --name "$CLUSTER_NAME" --region "$REGION" --force

    CF_STACK_NAME="eksctl-${CLUSTER_NAME}-cluster"
    CF_STATUS=$(aws cloudformation describe-stacks --stack-name "$CF_STACK_NAME" --region "$REGION" \
        --query "Stacks[0].StackStatus" --output text 2>/dev/null || echo "NOT_FOUND")

    if [ "$CF_STATUS" != "NOT_FOUND" ]; then
        echo "Waiting for CloudFormation stack deletion..."
        aws cloudformation wait stack-delete-complete --stack-name "$CF_STACK_NAME" --region "$REGION" 2>/dev/null || true
        print_success "CloudFormation stack deleted"
    fi

    print_success "EKS cluster '$CLUSTER_NAME' deleted"

    ELAPSED_MIN=$((SECONDS / 60))
    ELAPSED_SEC=$((SECONDS % 60))
    echo -e "\n${BLUE}⏱ Elapsed: ${ELAPSED_MIN}m ${ELAPSED_SEC}s${NC}"
}

# ─── Main: route subcommand ─────────────────────────────────────────────────
COMMAND=${1:-"deploy"}
if [ "$COMMAND" = "cleanup" ]; then
    check_prerequisites
    cleanup_cluster
    exit 0
fi

# ─── Deploy ──────────────────────────────────────────────────────────────────
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
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

check_prerequisites

# ─── Check for Existing Cluster ──────────────────────────────────────────────
print_section "Checking for Existing Cluster"

if aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" &>/dev/null; then
    print_success "Cluster '$CLUSTER_NAME' already exists in $REGION"
    echo "Updating kubeconfig..."
    aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$REGION"
    print_success "kubeconfig updated"

    # Ensure system node group exists
    if eksctl get nodegroup --cluster="$CLUSTER_NAME" --region="$REGION" --name=system-nodes &>/dev/null; then
        print_success "System node group already exists"
    else
        print_warning "System node group not found. Creating..."
        eksctl create nodegroup \
            --cluster="$CLUSTER_NAME" \
            --region="$REGION" \
            --name=system-nodes \
            --node-type="$SYSTEM_NODE_TYPE" \
            --nodes="$SYSTEM_NODE_COUNT" \
            --nodes-min="$SYSTEM_NODE_COUNT" \
            --nodes-max="$SYSTEM_NODE_COUNT" \
            --node-labels="role=system" \
            --managed
        print_success "System node group created"
    fi

    ELAPSED_MIN=$((SECONDS / 60))
    ELAPSED_SEC=$((SECONDS % 60))
    echo -e "\n${BLUE}⏱ Elapsed: ${ELAPSED_MIN}m ${ELAPSED_SEC}s${NC}"
    exit 0
fi

# Clean up stale CloudFormation stacks from failed previous attempts
CF_STACK_NAME="eksctl-${CLUSTER_NAME}-cluster"
CF_STATUS=$(aws cloudformation describe-stacks --stack-name "$CF_STACK_NAME" --region "$REGION" \
    --query "Stacks[0].StackStatus" --output text 2>/dev/null || echo "NOT_FOUND")

if [ "$CF_STATUS" != "NOT_FOUND" ]; then
    if [[ "$CF_STATUS" == *"DELETE_IN_PROGRESS"* ]]; then
        print_warning "Stack deletion in progress. Waiting..."
        aws cloudformation wait stack-delete-complete --stack-name "$CF_STACK_NAME" --region "$REGION" 2>/dev/null || true
        print_success "Stack deletion complete"
    elif [[ "$CF_STATUS" == *"FAILED"* ]] || [[ "$CF_STATUS" == *"ROLLBACK"* ]]; then
        print_warning "Stale CloudFormation stack found ($CF_STATUS). Cleaning up..."
        aws cloudformation delete-stack --stack-name "$CF_STACK_NAME" --region "$REGION" 2>/dev/null || true
        aws cloudformation wait stack-delete-complete --stack-name "$CF_STACK_NAME" --region "$REGION" 2>/dev/null || true
        print_success "Stale stack removed"
    else
        print_error "CloudFormation stack '$CF_STACK_NAME' exists ($CF_STATUS)"
        echo "Clean up with: eksctl delete cluster --name $CLUSTER_NAME --region $REGION"
        exit 1
    fi
fi

# ─── Step 1: Create Cluster ─────────────────────────────────────────────────
print_section "Step 1: Creating EKS Cluster"
echo "This will take 15-20 minutes..."

ALL_AZS=$(aws ec2 describe-availability-zones --region "$REGION" \
    --query 'AvailabilityZones[?State==`available`].ZoneName' --output json)
print_success "Available AZs: $ALL_AZS"

CLUSTER_CONFIG=$(mktemp)
cat > "$CLUSTER_CONFIG" << EOF
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: $CLUSTER_NAME
  region: $REGION
  version: "${K8S_VERSION}"

availabilityZones: ${ALL_AZS}

vpc:
  clusterEndpoints:
    privateAccess: true
    publicAccess: true

iam:
  withOIDC: true

addons:
  - name: vpc-cni
  - name: coredns
  - name: kube-proxy
  - name: aws-ebs-csi-driver
    wellKnownPolicies:
      ebsCSIController: true
EOF

eksctl create cluster -f "$CLUSTER_CONFIG"
rm -f "$CLUSTER_CONFIG"

if ! aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" &>/dev/null; then
    print_error "Cluster creation failed"
    exit 1
fi
print_success "EKS cluster created"

# ─── Step 2: Create System Node Group ────────────────────────────────────────
print_section "Step 2: Creating System Node Group"

echo "Creating ${SYSTEM_NODE_COUNT} x ${SYSTEM_NODE_TYPE} nodes..."

eksctl create nodegroup \
    --cluster="$CLUSTER_NAME" \
    --region="$REGION" \
    --name=system-nodes \
    --node-type="$SYSTEM_NODE_TYPE" \
    --nodes="$SYSTEM_NODE_COUNT" \
    --nodes-min="$SYSTEM_NODE_COUNT" \
    --nodes-max="$SYSTEM_NODE_COUNT" \
    --node-labels="role=system" \
    --managed

print_success "System node group created"

# ─── Step 3: Verify ──────────────────────────────────────────────────────────
print_section "Step 3: Verifying Cluster"

kubectl get nodes
echo
print_success "Cluster is ready"
echo
echo "Cluster endpoint:"
aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" \
    --query "cluster.endpoint" --output text

ELAPSED_MIN=$((SECONDS / 60))
ELAPSED_SEC=$((SECONDS % 60))
echo -e "\n${BLUE}⏱ Elapsed: ${ELAPSED_MIN}m ${ELAPSED_SEC}s${NC}"
