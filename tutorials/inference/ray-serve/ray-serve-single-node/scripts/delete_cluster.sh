#!/bin/bash
# =============================================================================
# delete_cluster.sh — Delete the EKS cluster and all associated resources.
#
# Usage:
#   bash delete_cluster.sh
# =============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/env.sh" ]; then
    source "$SCRIPT_DIR/env.sh"
fi

REGION=${REGION:-"us-west-2"}
CLUSTER_NAME=${CLUSTER_NAME:-"inference-cluster"}
export AWS_REGION="$REGION"
export AWS_DEFAULT_REGION="$REGION"

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error()   { echo -e "${RED}✗ $1${NC}"; }

SECONDS=0

# ─── Prerequisite Check ─────────────────────────────────────────────────────
for cmd in aws eksctl; do
    if ! command -v "$cmd" &>/dev/null; then
        print_error "Missing required tool: $cmd"
        exit 1
    fi
done

if ! aws sts get-caller-identity &>/dev/null; then
    print_error "AWS credentials not configured"
    exit 1
fi

# ─── Confirm ────────────────────────────────────────────────────────────────
echo -e "${BLUE}"
echo "=================================================="
echo "  Delete EKS Cluster"
echo "=================================================="
echo -e "${NC}"
echo "  Cluster: $CLUSTER_NAME"
echo "  Region:  $REGION"
echo

read -p "Are you sure you want to delete this cluster? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# ─── Delete Cluster ─────────────────────────────────────────────────────────
echo "Deleting EKS cluster '$CLUSTER_NAME'... (this takes 10-15 minutes)"
eksctl delete cluster --name "$CLUSTER_NAME" --region "$REGION"

# ─── Wait for CloudFormation cleanup ────────────────────────────────────────
CF_STACK_NAME="eksctl-${CLUSTER_NAME}-cluster"
CF_STATUS=$(aws cloudformation describe-stacks --stack-name "$CF_STACK_NAME" --region "$REGION" \
    --query "Stacks[0].StackStatus" --output text 2>/dev/null || echo "NOT_FOUND")

if [ "$CF_STATUS" != "NOT_FOUND" ]; then
    print_warning "Waiting for CloudFormation stack deletion..."
    aws cloudformation wait stack-delete-complete --stack-name "$CF_STACK_NAME" --region "$REGION" 2>/dev/null || true
    print_success "CloudFormation stack deleted"
fi

print_success "EKS cluster '$CLUSTER_NAME' deleted"

ELAPSED_MIN=$((SECONDS / 60))
ELAPSED_SEC=$((SECONDS % 60))
echo -e "\n${BLUE}⏱ Elapsed: ${ELAPSED_MIN}m ${ELAPSED_SEC}s${NC}"
