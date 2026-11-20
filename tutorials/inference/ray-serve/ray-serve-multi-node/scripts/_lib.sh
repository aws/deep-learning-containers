#!/bin/bash
# _lib.sh - Shared helpers for the deploy/delete scripts.

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_section() { echo -e "\n${BLUE}=== $1 ===${NC}"; }
print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
# stderr, so messages still surface from inside $( ) command substitution.
print_error()   { echo -e "${RED}✗ $1${NC}" >&2; }

retry() {
    local attempts="$1" delay="$2"
    shift 2
    local i
    for ((i = 1; i <= attempts; i++)); do
        "$@" && return 0
        [ "$i" -lt "$attempts" ] && sleep "$delay"
    done
    return 1
}

_aws_status() {
    local not_found_pattern="$1"
    shift
    local out i
    for ((i = 1; i <= 5; i++)); do
        if out=$(aws "$@" 2>&1); then
            echo "$out"
            return 0
        fi
        echo "$out" | grep -q "$not_found_pattern" && { echo "NOT_FOUND"; return 0; }
        [ "$i" -lt 5 ] && sleep 8
    done
    echo "UNKNOWN"
}

get_cluster_status() {
    _aws_status "ResourceNotFoundException" eks describe-cluster \
        --name "$CLUSTER_NAME" --region "$REGION" --query "cluster.status" --output text
}

get_nodegroup_status() {
    _aws_status "ResourceNotFoundException" eks describe-nodegroup \
        --cluster-name "$CLUSTER_NAME" --region "$REGION" --nodegroup-name "$1" --query "nodegroup.status" --output text
}

get_addon_status() {
    _aws_status "ResourceNotFoundException" eks describe-addon \
        --cluster-name "$CLUSTER_NAME" --region "$REGION" --addon-name "$1" --query "addon.status" --output text
}

get_cf_stack_status() {
    _aws_status "does not exist" cloudformation describe-stacks \
        --stack-name "$1" --region "$REGION" --query "Stacks[0].StackStatus" --output text
}

get_gpu_capable_azs() {
    retry 5 8 aws ec2 describe-instance-type-offerings --region "$REGION" \
        --location-type availability-zone \
        --filters "Name=instance-type,Values=${GPU_NODE_TYPE}" \
        --query 'InstanceTypeOfferings[].Location' --output text | tr '\t' '\n' | sort -u
}

# EFA traffic cannot cross an AZ, so the GPU node group lives in exactly one.
resolve_gpu_az() {
    local capable
    capable=$(get_gpu_capable_azs)
    if [ -z "$capable" ]; then
        print_error "No AZ in $REGION offers $GPU_NODE_TYPE. Pick another region or instance type."
        exit 1
    fi
    if [ -n "$GPU_AZ" ]; then
        if ! echo "$capable" | grep -qx "$GPU_AZ"; then
            print_error "GPU_AZ='$GPU_AZ' does not offer $GPU_NODE_TYPE in $REGION. Available: $(echo $capable | tr '\n' ' ')"
            exit 1
        fi
        echo "$GPU_AZ"
        return 0
    fi
    echo "$capable" | head -1
}

wait_for_no_active_update() {
    local max_wait=300 waited=0 update_id update_status
    while [ "$waited" -lt "$max_wait" ]; do
        update_id=$(aws eks list-updates --name "$CLUSTER_NAME" --region "$REGION" \
            --query "updateIds[0]" --output text 2>/dev/null || echo "None")
        [ "$update_id" = "None" ] && return 0

        update_status=$(aws eks describe-update --name "$CLUSTER_NAME" --region "$REGION" \
            --update-id "$update_id" --query "update.status" --output text 2>/dev/null || echo "")
        [ "$update_status" != "InProgress" ] && return 0

        print_warning "Cluster has an update in progress ($update_id). Waiting for it to finish..."
        sleep 15
        waited=$((waited + 15))
    done
    print_warning "An EKS update was still in progress after ${max_wait}s. Proceeding anyway -- it may need a retry if this causes a conflict."
}

print_stack_failure_reason() {
    print_error "CloudFormation stack '$1' reported DELETE_FAILED. Failure details:"
    aws cloudformation describe-stack-events --stack-name "$1" --region "$REGION" \
        --query "StackEvents[?ResourceStatus=='DELETE_FAILED'].{Resource:LogicalResourceId,Reason:ResourceStatusReason}" \
        --output table 2>/dev/null || echo "  (could not fetch stack events)"
}

delete_cf_stack_and_wait() {
    local stack_name="$1" max_wait_minutes=15 deadline
    deadline=$(($(date +%s) + max_wait_minutes * 60))

    while true; do
        local stack_status
        stack_status=$(get_cf_stack_status "$stack_name")

        case "$stack_status" in
            NOT_FOUND)
                return 0
                ;;
            UNKNOWN)
                print_error "Could not determine status of CloudFormation stack '$stack_name' (repeated API errors). Check your AWS session and re-run."
                exit 1
                ;;
            *DELETE_IN_PROGRESS*)
                print_warning "Waiting for CloudFormation stack '$stack_name' to finish deleting..."
                aws cloudformation wait stack-delete-complete --stack-name "$stack_name" --region "$REGION" 2>/dev/null || true
                ;;
            *DELETE_FAILED*)
                print_stack_failure_reason "$stack_name"
                if [ "$(date +%s)" -ge "$deadline" ]; then
                    print_error "Stack '$stack_name' still DELETE_FAILED after ${max_wait_minutes}m. Manual investigation needed -- see the failure reason above."
                    exit 1
                fi
                print_warning "Retrying deletion of '$stack_name'..."
                wait_for_no_active_update
                eksctl delete cluster --name "$CLUSTER_NAME" --region "$REGION" 2>/dev/null || \
                    aws cloudformation delete-stack --stack-name "$stack_name" --region "$REGION" 2>/dev/null || true
                aws cloudformation wait stack-delete-complete --stack-name "$stack_name" --region "$REGION" 2>/dev/null || true
                ;;
            *)
                if [ "$(date +%s)" -ge "$deadline" ]; then
                    print_error "Stack '$stack_name' stuck in state '$stack_status' after ${max_wait_minutes}m. Check the CloudFormation console."
                    exit 1
                fi
                print_warning "Stack '$stack_name' is in state '$stack_status'. Waiting..."
                sleep 15
                ;;
        esac
    done
}

check_credentials() {
    if ! retry 5 8 aws sts get-caller-identity &>/dev/null; then
        print_error "AWS credentials not configured (or your session has expired). Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE, then re-run."
        exit 1
    fi
}

check_kubectl_prerequisites() {
    command -v kubectl &>/dev/null || { print_error "kubectl not found"; exit 1; }

    if ! retry 5 8 kubectl cluster-info &>/dev/null; then
        print_error "Cannot connect to Kubernetes cluster (repeated errors). Check your kubeconfig and AWS session, then re-run."
        exit 1
    fi

    if [ -n "$1" ] && ! kubectl get crd "$1" &>/dev/null; then
        print_error "CRD '$1' not found. $2"
        exit 1
    fi
}

# Requires the caller to have set SECONDS=0 near the top of the script.
print_elapsed() {
    echo -e "\n${BLUE}⏱ Elapsed: $((SECONDS / 60))m $((SECONDS % 60))s${NC}"
}

delete_nodegroup_and_wait() {
    local name="$1" max_wait_minutes=10 deadline
    deadline=$(($(date +%s) + max_wait_minutes * 60))

    wait_for_no_active_update
    eksctl delete nodegroup --cluster="$CLUSTER_NAME" --region="$REGION" --name="$name"

    while true; do
        local status
        status=$(get_nodegroup_status "$name")
        case "$status" in
            NOT_FOUND)
                return 0
                ;;
            UNKNOWN)
                print_error "Could not determine status of node group '$name' (repeated API errors). Check your AWS session and re-run."
                exit 1
                ;;
            *)
                if [ "$(date +%s)" -ge "$deadline" ]; then
                    print_error "Node group '$name' still in state '$status' after ${max_wait_minutes}m. Check the EKS console."
                    exit 1
                fi
                print_warning "Node group '$name' is in state '$status'. Waiting..."
                sleep 15
                ;;
        esac
    done
}