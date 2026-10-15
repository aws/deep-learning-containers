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

# AZs where cluster $CLUSTER_NAME (region $1, default $REGION) actually has a
# private subnet today. Node groups use privateNetworking, so an AZ the
# instance type is OFFERED in is still unusable if the cluster has no subnet
# there -- this is the thing get_gpu_capable_azs cannot see on its own. Empty
# output (not an error) if the cluster doesn't exist yet.
get_cluster_private_azs() {
    local region="${1:-$REGION}" subnets
    subnets=$(aws eks describe-cluster --region "$region" --name "$CLUSTER_NAME" \
        --query 'cluster.resourcesVpcConfig.subnetIds' --output text 2>/dev/null) || return 0
    [ -z "$subnets" ] || [ "$subnets" = "None" ] && return 0
    aws ec2 describe-subnets --region "$region" --subnet-ids $subnets \
        --query 'Subnets[?MapPublicIpOnLaunch==`false`].AvailabilityZone' --output text 2>/dev/null \
        | tr '\t' '\n' | sort -u
}

# The cluster's private subnet id in a specific AZ (region $2, default $REGION).
get_cluster_private_subnet() {
    local az="$1" region="${2:-$REGION}" subnets
    subnets=$(aws eks describe-cluster --region "$region" --name "$CLUSTER_NAME" \
        --query 'cluster.resourcesVpcConfig.subnetIds' --output text 2>/dev/null) || return 1
    [ -z "$subnets" ] || [ "$subnets" = "None" ] && return 1
    aws ec2 describe-subnets --region "$region" --subnet-ids $subnets \
        --query "Subnets[?AvailabilityZone=='${az}' && MapPublicIpOnLaunch==\`false\`].SubnetId" \
        --output text 2>/dev/null | awk '{print $1; exit}'
}

# EFA traffic cannot cross an AZ, so the GPU node group lives in exactly one --
# and it must be an AZ the EXISTING cluster already has a private subnet in,
# not just one the instance type happens to be offered in. Those two can
# diverge: capacity for $GPU_NODE_TYPE can shift to an AZ the cluster's VPC
# never got a subnet in (e.g. the cluster was created before that AZ had
# capacity, or in a different AZ set entirely). Picking an AZ on offering
# alone sends that mismatch all the way to a failed CloudFormation stack;
# catching it here reports it immediately with the fix.
resolve_gpu_az() {
    local capable cluster_azs usable
    capable=$(get_gpu_capable_azs)
    if [ -z "$capable" ]; then
        print_error "No AZ in $REGION offers $GPU_NODE_TYPE. Pick another region or instance type."
        exit 1
    fi

    cluster_azs=$(get_cluster_private_azs)
    if [ -n "$cluster_azs" ]; then
        usable=$(comm -12 <(echo "$capable") <(echo "$cluster_azs"))
        if [ -z "$usable" ]; then
            print_error "Cluster '$CLUSTER_NAME' has private subnets only in: $(echo $cluster_azs | tr '\n' ' ') -- but $GPU_NODE_TYPE is offered in $REGION only in: $(echo $capable | tr '\n' ' '). No AZ has both, so no node group placement can line up with the cluster's subnets."
            print_error "Fix: delete and recreate the cluster (deploy_cluster.sh prefers GPU-capable AZs when creating a NEW cluster, so this won't recur), or run 'bash find_gpu_capacity.sh' to find a region where capacity and an existing/plannable cluster subnet actually line up."
            exit 1
        fi
    else
        # Cluster doesn't exist yet (or has no subnets) -- nothing to line up
        # against; deploy_cluster.sh is what pins cluster AZs to GPU-capable
        # ones in that case.
        usable="$capable"
    fi

    if [ -n "$GPU_AZ" ]; then
        if echo "$usable" | grep -qx "$GPU_AZ"; then
            echo "$GPU_AZ"
            return 0
        fi
        if echo "$capable" | grep -qx "$GPU_AZ"; then
            print_error "GPU_AZ='$GPU_AZ' offers $GPU_NODE_TYPE, but cluster '$CLUSTER_NAME' has no private subnet there. Cluster's private-subnet AZs: $(echo $cluster_azs | tr '\n' ' ')"
        else
            print_error "GPU_AZ='$GPU_AZ' does not offer $GPU_NODE_TYPE in $REGION. Available: $(echo $capable | tr '\n' ' ')"
        fi
        exit 1
    fi
    echo "$usable" | head -1
}

# Resolves the AZ a GPU node group should launch into AND the cluster's exact
# subnet id there, so deploy_node_group.sh can hand eksctl a subnet id
# directly instead of an AZ it would have to re-resolve to a subnet itself --
# removing the one remaining step where it could land somewhere other than
# the cluster's own subnet.
resolve_gpu_subnet() {
    local az="$1" subnet
    subnet=$(get_cluster_private_subnet "$az")
    if [ -z "$subnet" ]; then
        print_error "No private subnet found for AZ '$az' in cluster '$CLUSTER_NAME'. Create the cluster first with deploy_cluster.sh."
        exit 1
    fi
    echo "$subnet"
}

# Validates $1 as a usable On-Demand Capacity Reservation and echoes its AZ.
resolve_capacity_reservation() {
    local id="$1" info state itype az avail
    info=$(aws ec2 describe-capacity-reservations --region "$REGION" --capacity-reservation-ids "$id" \
        --query 'CapacityReservations[0].[State,InstanceType,AvailabilityZone,AvailableInstanceCount]' \
        --output text 2>/dev/null) || true
    if [ -z "$info" ] || [ "$info" = "None" ]; then
        print_error "Capacity reservation '$id' not found in $REGION. Set REGION to the reservation's Region."
        exit 1
    fi
    read -r state itype az avail <<< "$info"
    if [ "$state" != "active" ]; then
        print_error "Capacity reservation '$id' is '$state', not 'active'."
        exit 1
    fi
    if [ "$itype" != "$GPU_NODE_TYPE" ]; then
        print_error "Capacity reservation '$id' is for $itype, but GPU_NODE_TYPE is $GPU_NODE_TYPE."
        exit 1
    fi
    if [ "$avail" -lt "$GPU_NODE_COUNT" ]; then
        print_error "Capacity reservation '$id' has $avail instance(s) available, but GPU_NODE_COUNT is $GPU_NODE_COUNT."
        exit 1
    fi
    echo "$az"
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

# Newer eksctl builds enable termination protection on every stack they create,
# so a plain delete-stack on a failed one is refused. Lifted only on stacks
# this cluster's eksctl owns (checked via its tag), never on anything else.
disable_eksctl_stack_protection() {
    local stack_name="$1" owner
    owner=$(aws cloudformation describe-stacks --stack-name "$stack_name" --region "$REGION" \
        --query "Stacks[0].Tags[?Key=='alpha.eksctl.io/cluster-name'].Value | [0]" --output text 2>/dev/null || true)
    if [ "$owner" != "$CLUSTER_NAME" ]; then
        print_error "Stack '$stack_name' is not tagged as eksctl's for cluster '$CLUSTER_NAME' (got '${owner}'). Not touching its termination protection."
        return 1
    fi
    aws cloudformation update-termination-protection --no-enable-termination-protection \
        --stack-name "$stack_name" --region "$REGION" >/dev/null
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

# Interactive "Proceed? (y/N)" confirmation, skippable with ASSUME_YES=1 --
# set by find_gpu_capacity.sh when it drives this script as part of its own
# already-confirmed deploy chain, so the user isn't asked the same question
# twice.
confirm() {
    local msg="${1:-Proceed?}"
    if [ "${ASSUME_YES:-}" = "1" ]; then
        echo "${msg} (y/N): y (ASSUME_YES=1)"
        return 0
    fi
    read -p "${msg} (y/N): " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || { echo "Cancelled."; exit 0; }
}

check_credentials() {
    if ! retry 5 8 aws sts get-caller-identity &>/dev/null; then
        print_error "AWS credentials not configured (or your session has expired). Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE, then re-run."
        exit 1
    fi
}

# Ensure helm is on PATH, installing it if missing: Homebrew when available
# (macOS/Linuxbrew), otherwise the official installer script.
ensure_helm() {
    command -v helm &>/dev/null && return 0
    print_warning "helm not found -- installing it..."
    if command -v brew &>/dev/null; then
        brew install helm
    elif command -v curl &>/dev/null; then
        curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
    else
        print_error "Cannot auto-install helm (no brew or curl). Install manually: https://helm.sh/docs/intro/install/"
        exit 1
    fi
    command -v helm &>/dev/null || { print_error "helm install ran but helm is still not on PATH."; exit 1; }
    print_success "helm installed: $(helm version --short 2>/dev/null || echo ok)"
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
