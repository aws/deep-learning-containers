#!/bin/bash
# deploy_all.sh - One command for the infrastructure: find where the GPU nodes
# can launch, then build the cluster, GPU node group, device plugins and KubeRay
# there. It stops after KubeRay unless --train is given; the training job is a
# separate step you re-run on its own. Each step's result feeds the next:
#
#   find_gpu_capacity.sh  -> picks REGION + a ranked list of candidate AZs
#   deploy_cluster.sh     -> built (or reused) in that REGION
#   deploy_node_group.sh  -> tried in each candidate AZ in turn; if the node
#                            group doesn't reach ACTIVE, it is removed and the
#                            next AZ is tried
#   install_gpu_plugins.sh, install_kuberay.sh
#   deploy_ray_train_job.sh  (only with --train)
#
# Each step is the same idempotent script you can run by hand; this sequences
# them, passes REGION/GPU_AZ along, and answers their prompts after asking you
# once up front. Re-running it after a failure picks up where it stopped.
#
# Usage:
#   bash deploy_all.sh                          # search REGION from env.sh, build, stop after KubeRay
#   bash deploy_all.sh -r "us-east-2 us-east-1"  # search several Regions
#   bash deploy_all.sh --train                  # also run the training job
#   bash deploy_all.sh --plan                   # show what it would do; read-only
#   bash deploy_all.sh --no-probe               # rank AZs on quota only (see below)
#   bash deploy_all.sh -y                       # don't ask at all
#   GPU_AZ=us-east-2b bash deploy_all.sh        # skip the search, use this AZ
#
# Capacity: quota and offerings alone can't tell an AZ with capacity from one
# without, so by default the search probes each candidate AZ with an On-Demand
# Capacity Reservation that is cancelled immediately (see find_gpu_capacity.sh;
# billed at the On-Demand rate for the seconds it exists). --no-probe skips it.
#
# The chosen REGION/GPU_AZ are saved to scripts/.placement.env, which env.sh
# loads, so later runs of deploy_ray_train_job.sh or the delete_*.sh scripts
# target the same Region without re-exporting anything.

# Captured before env.sh can fill them from saved state or defaults, so an
# explicit choice by the caller is distinguishable from a remembered one.
CALLER_GPU_AZ="${GPU_AZ:-}"

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
source "$SCRIPT_DIR/_lib.sh"

SECONDS=0
PLACEMENT_FILE="$SCRIPT_DIR/.placement.env"
REGIONS="$REGION"
ALL_REGIONS=false
PROBE=auto
TRAIN=false
YES=false
PLAN=false
AZ_LIST=""

usage() { sed -n '2,35p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        -r|--regions) REGIONS="$(echo "$2" | tr ',' ' ')"; shift 2 ;;
        -a|--all-regions) ALL_REGIONS=true; shift ;;
        --azs) AZ_LIST="$(echo "$2" | tr ',' ' ')"; shift 2 ;;
        --probe) PROBE=true; shift ;;
        --no-probe) PROBE=false; shift ;;
        --train) TRAIN=true; shift ;;
        --plan) PLAN=true; shift ;;
        -y|--yes) YES=true; shift ;;
        -h|--help) usage 0 ;;
        *) print_error "Unknown argument: $1"; usage 1 ;;
    esac
done

# A plan is read-only, so it never probes unless asked to explicitly.
if [ "$PROBE" = auto ]; then
    $PLAN && PROBE=false || PROBE=true
fi

set_region() {
    export REGION="$1" AWS_REGION="$1" AWS_DEFAULT_REGION="$1"
}

save_placement() {
    cat > "$PLACEMENT_FILE" <<EOF
# Written by deploy_all.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ). Loaded by env.sh so the
# other scripts target the same place. Values you export yourself still win.
# Delete this file to go back to the defaults in env.sh.
export CLUSTER_NAME="\${CLUSTER_NAME:-${CLUSTER_NAME}}"
export REGION="\${REGION:-${REGION}}"
export GPU_AZ="\${GPU_AZ:-${GPU_AZ:-}}"
EOF
}

nodegroup_az() {
    local subnet
    subnet=$(aws eks describe-nodegroup --region "$REGION" --cluster-name "$CLUSTER_NAME" \
        --nodegroup-name "$GPU_NODEGROUP_NAME" --query 'nodegroup.subnets[0]' --output text 2>/dev/null) || return 0
    aws ec2 describe-subnets --region "$REGION" --subnet-ids "$subnet" \
        --query 'Subnets[0].AvailabilityZone' --output text 2>/dev/null || true
}

# Why the last launch attempt failed, straight from the Auto Scaling group.
last_launch_error() {
    local asg
    asg=$(aws autoscaling describe-auto-scaling-groups --region "$REGION" \
        --filters "Name=tag:eks:nodegroup-name,Values=${GPU_NODEGROUP_NAME}" \
        --query 'AutoScalingGroups[0].AutoScalingGroupName' --output text 2>/dev/null || true)
    [ -z "$asg" ] || [ "$asg" = "None" ] && return 0
    aws autoscaling describe-scaling-activities --region "$REGION" --auto-scaling-group-name "$asg" \
        --max-items 1 --query 'Activities[0].StatusMessage' --output text 2>/dev/null | cut -c1-220 || true
}

step() {
    local n="$1" label="$2"; shift 2
    print_section "Step ${n}: ${label}"
    bash "$SCRIPT_DIR/$@"
}

# ------------------------------------------------------------------ banner
echo -e "${BLUE}"
echo "=================================================="
echo "  Deploy infrastructure: multi-node Ray Train on EKS"
echo "=================================================="
echo -e "${NC}"
echo "  GPU nodes:    ${GPU_NODE_COUNT} x ${GPU_NODE_TYPE}"
if [ -n "$CAPACITY_RESERVATION_ID" ]; then
    echo "  Placement:    ${REGION}, from capacity reservation ${CAPACITY_RESERVATION_ID} (no search)"
elif [ -n "$CALLER_GPU_AZ" ]; then
    echo "  Placement:    ${REGION} / ${CALLER_GPU_AZ} (set by you, no search)"
else
    echo "  Search:       $($ALL_REGIONS && echo "all enabled Regions" || echo "$REGIONS")"
    echo "  Probe:        $($PROBE && echo "yes: a Capacity Reservation per candidate AZ, cancelled immediately" || echo "no: rank on quota only")"
fi
echo "  Then:         cluster -> GPU node group -> device plugins -> KubeRay$($TRAIN && echo " -> training job")"
$PLAN && echo "  Mode:         plan only (nothing is created)"
echo

for c in aws eksctl kubectl envsubst; do
    command -v "$c" &>/dev/null || { print_error "Missing required tool: $c"; exit 1; }
done
check_credentials

if ! $PLAN; then
    echo "  Creates billable resources: an EKS cluster (if none exists), ${GPU_NODE_COUNT} x ${GPU_NODE_TYPE},"
    echo "  a NAT gateway and the system node. Roughly \$9.50/hr while the GPU nodes are up."
    echo
    if ! $YES; then
        confirm "Proceed with the whole chain without further prompts?"
    fi
    # Every script below would otherwise ask again for the decision just made.
    export ASSUME_YES=1
fi

# ------------------------------------------------------------------ placement
CANDIDATES=""
SOURCE=""
if [ -n "$CAPACITY_RESERVATION_ID" ]; then
    CANDIDATES=$(resolve_capacity_reservation "$CAPACITY_RESERVATION_ID")
    SOURCE="capacity reservation $CAPACITY_RESERVATION_ID"
elif [ -n "$CALLER_GPU_AZ" ]; then
    CANDIDATES="$CALLER_GPU_AZ"
    SOURCE="GPU_AZ set by you"
elif [ -n "$AZ_LIST" ]; then
    CANDIDATES="$AZ_LIST"
    SOURCE="--azs"
else
    # Reuse a GPU node group that is already up rather than searching again.
    for R in $($ALL_REGIONS && echo "$REGION" || echo "$REGIONS"); do
        st=$(aws eks describe-nodegroup --region "$R" --cluster-name "$CLUSTER_NAME" \
            --nodegroup-name "$GPU_NODEGROUP_NAME" --query 'nodegroup.status' --output text 2>/dev/null || true)
        if [ "$st" = "ACTIVE" ]; then
            set_region "$R"
            CANDIDATES="$(nodegroup_az)"
            SOURCE="existing ACTIVE node group"
            break
        fi
    done
fi

if [ -z "$CANDIDATES" ]; then
    print_section "Step 0: Finding a Region/AZ with quota and capacity"
    EMIT=$(mktemp)
    SEARCH_ARGS=(--emit "$EMIT" --no-offer)
    $ALL_REGIONS && SEARCH_ARGS+=(-a) || SEARCH_ARGS+=(-r "$REGIONS")
    $PROBE && SEARCH_ARGS+=(--probe)
    set +e
    bash "$SCRIPT_DIR/find_gpu_capacity.sh" "${SEARCH_ARGS[@]}"
    set -e
    if [ ! -s "$EMIT" ]; then
        rm -f "$EMIT"
        print_error "No usable Region/AZ for ${GPU_NODE_COUNT} x ${GPU_NODE_TYPE}$($PROBE && echo " with capacity right now"). See the table above: request more quota where it says SHORT, or widen the search with -r \"...\" or -a."
        exit 1
    fi
    # Build in the best Region; fall back across its other candidate AZs.
    BEST_REGION=$(head -1 "$EMIT" | cut -d' ' -f1)
    set_region "$BEST_REGION"
    CANDIDATES=$(awk -v r="$BEST_REGION" '$1 == r { print $2 }' "$EMIT" | tr '\n' ' ')
    rm -f "$EMIT"
    SOURCE="$($PROBE && echo "capacity probe" || echo "quota only, capacity not confirmed")"
fi

print_section "Placement"
print_success "Region ${REGION}, GPU AZ order: ${CANDIDATES}  (${SOURCE})"

if $PLAN; then
    echo
    echo "  A real run would execute, with these values:"
    echo "    REGION=${REGION} bash deploy_cluster.sh"
    for AZ in $CANDIDATES; do
        echo "    REGION=${REGION} GPU_AZ=${AZ} bash deploy_node_group.sh   # next AZ only if this one fails"
    done
    echo "    bash install_gpu_plugins.sh && bash install_kuberay.sh"
    $TRAIN && echo "    bash deploy_ray_train_job.sh"
    $PROBE || echo -e "\n  Capacity was not checked. The real run probes each AZ unless you pass --no-probe."
    exit 0
fi

# ------------------------------------------------------------------ build
step 1 "EKS cluster in ${REGION}" deploy_cluster.sh
GPU_AZ="" save_placement

NG_OK=false
for AZ in $CANDIDATES; do
    export GPU_AZ="$AZ"
    st=$(get_nodegroup_status "$GPU_NODEGROUP_NAME")
    case "$st" in
        ACTIVE)
            GPU_AZ="$(nodegroup_az)"; export GPU_AZ
            print_success "GPU node group already ACTIVE in ${GPU_AZ}"
            NG_OK=true; break ;;
        CREATING|UPDATING|DELETING)
            print_error "GPU node group is ${st} from another run. Wait for it to finish, then re-run this script."
            exit 1 ;;
        UNKNOWN)
            print_error "Could not read the GPU node group status (repeated API errors). Check your AWS session and re-run."
            exit 1 ;;
        NOT_FOUND) ;;
        *)
            print_warning "Removing GPU node group left in state ${st}..."
            delete_nodegroup_and_wait "$GPU_NODEGROUP_NAME" ;;
    esac

    attempt_start=$(date -u +%Y-%m-%dT%H:%M:%S)
    set +e
    step 2 "GPU node group in ${AZ}" deploy_node_group.sh
    rc=$?
    set -e
    st=$(get_nodegroup_status "$GPU_NODEGROUP_NAME")
    if [ "$st" = "ACTIVE" ]; then
        NG_OK=true; break
    fi

    # Another AZ only helps if this attempt got as far as launching instances.
    # No node group stack created during this attempt means it failed before
    # that (e.g. a stale stack it could not remove), and every AZ would fail
    # the same way. UTC ISO timestamps compare correctly as strings.
    stack_created=$(aws cloudformation describe-stacks --region "$REGION" \
        --stack-name "eksctl-${CLUSTER_NAME}-nodegroup-${GPU_NODEGROUP_NAME}" \
        --query 'Stacks[0].CreationTime' --output text 2>/dev/null || true)
    if [ "$st" = "NOT_FOUND" ] && { [ -z "$stack_created" ] || [[ "$stack_created" < "$attempt_start" ]]; }; then
        print_error "deploy_node_group.sh failed in ${AZ} (exit ${rc}) before creating the node group, so this is not a capacity problem and other AZs would fail the same way. Fix the error above and re-run."
        exit 1
    fi

    reason=$(last_launch_error)
    print_warning "GPU node group did not become ACTIVE in ${AZ} (exit ${rc}, status ${st})."
    [ -n "$reason" ] && [ "$reason" != "None" ] && echo "  Last launch attempt: ${reason}"
    case "$st" in
        NOT_FOUND|UNKNOWN) ;;
        *) print_warning "Removing it before trying the next AZ..."
           delete_nodegroup_and_wait "$GPU_NODEGROUP_NAME" ;;
    esac
done

if ! $NG_OK; then
    print_error "The GPU node group could not launch in any of: ${CANDIDATES}. Re-run later, or search more Regions: bash deploy_all.sh -r \"us-east-1 us-west-2\""
    exit 1
fi
save_placement
print_success "GPU nodes up in ${REGION} / ${GPU_AZ} (saved to $(basename "$PLACEMENT_FILE"))"

step 3 "NVIDIA + EFA device plugins" install_gpu_plugins.sh
step 4 "KubeRay operator" install_kuberay.sh

if $TRAIN; then
    step 5 "RayCluster + training job" deploy_ray_train_job.sh
fi

# ------------------------------------------------------------------ summary
print_section "Done"
print_success "Region ${REGION}, GPU AZ ${GPU_AZ}, ${GPU_NODE_COUNT} x ${GPU_NODE_TYPE}"
$TRAIN || echo "  Run the training job:     bash deploy_ray_train_job.sh"
echo "  Tear down (reverse order, same Region via .placement.env):"
echo "    bash delete_ray_train_job.sh && bash install_kuberay.sh cleanup && \\"
echo "    bash install_gpu_plugins.sh cleanup && bash delete_node_group.sh && bash delete_cluster.sh"
print_elapsed
