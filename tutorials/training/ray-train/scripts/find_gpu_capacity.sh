#!/bin/bash
# find_gpu_capacity.sh - Find Regions/AZs where you can actually launch
# GPU_NODE_COUNT x GPU_NODE_TYPE, before deploy_node_group.sh finds out the
# slow way (an EKS node group that sits in CREATING while its Auto Scaling
# group retries InsufficientInstanceCapacity every few minutes).
#
# Three different things have to be true, and AWS reports them separately:
#   1. The instance type is OFFERED in the AZ        (describe-instance-type-offerings)
#   2. Your On-Demand vCPU QUOTA has room for N nodes (Service Quotas + running instances)
#   3. There is CAPACITY right now                    (no read-only API answers this)
# (1) and (2) are checked for free. For (3) the script reports EC2's Spot
# placement score as a rough signal, and --probe gives the real answer by
# creating an On-Demand Capacity Reservation for N instances and cancelling it
# immediately (billed at the On-Demand rate for the moment it exists).
#
# Usage:
#   bash find_gpu_capacity.sh                       # REGION from env.sh
#   bash find_gpu_capacity.sh -r "us-east-2 us-west-2 us-east-1"
#   bash find_gpu_capacity.sh -a                    # every enabled Region (slower)
#   bash find_gpu_capacity.sh -t p5.48xlarge -n 2   # a different shape
#   bash find_gpu_capacity.sh -r us-east-2 --probe  # definitive capacity check
#   --emit FILE   write every usable "region az" pair, best first (for deploy_all.sh)
#   --no-offer    don't offer to deploy at the end (deploy_all.sh does that part)
#
# Read-only unless --probe is given (and confirmed), OR you say "yes" to the
# deploy prompt at the end: once a Region/AZ is recommended, this script asks
# once whether to build everything there now, and on "yes" it exports
# REGION/GPU_AZ from its own recommendation and runs deploy_cluster.sh,
# deploy_node_group.sh, install_gpu_plugins.sh, and install_kuberay.sh in
# that order -- so the cluster, GPU node group, and system node can never
# land in a Region/AZ other than the one capacity was actually found in.
# It stops there; the training job is run separately (deploy_ray_train_job.sh).

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
source "$SCRIPT_DIR/_lib.sh"

SECONDS=0
TYPE="$GPU_NODE_TYPE"
COUNT="$GPU_NODE_COUNT"
REGIONS="$REGION"
ALL_REGIONS=false
PROBE=false
EMIT_FILE=""
NO_OFFER=false

usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        -r|--regions) REGIONS="$(echo "$2" | tr ',' ' ')"; shift 2 ;;
        -a|--all-regions) ALL_REGIONS=true; shift ;;
        -t|--type) TYPE="$2"; shift 2 ;;
        -n|--count) COUNT="$2"; shift 2 ;;
        --probe) PROBE=true; shift ;;
        --emit) EMIT_FILE="$2"; shift 2 ;;
        --no-offer) NO_OFFER=true; shift ;;
        -h|--help) usage 0 ;;
        *) print_error "Unknown argument: $1"; usage 1 ;;
    esac
done

command -v aws &>/dev/null || { print_error "aws CLI not found"; exit 1; }
check_credentials

if $ALL_REGIONS; then
    REGIONS=$(aws ec2 describe-regions --region "$REGION" --query 'Regions[].RegionName' --output text | tr '\t' '\n' | sort | tr '\n' ' ')
fi

# --- instance family -> quota name + the instance-type prefixes it covers ---
FAMILY=$(echo "$TYPE" | sed -E 's/^([a-z]+)[0-9].*/\1/')
case "$FAMILY" in
    g|vt) QLABEL="G and VT"; USAGE_EXPR="starts_with(InstanceType,'g') || starts_with(InstanceType,'vt')" ;;
    p)    QLABEL="P";        USAGE_EXPR="starts_with(InstanceType,'p')" ;;
    trn)  QLABEL="Trn";      USAGE_EXPR="starts_with(InstanceType,'trn')" ;;
    inf)  QLABEL="Inf";      USAGE_EXPR="starts_with(InstanceType,'inf')" ;;
    dl)   QLABEL="DL";       USAGE_EXPR="starts_with(InstanceType,'dl')" ;;
    *)    QLABEL="Standard (A, C, D, H, I, M, R, T, Z)"; USAGE_EXPR="" ;;
esac
QUOTA_NAME="Running On-Demand ${QLABEL} instances"
SPOT_QUOTA_NAME="All ${QLABEL} Spot Instance Requests"

# Look a quota up by name. The CLI paginates list-service-quotas and applies
# --query to every page, so filter per page and keep the first real hit
# rather than indexing with [0] (which yields "None" for each other page).
# Falls back to the AWS default when the quota was never changed.
quota_lookup() {  # $1=region $2=quota name -> "value code"
    local r="$1" name="$2" hit
    for api in list-service-quotas list-aws-default-service-quotas; do
        hit=$(retry 4 5 aws service-quotas "$api" --region "$r" --service-code ec2 \
            --query "Quotas[?QuotaName=='${name}'].[Value,QuotaCode]" --output text 2>/dev/null \
            | awk 'NF == 2 && $1 != "None" && !seen++ { print }') || true
        [ -n "$hit" ] && { echo "$hit"; return; }
    done
    echo "None None"
}

# Reservations created by --probe; cancelled on any exit so an interrupted run
# never leaves a billed reservation behind.
PROBE_IDS=""
cleanup_probes() {
    local entry r id
    for entry in $PROBE_IDS; do
        r="${entry%%:*}"; id="${entry#*:}"
        aws ec2 cancel-capacity-reservation --region "$r" --capacity-reservation-id "$id" >/dev/null 2>&1 \
            && echo "  cancelled leftover reservation $id ($r)" >&2 || true
    done
}
trap cleanup_probes EXIT

echo -e "${BLUE}"
echo "=================================================="
echo "  Find GPU capacity: ${COUNT} x ${TYPE}"
echo "=================================================="
echo -e "${NC}"
echo "  Regions: $REGIONS"
echo "  Probe:   $($PROBE && echo "yes (creates + cancels a Capacity Reservation per candidate AZ)" || echo "no (read-only)")"
echo

if $PROBE; then
    print_warning "--probe creates an On-Demand Capacity Reservation for ${COUNT} x ${TYPE} in each candidate AZ and cancels it immediately. Each one is billed at the On-Demand rate for the seconds it exists (typically cents)."
    if [ "${ASSUME_YES:-}" != "1" ]; then
        read -p "Proceed with probing? (y/N): " -n 1 -r
        echo
        [[ $REPLY =~ ^[Yy]$ ]] || PROBE=false
    fi
fi

ROWS=""           # region|az|quota_ok|spot|subnet|probe
VCPU=""; EFA=""
QUOTA_HINTS=""

for R in $REGIONS; do
    echo "  checking ${R}..." >&2
    AZS=$(aws ec2 describe-instance-type-offerings --region "$R" --location-type availability-zone \
          --filters "Name=instance-type,Values=${TYPE}" --query 'InstanceTypeOfferings[].Location' \
          --output text 2>/dev/null | tr '\t' '\n' | sort || true)
    if [ -z "$AZS" ]; then
        ROWS+="$R|-|-|-|-|not offered"$'\n'
        continue
    fi

    if [ -z "$VCPU" ]; then
        read -r VCPU EFA < <(aws ec2 describe-instance-types --region "$R" --instance-types "$TYPE" \
            --query 'InstanceTypes[0].[VCpuInfo.DefaultVCpus,NetworkInfo.EfaSupported]' --output text)
    fi
    NEED=$((VCPU * COUNT))

    read -r QUOTA QCODE < <(quota_lookup "$R" "$QUOTA_NAME")
    QUOTA=${QUOTA%.*}
    read -r SPOT_QUOTA _ < <(quota_lookup "$R" "$SPOT_QUOTA_NAME")
    SPOT_QUOTA=${SPOT_QUOTA%.*}
    # Spot placement scores factor in your Spot quota: below NEED they are
    # pinned near 1 and say nothing about real capacity, so flag them.
    SPOT_CAPPED=false
    if [ "$SPOT_QUOTA" != "None" ] && [ -n "$SPOT_QUOTA" ] && [ "$SPOT_QUOTA" -lt "$NEED" ]; then
        SPOT_CAPPED=true
    fi

    USED=0
    if [ -n "$USAGE_EXPR" ]; then
        USED=$(aws ec2 describe-instances --region "$R" --filters Name=instance-state-name,Values=pending,running \
            --query "Reservations[].Instances[?${USAGE_EXPR}].[CpuOptions.CoreCount,CpuOptions.ThreadsPerCore]" \
            --output text 2>/dev/null | awk '{ s += $1 * $2 } END { print s + 0 }')
    fi
    if [ "$QUOTA" = "None" ] || [ -z "$QUOTA" ]; then
        QUOTA_OK="unknown"
    elif [ $((QUOTA - USED)) -ge "$NEED" ]; then
        QUOTA_OK="ok ${USED}/${QUOTA}"
    else
        QUOTA_OK="SHORT ${USED}/${QUOTA}"
        QUOTA_HINTS+="  aws service-quotas request-service-quota-increase --region $R --service-code ec2 --quota-code $QCODE --desired-value $((USED + NEED))"$'\n'
    fi

    # Spot placement score (1-10) per AZ: a capacity *signal* for the pool, not
    # an On-Demand guarantee. Needs ec2:GetSpotPlacementScores; skipped if denied.
    ZONEMAP=$(aws ec2 describe-availability-zones --region "$R" --query 'AvailabilityZones[].[ZoneId,ZoneName]' --output text 2>/dev/null || true)
    SCORES=$(aws ec2 get-spot-placement-scores --region "$R" --instance-types "$TYPE" --target-capacity "$COUNT" \
        --single-availability-zone --region-names "$R" \
        --query 'SpotPlacementScores[].[AvailabilityZoneId,Score]' --output text 2>/dev/null || true)

    # Does the existing cluster (if any) have a private subnet in each AZ?
    # Same check deploy_node_group.sh's resolve_gpu_az() enforces before
    # creating a node group -- shared here via _lib.sh so the two can't drift.
    CLUSTER_AZS=""
    SUBNETS=$(aws eks describe-cluster --region "$R" --name "$CLUSTER_NAME" \
        --query 'cluster.resourcesVpcConfig.subnetIds' --output text 2>/dev/null || true)
    if [ -n "$SUBNETS" ] && [ "$SUBNETS" != "None" ]; then
        CLUSTER_AZS=$(get_cluster_private_azs "$R")
    fi

    for AZ in $AZS; do
        ZID=$(echo "$ZONEMAP" | awk -v z="$AZ" '$2 == z { print $1 }')
        SPOT=$(echo "$SCORES" | awk -v id="$ZID" '$1 == id { print $2 }')
        SPOT=${SPOT:--}
        $SPOT_CAPPED && [ "$SPOT" != "-" ] && SPOT="${SPOT}*"
        if [ -z "$SUBNETS" ] || [ "$SUBNETS" = "None" ]; then
            SUBNET="no cluster"
        elif echo "$CLUSTER_AZS" | grep -qx "$AZ"; then
            SUBNET="yes"
        else
            SUBNET="NO"
        fi

        RESULT="-"
        if $PROBE && [[ "$QUOTA_OK" == ok* ]]; then
            OUT=$(aws ec2 create-capacity-reservation --region "$R" --instance-type "$TYPE" \
                --instance-platform Linux/UNIX --availability-zone "$AZ" --instance-count "$COUNT" \
                --instance-match-criteria targeted --end-date-type unlimited \
                --query 'CapacityReservation.CapacityReservationId' --output text 2>&1) && {
                PROBE_IDS+="$R:$OUT "
                aws ec2 cancel-capacity-reservation --region "$R" --capacity-reservation-id "$OUT" >/dev/null 2>&1 \
                    && PROBE_IDS="${PROBE_IDS%"$R:$OUT "}"
                RESULT="AVAILABLE"
            } || {
                case "$OUT" in
                    *InsufficientInstanceCapacity*) RESULT="no capacity" ;;
                    *InstanceLimitExceeded*|*VcpuLimitExceeded*|*ReservationCapacityExceeded*) RESULT="quota" ;;
                    *) RESULT="error: $(echo "$OUT" | head -1 | cut -c1-60)" ;;
                esac
            }
        elif $PROBE; then
            RESULT="skipped (quota)"
        fi
        $PROBE && echo "    ${AZ}: ${RESULT}" >&2
        ROWS+="$R|$AZ|$QUOTA_OK|$SPOT|$SUBNET|$RESULT"$'\n'
    done
done

print_section "Results: ${COUNT} x ${TYPE}  (${VCPU:-?} vCPU each, EFA: ${EFA:-?}, need $(( ${VCPU:-0} * COUNT )) vCPU)"
printf "  %-15s %-12s %-18s %-6s %-14s %s\n" "REGION" "AZ" "QUOTA used/limit" "SPOT" "CLUSTER SUBNET" "PROBE"
printf "  %-15s %-12s %-18s %-6s %-14s %s\n" "------" "--" "----------------" "----" "--------------" "-----"
echo -n "$ROWS" | while IFS='|' read -r r az q sp sn pr; do
    printf "  %-15s %-12s %-18s %-6s %-14s %s\n" "$r" "$az" "$q" "$sp" "$sn" "$pr"
done
echo
echo "  QUOTA: '${QUOTA_NAME}' On-Demand vCPUs. SPOT: EC2 Spot placement score 1-10"
echo "  (a capacity signal, not an On-Demand guarantee). PROBE: --probe result."
echo -n "$ROWS" | grep -q '\*|' && echo "  * Spot quota ('${SPOT_QUOTA_NAME}') is below the vCPUs needed, which caps the score; ignore it there."

# --- recommendation: probe-confirmed first, then quota ok + usable subnet, by spot score
BEST=$(echo -n "$ROWS" | awk -F'|' '
    $3 ~ /^ok/ && $6 !~ /no capacity|quota|error/ {
        sp = $4; sub(/\*$/, "", sp)
        rank = ($6 == "AVAILABLE" ? 2000 : 0) + ($5 == "yes" ? 100 : ($5 == "no cluster" ? 50 : 0)) + (sp ~ /^[0-9]+$/ ? sp : 0)
        print rank "|" $1 "|" $2 "|" $5 "|" $6
    }' | sort -t'|' -k1,1nr | head -1)

# Every usable placement, best first: what deploy_all.sh falls back through.
# A probed run only lists AZs the probe confirmed; AZs where an existing
# cluster has no private subnet are never listed.
if [ -n "$EMIT_FILE" ]; then
    echo -n "$ROWS" | awk -F'|' -v probed="$($PROBE && echo 1 || echo 0)" '
        $3 ~ /^ok/ && $5 != "NO" && $6 !~ /no capacity|quota|error/ && (probed == 0 || $6 == "AVAILABLE") {
            sp = $4; sub(/\*$/, "", sp)
            rank = ($6 == "AVAILABLE" ? 2000 : 0) + ($5 == "yes" ? 100 : ($5 == "no cluster" ? 50 : 0)) + (sp ~ /^[0-9]+$/ ? sp : 0)
            print rank "|" $1 " " $2
        }' | sort -t'|' -k1,1nr | cut -d'|' -f2 > "$EMIT_FILE"
fi

print_section "Recommendation"
if [ -n "$BEST" ]; then
    IFS='|' read -r _ BR BAZ BSUB BPROBE <<< "$BEST"
    [ "$BPROBE" = "AVAILABLE" ] && VERDICT="capacity confirmed by probe" || VERDICT="quota ok; capacity not confirmed (re-run with --probe for certainty)"
    print_success "${BR} / ${BAZ}: ${VERDICT}"
    if [ "$BSUB" = "NO" ]; then
        print_warning "Cluster '$CLUSTER_NAME' in $BR has no private subnet in $BAZ; the node group can't use it without recreating the cluster."
    fi
    echo
    echo "  export REGION=${BR} GPU_AZ=${BAZ}"
    if [ "$BSUB" = "no cluster" ]; then
        echo "  bash deploy_cluster.sh       # no '$CLUSTER_NAME' cluster in ${BR} yet"
    fi
    echo "  bash deploy_node_group.sh"
else
    print_error "No Region/AZ has quota headroom (and capacity, if probed) for ${COUNT} x ${TYPE}."
fi

if [ -n "$QUOTA_HINTS" ]; then
    echo
    print_warning "Quota too low where marked SHORT. To request an increase (approval can take a while):"
    echo -n "$QUOTA_HINTS"
fi

# A node group stuck in CREATING blocks a retry in another AZ: deploy_node_group.sh
# won't create a duplicate. Point at it so the user knows to delete it first.
for R in $REGIONS; do
    NG=$(aws eks describe-nodegroup --region "$R" --cluster-name "$CLUSTER_NAME" --nodegroup-name "$GPU_NODEGROUP_NAME" \
        --query 'nodegroup.[status,subnets[0]]' --output text 2>/dev/null || true)
    [ -z "$NG" ] && continue
    read -r NG_STATUS NG_SUBNET <<< "$NG"
    [ "$NG_STATUS" = "ACTIVE" ] && continue
    NG_AZ=$(aws ec2 describe-subnets --region "$R" --subnet-ids "$NG_SUBNET" --query 'Subnets[0].AvailabilityZone' --output text 2>/dev/null || echo "?")
    echo
    print_warning "Node group '$GPU_NODEGROUP_NAME' in $R is $NG_STATUS (AZ $NG_AZ). Delete it before retrying elsewhere: bash delete_node_group.sh"
done

print_elapsed

# --- offer to deploy right now, using the capacity this run just found ---
if ! $NO_OFFER && [ -n "$BEST" ] && [ "$BSUB" != "NO" ]; then
    echo
    read -p "Capacity is available in ${BR} / ${BAZ}. Build the cluster, GPU node group, and install the device plugins + KubeRay there now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        export REGION="$BR"
        export GPU_AZ="$BAZ"
        # This run already asked once; deploy_cluster.sh/deploy_node_group.sh
        # would otherwise ask again for the exact same decision.
        export ASSUME_YES=1

        print_section "Deploying: cluster -> GPU node group -> device plugins -> KubeRay"
        bash "$SCRIPT_DIR/deploy_cluster.sh"
        bash "$SCRIPT_DIR/deploy_node_group.sh"
        bash "$SCRIPT_DIR/install_gpu_plugins.sh"
        bash "$SCRIPT_DIR/install_kuberay.sh"

        print_success "Cluster, GPU node group, device plugins, and KubeRay are all up in ${REGION} / ${GPU_AZ}."

        # Stops at infrastructure: the training job is a separate, re-runnable
        # step, not part of building the cluster.
        echo
        echo "Next, from scripts/ (REGION is not saved by this script, so keep it exported):"
        echo "  export REGION=${REGION}"
        echo "  bash deploy_ray_train_job.sh    # applies the RayCluster and runs the training job"
    else
        echo "Skipped. Re-run manually when ready:"
        echo "  export REGION=${BR} GPU_AZ=${BAZ}"
        if [ "$BSUB" = "no cluster" ]; then
            echo "  bash deploy_cluster.sh       # no '$CLUSTER_NAME' cluster in ${BR} yet"
        fi
        echo "  bash deploy_node_group.sh && bash install_gpu_plugins.sh && bash install_kuberay.sh"
    fi
elif ! $NO_OFFER && [ -n "$BEST" ]; then
    print_warning "Not offering to auto-deploy: cluster '$CLUSTER_NAME' in $BR has no private subnet in $BAZ (see above) -- recreate the cluster or pick a different AZ before building the GPU node group there."
fi

[ -n "$BEST" ]
