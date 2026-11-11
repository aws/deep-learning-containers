#!/bin/bash
# install_gpu_plugins.sh - Ensure both device plugins are present: the NVIDIA
# device plugin (advertises nvidia.com/gpu) and the AWS EFA device plugin
# (advertises vpc.amazonaws.com/efa, injects /dev/infiniband). The accelerated
# AMI ships the GPU driver and the EFA kernel module/rdma-core, but a plugin
# must advertise each to kubelet. Recent eksctl already bundles both into an
# EFA-enabled node group; this script detects that and only Helm-installs
# whatever is genuinely missing, so it is safe on either kind of cluster.
#
# Usage:
#   bash install_gpu_plugins.sh            # Install both
#   bash install_gpu_plugins.sh cleanup    # Uninstall both
#
# Prerequisites: GPU node group running (deploy_node_group.sh). helm is
# auto-installed (via brew or the official script) if missing.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
source "$SCRIPT_DIR/_lib.sh"

SECONDS=0

check_prerequisites() {
    ensure_helm
    check_kubectl_prerequisites
    print_success "Prerequisites satisfied (kubectl, helm)"
}

# Count GPU nodes already advertising a given allocatable resource. Recent
# eksctl bundles both device plugins into an EFA-enabled node group, so the
# resources are often advertised before this script runs -- in which case
# there is nothing to install.
nodes_advertising() {
    kubectl get nodes -l role=gpu-worker \
        -o jsonpath="{range .items[*]}{.status.allocatable.$1}{\"\n\"}{end}" 2>/dev/null \
        | grep -cE '^[1-9]' || true
}

plugin_present() {
    kubectl get ds -n kube-system -o name 2>/dev/null | grep -qE "/${1}(-daemonset)?\$"
}

cleanup() {
    print_section "Uninstalling GPU Device Plugins"
    helm uninstall nvidia-device-plugin -n kube-system 2>/dev/null || true
    helm uninstall aws-efa-k8s-device-plugin -n kube-system 2>/dev/null || true
    print_success "Device plugins uninstalled (any eksctl-bundled plugins are left in place)"
}

if [ "${1:-install}" = "cleanup" ]; then
    check_prerequisites
    cleanup
    exit 0
fi

echo -e "${BLUE}"
echo "=================================================="
echo "  Install GPU Device Plugins"
echo "=================================================="
echo -e "${NC}"
echo "  NVIDIA device plugin: $NVIDIA_DEVICE_PLUGIN_VERSION"
echo "  AWS EFA device plugin: $EFA_DEVICE_PLUGIN_VERSION"
echo

check_prerequisites

helm repo add nvdp https://nvidia.github.io/k8s-device-plugin >/dev/null 2>&1 || true
helm repo add eks https://aws.github.io/eks-charts >/dev/null 2>&1 || true
helm repo update >/dev/null 2>&1 || true

print_section "Installing NVIDIA device plugin"
if [ "$(nodes_advertising 'nvidia\.com/gpu')" -ge "$GPU_NODE_COUNT" ]; then
    print_success "nvidia.com/gpu already advertised on all $GPU_NODE_COUNT node(s) (eksctl-bundled or prior run) -- skipping"
elif plugin_present nvidia-device-plugin; then
    print_success "NVIDIA device plugin DaemonSet already present (eksctl-bundled) -- skipping"
elif helm status nvidia-device-plugin -n kube-system &>/dev/null; then
    print_success "NVIDIA device plugin Helm release already present"
else
    # nodeSelector scopes it to GPU nodes; the chart's own affinity on
    # nvidia.com/gpu.present (set on the node group) is the second gate.
    helm install nvidia-device-plugin nvdp/nvidia-device-plugin \
        --version "$NVIDIA_DEVICE_PLUGIN_VERSION" \
        --namespace kube-system \
        --set nodeSelector.role=gpu-worker
    print_success "NVIDIA device plugin installed"
fi

print_section "Installing AWS EFA device plugin"
if [ "$(nodes_advertising 'vpc\.amazonaws\.com/efa')" -ge "$GPU_NODE_COUNT" ]; then
    print_success "vpc.amazonaws.com/efa already advertised on all $GPU_NODE_COUNT node(s) (eksctl-bundled or prior run) -- skipping"
elif plugin_present aws-efa-k8s-device-plugin; then
    print_success "AWS EFA device plugin DaemonSet already present (eksctl-bundled) -- skipping"
elif helm status aws-efa-k8s-device-plugin -n kube-system &>/dev/null; then
    print_success "AWS EFA device plugin Helm release already present"
else
    # The chart's own affinity also allowlists node.kubernetes.io/instance-type;
    # v0.5.32 is confirmed to include g6.12xlarge (see env.sh).
    helm install aws-efa-k8s-device-plugin eks/aws-efa-k8s-device-plugin \
        --version "$EFA_DEVICE_PLUGIN_VERSION" \
        --namespace kube-system \
        --set nodeSelector.role=gpu-worker
    print_success "AWS EFA device plugin installed"
fi

# The real success criterion is that kubelet advertises both resources, no
# matter which plugin (eksctl-bundled or Helm) provides them -- so verify the
# allocatable counts on the nodes rather than a specifically-named DaemonSet.
print_section "Waiting for GPU + EFA to be Allocatable on Both Nodes"
for _ in $(seq 1 36); do
    GPU_NODES=$(nodes_advertising 'nvidia\.com/gpu')
    EFA_NODES=$(nodes_advertising 'vpc\.amazonaws\.com/efa')
    [ "$GPU_NODES" -ge "$GPU_NODE_COUNT" ] && [ "$EFA_NODES" -ge "$GPU_NODE_COUNT" ] && break
    sleep 5
done

print_section "Allocatable Resources Per GPU Node"
kubectl get nodes -l role=gpu-worker \
    -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu,EFA:.status.allocatable.vpc\.amazonaws\.com/efa'

GPU_NODES=$(nodes_advertising 'nvidia\.com/gpu')
EFA_NODES=$(nodes_advertising 'vpc\.amazonaws\.com/efa')
if [ "$GPU_NODES" -lt "$GPU_NODE_COUNT" ] || [ "$EFA_NODES" -lt "$GPU_NODE_COUNT" ]; then
    print_error "GPU/EFA not allocatable on all $GPU_NODE_COUNT node(s) (gpu_nodes=$GPU_NODES efa_nodes=$EFA_NODES). Check 'kubectl get pods -n kube-system -o wide' and node labels/affinity."
    exit 1
fi
print_success "GPU and EFA allocatable on all $GPU_NODE_COUNT GPU node(s)"
print_elapsed
