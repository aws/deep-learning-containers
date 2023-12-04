#!/bin/bash
#/ Usage: ./patch_hosts.sh

set -ex
source ./helper.sh

# Parse parameters from build_param.json config file
EKS_CLUSTERS=($(jq -r '.eks_clusters[]' patch_hosts.json))
CONTEXTS=($(jq -r '.contexts[]' patch_hosts.json))
EKS_VERSION=$(jq -r '.eks_version' build_param.json)

# Upgrade hosts to latest ami
upgrade_nodegroup
