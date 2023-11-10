#!/bin/bash
#/ Usage: ./patch_hosts.sh

set -ex
source ./helper.sh

# Parse parameters from build_param.json config file
EKS_CLUSTERS=($(jq -r '.eks_clusters[]' build_param.json))
CONTEXTS=($(jq -r '.contexts[]' build_param.json))
EKS_VERSION=$(jq -r '.eks_version' build_param.json)

# Upgrade hosts to latest ami
upgrade_nodegroup
