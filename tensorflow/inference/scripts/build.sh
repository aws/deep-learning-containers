#!/bin/bash
#
# Build the docker images.

set -euo pipefail

source scripts/shared.sh

parse_std_args "$@"

get_ei_executable

echo "pulling previous image for layer cache... "
$(aws ecr get-login --no-include-email --registry-id $aws_account) &>/dev/null || echo 'warning: ecr login failed'
docker pull $aws_account.dkr.ecr.$aws_region.amazonaws.com/$repository:$full_version-$device &>/dev/null || echo 'warning: pull failed'
docker logout https://$aws_account.dkr.ecr.$aws_region.amazonaws.com &>/dev/null

echo "building image... "
cp -r docker/build_artifacts/* docker/$short_version/
docker build \
    --cache-from $aws_account.dkr.ecr.$aws_region.amazonaws.com/$repository:$full_version-$device \
    --build-arg TFS_VERSION=$full_version \
    --build-arg TFS_SHORT_VERSION=$short_version \
    -f docker/$short_version/Dockerfile.$arch \
    -t $repository:$full_version-$device \
    -t $repository:$short_version-$device \
    docker/$short_version/

remove_ei_executable
