#!/bin/bash
#
# Publish images to your ECR account.

set -euo pipefail

source scripts/shared.sh

parse_std_args "$@"

$(aws ecr get-login --no-include-email --registry-id $aws_account)
docker tag $repository:$full_version-$device $aws_account.dkr.ecr.$aws_region.amazonaws.com/$repository:$full_version-$device
docker tag $repository:$full_version-$device $aws_account.dkr.ecr.$aws_region.amazonaws.com/$repository:$short_version-$device
docker push $aws_account.dkr.ecr.$aws_region.amazonaws.com/$repository:$full_version-$device
docker push $aws_account.dkr.ecr.$aws_region.amazonaws.com/$repository:$short_version-$device
docker logout https://$aws_account.dkr.ecr.$aws_region.amazonaws.com
