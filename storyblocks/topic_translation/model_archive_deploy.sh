#!/usr/bin/env bash

ENV=${1:-dev}
SRC_CLASS=${2:-video}
TGT_CLASS=${3:-audio}

HERE=$(pwd)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_TAR=${DIR}/model.tar.gz
MODEL_DIR=${DIR}/model
CODE_DIR=${MODEL_DIR}/code

# build the path
BUCKET="videoblocks-ml"
MODEL_ARCH_TS=$(cat "${CODE_DIR}"/model_archive_timestamp)
MODEL_S3_PATH="s3://${BUCKET}/models/topic-tran/storyblocks/${ENV}/${SRC_CLASS}-${TGT_CLASS}-${MODEL_ARCH_TS}/model.tar.gz"
aws s3 cp "${MODEL_TAR}" "${MODEL_S3_PATH}"

echo "successfully published archive to:"
echo "${MODEL_S3_PATH}"

cd "${HERE}" || exit

exit $?
