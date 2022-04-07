#!/usr/bin/env bash
# re-run this code any time you want to update local arrays -- also required to
# publish artifacts

ENV=${1:-dev}
SRC_CLASS=${2:-video}
TGT_CLASS=${3:-audio}

HERE=$(pwd)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODEL_DIR=${DIR}/model
CODE_DIR=${MODEL_DIR}/code
ARRAY_SUB_DIR=${SRC_CLASS}-${TGT_CLASS}-arrays
LOCAL_ARRAY_DIR=${CODE_DIR}/${ARRAY_SUB_DIR}

# download the items using aws
mkdir -p ${LOCAL_ARRAY_DIR}
rm -f ${LOCAL_ARRAY_DIR}/*.npy
aws s3 cp --recursive s3://videoblocks-ml/models/topic-tran/storyblocks/${ENV}/${ARRAY_SUB_DIR} ${LOCAL_ARRAY_DIR}
cd "${HERE}" || exit

exit $?
