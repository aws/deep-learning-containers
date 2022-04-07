#!/usr/bin/env bash
# run this the first time you clone the repo to populate the array files

ENV=${1:-dev}
SRC_CLASS=${2:-video}
TGT_CLASS=${3:-audio}

HERE=$(pwd)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODEL_DIR=${DIR}/model
CODE_DIR=${MODEL_DIR}/code

cd ${DIR} || exit
/usr/bin/env bash download_translation_arrays.sh ${ENV} ${SRC_CLASS} ${TGT_CLASS}
cd ${HERE} || exit

exit $?
