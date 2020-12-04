#!/usr/bin/env bash
# run this the first time you clone the repo to populate the efficiendet model
# code

HERE=$(pwd)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODEL_DIR=${DIR}/model
CODE_DIR=${MODEL_DIR}/code

# clone the repo, and save the current commit sha to a file that will be
# included in the artifacts we build
cd "${CODE_DIR}" || exit
git clone https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch.git
cd Yet-Another-EfficientDet-Pytorch
YAEDP_CURRENT_SHA=$(git rev-parse HEAD)
cd "${CODE_DIR}" || exit
echo "${YAEDP_CURRENT_SHA}" > yaefd_current_sha
cd "${HERE}" || exit

exit $?