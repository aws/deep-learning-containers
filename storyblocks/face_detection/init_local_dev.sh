#!/usr/bin/env bash
# run this the first time you clone this repo to populate the DSFD/retinaface
# model code

HERE=$(pwd)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODEL_DIR=${DIR}/model
CODE_DIR=${MODEL_DIR}/code

cd "${CODE_DIR}" || exit
git clone https://github.com/hukkelas/DSFD-Pytorch-Inference.git
cd DSFD-Pytorch-Inference

# note: this is the last commit before they started requesting pytorch>=1.6,
# so don't update the repo unless you intend to use the 1.6 container
git checkout 36fda4eaf

cd "${HERE}" || exit

exit $?