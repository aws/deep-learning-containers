#!/usr/bin/env bash

HERE=$(pwd)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR=${DIR}/model
CODE_DIR=${MODEL_DIR}/code

# timestamp for the model we're building
cd ${CODE_DIR}
date -u +"%Y%m%dT%H%M%S" > model_archive_timestamp

# create the archive
cd "${MODEL_DIR}" || exit
if [ -f "model.tar.gz" ] ; then
    rm "model.tar.gz"
fi
tar --exclude "*.git" --exclude="*.png" --exclude="*.jpg" -czvf model.tar.gz ./*
mv model.tar.gz ../

cd "${HERE}" || exit

exit $?
