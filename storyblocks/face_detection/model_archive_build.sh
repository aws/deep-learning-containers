#!/usr/bin/env bash

HERE=$(pwd)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR=${DIR}/model
CODE_DIR=${MODEL_DIR}/code
DSFDPI_DIR="${CODE_DIR}/DSFD-Pytorch-Inference"

# make sure that the dsfdpi_current_sha file is up to date
cd "${DSFDPI_DIR}" || exit
DSFDPI_CURRENT_SHA=$(git rev-parse HEAD)
cd "${CODE_DIR}" || exit
echo "${DSFDPI_CURRENT_SHA}" > yaefd_current_sha

# add a timestamp while we are here
date -u +"%Y%m%dT%H%M%S" > model_archive_timestamp

# create the archive
cd "${MODEL_DIR}" || exit
if [ -f "model.tar.gz" ] ; then
    rm "model.tar.gz"
fi
tar \
  --exclude "*.git" --exclude="*.png" --exclude="*.jpg" --exclude="*.pyc" \
  --exclude="*face_detection.egg*" --exclude="*__pycache__*" \
  -czvf model.tar.gz ./*
mv model.tar.gz ../

cd "${HERE}" || exit

exit $?
