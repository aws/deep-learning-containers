#!/bin/bash
# install_gdrcopy.sh — Build GDRCopy userspace library from source
set -ex

GDRCOPY_VERSION="${1:?Usage: install_gdrcopy.sh <version>}"

dnf install -y make gcc git
cd /tmp
git clone --depth 1 --branch "v${GDRCOPY_VERSION}" https://github.com/NVIDIA/gdrcopy.git
cd gdrcopy
make PREFIX=/usr/local CUDA=/usr/local/cuda lib lib_install
rm -rf /tmp/gdrcopy
ldconfig
