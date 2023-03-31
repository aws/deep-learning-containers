#!/bin/bash

PYTHON_VERSION=$(python -c 'import sys; print(sys.version_info[0])' | tr -d "'")
if [ "$PYTHON_VERSION" -eq 2 ]
then
  exit 0
fi
HOME_DIR=/test/benchmark
BIN_DIR=${HOME_DIR}/bin
LOG_DIR=${HOME_DIR}/logs

mkdir -p ${HOME_DIR}
mkdir -p ${BIN_DIR}
mkdir -p ${LOG_DIR}

python -c "import sys;
assert sys.version_info < (3, 7);"
PYTHON_VERSION_CHECK=`echo $?`

set -e
cd $BIN_DIR
# tmp fix for numpy deprecations, PR raised: https://github.com/HewlettPackard/dlcookbook-dlbs/pull/19
git clone --quiet --single-branch --depth 1 --branch v2.0.0 https://github.com/pytorch/pytorch.git pytorch

cd pytorch

pip install -U numpy
# pip install deepspeed==0.8.2
pip install gitpython
pip install tabulate==0.9.0

# install torchdata and torchtext before installing torchbench
git clone --branch v0.6.0 https://github.com/pytorch/data.git 
cd data
pip install .

# cd ..
# git clone --branch v0.15.1 https://github.com/pytorch/text torchtext
# cd torchtext
# git submodule update --init --recursive
# FORCE_CUDA=1 python setup.py clean install

# cd ../..
# git clone --quiet --single-branch --depth 1 https://github.com/pytorch/benchmark.git
# cd benchmark
# python install.py

cd ../pytorch
python benchmarks/dynamo/runner.py --suites=timm_models --training --dtypes=float32 --compilers=inductor --output-dir=timm_logs --extra-args='--output-directory=/opt/ml/output/data'

exit 0