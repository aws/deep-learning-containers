#!/bin/bash

set -ex

function install_python {
    PYTHON_VERSION=$1
    PYTHON_SHORT_VERSION=${PYTHON_VERSION%.*}
    # install python
    cd /tmp/
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
    tar xzf Python-${PYTHON_VERSION}.tgz
    cd Python-${PYTHON_VERSION}
    ./configure --enable-optimizations --with-lto --with-computed-gotos --with-system-ffi
    make -j "$(nproc)"
    make altinstall
    cd ..
    rm -rf Python-${PYTHON_VERSION}
    rm Python-${PYTHON_VERSION}.tgz
    ln -s /usr/local/bin/python${PYTHON_SHORT_VERSION} /usr/local/bin/python
    ln -s /usr/local/bin/python${PYTHON_SHORT_VERSION} /usr/local/bin/python3
    # This installation generate a .python_history file in the root directory leads sanity check to fail
    rm -f /root/.python_history 

    # this will add pip systemlink to pip${PYTHON_MAJOR_VERSION}
    python -m pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org
    python -m pip install --no-cache-dir awscli boto3 requests setuptools>=70.0.0
}

# idiomatic parameter and option handling in sh
while test $# -gt 0
do
    case "$1" in
    [0-9].[0-9]*.[0-9]*) install_python $1; 
        ;;
    *) echo "bad argument $1"; exit 1
        ;;
    esac
    shift
done
