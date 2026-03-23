#!/bin/bash

set -ex

function install_python {
    PYTHON_VERSION=$1
    PYTHON_SHORT_VERSION=${PYTHON_VERSION%.*}

    # install python from source
    cd /tmp/
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
    tar xzf Python-${PYTHON_VERSION}.tgz
    cd Python-${PYTHON_VERSION}
    ./configure --enable-optimizations --with-lto --with-computed-gotos --with-system-ffi \
      CFLAGS="-fstack-protector-strong -D_FORTIFY_SOURCE=2" \
      LDFLAGS="-Wl,-z,relro,-z,now"
    make -j "$(nproc)"
    make altinstall
    cd ..
    rm -rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tgz

    ln -sf /usr/local/bin/python${PYTHON_SHORT_VERSION} /usr/local/bin/python
    ln -sf /usr/local/bin/python${PYTHON_SHORT_VERSION} /usr/local/bin/python3

    # clean up history file that causes sanity check failures
    rm -f /root/.python_history

    # install uv and base packages from pinned requirements
    python -m pip install --no-cache-dir --upgrade pip
    python -m pip install --no-cache-dir uv
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
