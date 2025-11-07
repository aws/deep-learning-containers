#!/bin/bash

set -ex

function install_oss_compliance {
    HOME_DIR="/root"
    PYTHON=$1

    if [ -z "$PYTHON" ]; then
        echo "Python version not specified. Using default Python."
        PYTHON="python3"
    fi
    curl -o ${HOME_DIR}/oss_compliance.zip https://aws-dlinfra-utilities.s3.amazonaws.com/oss_compliance.zip
    ${PYTHON} -c "import zipfile, os; zipfile.ZipFile('/root/oss_compliance.zip').extractall('/root/'); os.remove('/root/oss_compliance.zip')"
    cp ${HOME_DIR}/oss_compliance/test/testOSSCompliance /usr/local/bin/testOSSCompliance
    chmod +x /usr/local/bin/testOSSCompliance
    chmod +x ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh
    ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh ${HOME_DIR} ${PYTHON}
    rm -rf ${HOME_DIR}/oss_compliance*
    rm -rf /tmp/tmp*
    # Removing the cache as it is needed for security verification
    rm -rf /root/.cache | true
}

while test $# -gt 0
do
    case "$1" in
        python*) install_oss_compliance $1;
            ;;
        *) echo "bad argument $1"; exit 1
            ;;
    esac
    shift
done