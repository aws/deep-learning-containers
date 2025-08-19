#!/bin/bash

set -ex

HOME_DIR=/root 
curl -o ${HOME_DIR}/oss_compliance.zip https://aws-dlinfra-utilities.s3.amazonaws.com/oss_compliance.zip 
unzip ${HOME_DIR}/oss_compliance.zip -d ${HOME_DIR}/ 
cp ${HOME_DIR}/oss_compliance/test/testOSSCompliance /usr/local/bin/testOSSCompliance 
chmod +x /usr/local/bin/testOSSCompliance 
chmod +x ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh 
${HOME_DIR}/oss_compliance/generate_oss_compliance.sh ${HOME_DIR} ${PYTHON} 
rm -rf ${HOME_DIR}/oss_compliance* 
rm -rf /tmp/tmp*