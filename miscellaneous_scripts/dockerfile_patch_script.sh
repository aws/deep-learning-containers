#!/bin/bash
set -e

RELEASED_IMAGE_SHA=$1

# If patch-details-archive is not present, create it for the first time and add first_image_sha.txt
if !(( $([ -d /opt/aws/dlc/patch-details-archive ]) )) ; then \
    mkdir /opt/aws/dlc/patch-details-archive && \
    echo $RELEASED_IMAGE_SHA >> /opt/aws/dlc/patch-details-archive/first_image_sha.txt ; \
fi

# If patch-details is present, move it to patch-details-archive and add image_sha to the folder
if [ -d /opt/aws/dlc/patch-details ] ; then \
    existing_file_count=$(ls -l /opt/aws/dlc/patch-details-archive | wc -l) && \
    reduce_count_value=1 && \
    patch_count=$((existing_file_count-reduce_count_value)) && \
    mv /opt/aws/dlc/patch-details /opt/aws/dlc/patch-details-archive/patch-details-$patch_count && \
    echo $RELEASED_IMAGE_SHA >> /opt/aws/dlc/patch-details-archive/patch-details-$patch_count/image_sha.txt ; \
fi

# Rename the patch-details-latest folder to patch-details
mv /opt/aws/dlc/patch-details-latest /opt/aws/dlc/patch-details

# Install packages and derive history and package diff data
chmod +x /opt/aws/dlc/patch-details/install_script_language.sh && \
/opt/aws/dlc/patch-details/install_script_language.sh && \
chmod +x /opt/aws/dlc/patch-details/install_script_os.sh && \
/opt/aws/dlc/patch-details/install_script_os.sh && \
python /opt/aws/dlc/miscellaneous_scripts/derive_history.py && \
python /opt/aws/dlc/miscellaneous_scripts/extract_apt_patch_data.py --save-result-path /opt/aws/dlc/patch-details/os_summary.json --mode_type modify

HOME_DIR=/root \
    && curl -o ${HOME_DIR}/oss_compliance.zip https://aws-dlinfra-utilities.s3.amazonaws.com/oss_compliance.zip \
    && unzip ${HOME_DIR}/oss_compliance.zip -d ${HOME_DIR}/ \
    && cp ${HOME_DIR}/oss_compliance/test/testOSSCompliance /usr/local/bin/testOSSCompliance \
    && chmod +x /usr/local/bin/testOSSCompliance \
    && chmod +x ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh \
    && ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh ${HOME_DIR} ${PYTHON} \
    && rm -rf ${HOME_DIR}/oss_compliance*

rm -rf /tmp/* && rm -rf /opt/aws/dlc/miscellaneous_scripts
