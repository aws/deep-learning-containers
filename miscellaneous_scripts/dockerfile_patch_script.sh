#!/bin/bash
set -e

LATEST_RELEASED_IMAGE_SHA=$1
PATCHING_INFO_PATH=/opt/aws/dlc/patching-info

# If patch-details-archive is not present, create it for the first time and add first_image_sha.txt
if [ ! -d $PATCHING_INFO_PATH/patch-details-archive ] ; then \
    mkdir $PATCHING_INFO_PATH/patch-details-archive && \
    echo $LATEST_RELEASED_IMAGE_SHA >> $PATCHING_INFO_PATH/patch-details-archive/first_image_sha.txt ; \
fi

## We use > instead of >> since we want to override the contents of the previous file.
echo $LATEST_RELEASED_IMAGE_SHA > $PATCHING_INFO_PATH/patch-details-archive/last_released_image_sha.txt

# If patch-details is present, move it to patch-details-archive and add image_sha to the folder
if [ -d $PATCHING_INFO_PATH/patch-details ] ; then \
    existing_file_count=$(ls -ld $PATCHING_INFO_PATH/patch-details-archive/patch-details-*/ | wc -l) && \
    add_count_value=1 && \
    patch_count=$((existing_file_count+add_count_value)) && \
    mv $PATCHING_INFO_PATH/patch-details $PATCHING_INFO_PATH/patch-details-archive/patch-details-$patch_count && \
    echo $LATEST_RELEASED_IMAGE_SHA >> $PATCHING_INFO_PATH/patch-details-archive/patch-details-$patch_count/image_sha.txt ; \
fi

# Rename the patch-details-current folder to patch-details
mv $PATCHING_INFO_PATH/patch-details-current $PATCHING_INFO_PATH/patch-details

# Install packages and derive history and package diff data
chmod +x $PATCHING_INFO_PATH/patch-details/install_script_language.sh && \
$PATCHING_INFO_PATH/patch-details/install_script_language.sh

pip cache purge

chmod +x $PATCHING_INFO_PATH/patch-details/install_script_os.sh && \
$PATCHING_INFO_PATH/patch-details/install_script_os.sh

rm -rf /var/lib/apt/lists/* && \
  apt-get clean

python /opt/aws/dlc/miscellaneous_scripts/derive_history.py

python /opt/aws/dlc/miscellaneous_scripts/extract_apt_patch_data.py --save-result-path $PATCHING_INFO_PATH/patch-details/os_summary.json --mode_type modify

HOME_DIR=/root \
    && curl -o ${HOME_DIR}/oss_compliance.zip https://aws-dlinfra-utilities.s3.amazonaws.com/oss_compliance.zip \
    && unzip ${HOME_DIR}/oss_compliance.zip -d ${HOME_DIR}/ \
    && cp ${HOME_DIR}/oss_compliance/test/testOSSCompliance /usr/local/bin/testOSSCompliance \
    && chmod +x /usr/local/bin/testOSSCompliance \
    && chmod +x ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh \
    && ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh ${HOME_DIR} ${PYTHON} \
    && rm -rf ${HOME_DIR}/oss_compliance*

rm -rf /tmp/* && rm -rf /opt/aws/dlc/miscellaneous_scripts
