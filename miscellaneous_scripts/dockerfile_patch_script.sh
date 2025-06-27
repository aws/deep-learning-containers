#!/bin/bash
set -e

LATEST_RELEASED_IMAGE_SHA=$1
LATEST_RELEASED_IMAGE_URI=$2

PATCHING_INFO_PATH=/opt/aws/dlc/patching-info

# If patch-details-archive is not present, create it for the first time and add first_image_sha.txt
if [ ! -d $PATCHING_INFO_PATH/patch-details-archive ] ; then \
    mkdir $PATCHING_INFO_PATH/patch-details-archive && \
    echo $LATEST_RELEASED_IMAGE_SHA >> $PATCHING_INFO_PATH/patch-details-archive/first_image_sha.txt ; \
fi

# We use > instead of >> since we want to override the contents of the previous file
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

# Language patching
chmod +x $PATCHING_INFO_PATH/patch-details/install_script_language.sh && \
$PATCHING_INFO_PATH/patch-details/install_script_language.sh


##### Temporary Fixes #####

# For TF 2.18 training sm gpu, replace entrypoint
if [[ $LATEST_RELEASED_IMAGE_URI =~ ^763104351884\.dkr\.ecr\.us-west-2\.amazonaws\.com/tensorflow-training:2\.18\.0-gpu(.+)sagemaker ]]; then
    mv /tmp/new-tf-entrypoint /usr/local/bin/dockerd-entrypoint.py
    chmod +x /usr/local/bin/dockerd-entrypoint.py
fi

# For PT 2.4, 2.5 and 2.6 inference, install openssh-client to make mpi4py working
if [[ $LATEST_RELEASED_IMAGE_URI =~ ^763104351884\.dkr\.ecr\.us-west-2\.amazonaws\.com/pytorch-inference:2\.[4-6]\.[0-9]+-gpu ]]; then
    apt update && apt install -y --no-install-recommends openssh-client openssh-server && echo "Installed openssh-client openssh-server"
fi

# For PT 2.6, 2.7, rerun license file
if [[ $LATEST_RELEASED_IMAGE_URI =~ ^763104351884\.dkr\.ecr\.us-west-2\.amazonaws\.com/pytorch-(.+):2\.6 ]]; then
    curl -o /license.txt https://aws-dlc-licenses.s3.amazonaws.com/pytorch-2.6/license.txt
elif [[ $LATEST_RELEASED_IMAGE_URI =~ ^763104351884\.dkr\.ecr\.us-west-2\.amazonaws\.com/pytorch-(.+):2\.7 ]]; then
    curl -o /license.txt https://aws-dlc-licenses.s3.amazonaws.com/pytorch-2.7/license.txt
fi

# Upgrade sagemaker-training
if [[ $LATEST_RELEASED_IMAGE_URI =~ ^763104351884\.dkr\.ecr\.us-west-2\.amazonaws\.com/pytorch-training:2\.[4-6](.+)sagemaker ]]; then
    pip install -U "sagemaker-training>4.7.4" "protobuf>=4.25.8,<6"
fi

# For PT inference gpu sagemaker images, replace start_cuda_compat.sh
if [[ $LATEST_RELEASED_IMAGE_URI =~ ^763104351884\.dkr\.ecr\.us-west-2\.amazonaws\.com/pytorch-inference:2\.[4-6]\.[0-9]+-gpu(.+)sagemaker ]]; then
    mv /tmp/new_pytorch_inference_start_cuda_compat /usr/local/bin/start_cuda_compat.sh
    chmod +x /usr/local/bin/start_cuda_compat.sh

# For PT training gpu sagemaker images, add dynamic cuda compat mounting script to entrypoint
if [[ $LATEST_RELEASED_IMAGE_URI =~ ^763104351884\.dkr\.ecr\.us-west-2\.amazonaws\.com/pytorch-training:2\.[4-6]\.[0-9]+-gpu(.+)sagemaker ]]; then
    mv /tmp/new_start_with_right_hostname /usr/local/bin/start_with_right_hostname.sh
    mv /tmp/new_pytorch_training_start_cuda_compat /usr/local/bin/start_cuda_compat.sh
    chmod +x /usr/local/bin/start_with_right_hostname.sh
    chmod +x /usr/local/bin/start_cuda_compat.sh
fi

# For all GPU images, remove cuobjdump and nvdisasm
if [[ $LATEST_RELEASED_IMAGE_URI =~ ^763104351884\.dkr\.ecr\.us-west-2\.amazonaws\.com/(pytorch|tensorflow)(.+)gpu(.+) ]]; then
    rm -rf /usr/local/cuda/bin/cuobjdump*
    rm -rf /usr/local/cuda/bin/nvdisasm*
fi

###########################


## Update GPG key in case Nginx exists
VARIABLE=$(apt-key list 2>&1  |  { grep -c nginx || true; }) && \
if [ $VARIABLE != 0 ]; then \
    echo "Nginx exists, thus upgrade" && \
    curl https://nginx.org/keys/nginx_signing.key | gpg --dearmor | tee /usr/share/keyrings/nginx-archive-keyring.gpg >/dev/null && \
    apt-key add /usr/share/keyrings/nginx-archive-keyring.gpg;
fi

# OS patching
chmod +x $PATCHING_INFO_PATH/patch-details/install_script_os.sh && \
$PATCHING_INFO_PATH/patch-details/install_script_os.sh

# Derive history and package diff data
python /opt/aws/dlc/miscellaneous_scripts/derive_history.py
python /opt/aws/dlc/miscellaneous_scripts/extract_apt_patch_data.py --save-result-path $PATCHING_INFO_PATH/patch-details/os_summary.json --mode_type modify

set -e

HOME_DIR=/root \
    && rm -rf ${HOME_DIR}/oss_compliance* \
    && curl -o ${HOME_DIR}/oss_compliance.zip https://aws-dlinfra-utilities.s3.amazonaws.com/oss_compliance.zip \
    && unzip ${HOME_DIR}/oss_compliance.zip -d ${HOME_DIR}/ \
    && cp ${HOME_DIR}/oss_compliance/test/testOSSCompliance /usr/local/bin/testOSSCompliance \
    && chmod +x /usr/local/bin/testOSSCompliance \
    && chmod +x ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh \
    && ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh ${HOME_DIR} python \
    && rm -rf ${HOME_DIR}/oss_compliance* || exit

# Clean up
pip cache purge
apt-get clean
rm -rf /var/lib/apt/lists/*
rm -rf /tmp/*
rm -rf /opt/aws/dlc/miscellaneous_scripts
