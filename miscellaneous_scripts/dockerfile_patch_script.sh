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


# For PT 2.2 and 2.3 training, install mpi4py from pip to remedy https://github.com/aws/deep-learning-containers/issues/4090
# Explicitly pinning these framework versions as they have the same mpi4py requirements in core packages
if [[ $LATEST_RELEASED_IMAGE_URI =~ ^763104351884\.dkr\.ecr\.us-west-2\.amazonaws\.com/pytorch-training:2\.[2-3]\.[0-9]+-cpu ]]; then
    conda uninstall mpi4py && pip install "mpi4py>=3.1.4,<3.2" && echo "Installed mpi4py from pip"
fi

# For PT 2.3 inference and training, install pyarrow from pip to remedy https://nvd.nist.gov/vuln/detail/CVE-2024-52338
if [[ $LATEST_RELEASED_IMAGE_URI =~ ^763104351884\.dkr\.ecr\.us-west-2\.amazonaws\.com/pytorch-inference:2\.3\.[0-9]+-* ]] || \
   [[ $LATEST_RELEASED_IMAGE_URI =~ ^763104351884\.dkr\.ecr\.us-west-2\.amazonaws\.com/pytorch-training:2\.3\.[0-9]+-* ]]; then
    pip uninstall -y pyarrow && pip install "pyarrow>=17.0.0" && echo "Installed pyarrow from pip"
fi

# Install packages and derive history and package diff data
chmod +x $PATCHING_INFO_PATH/patch-details/install_script_language.sh && \
$PATCHING_INFO_PATH/patch-details/install_script_language.sh

if [ $LATEST_RELEASED_IMAGE_URI == "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu121-ubuntu20.04-sagemaker" ]; then
    SMP_URL=https://smppy.s3.amazonaws.com/pytorch/cu121/smprof-0.3.334-cp310-cp310-linux_x86_64.whl
    pip install --no-cache-dir -U ${SMP_URL}
    echo "Installed SMP";
fi

if [ $LATEST_RELEASED_IMAGE_URI == "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker" ]; then
    SMP_URL=https://smppy.s3.amazonaws.com/pytorch/cu118/smprof-0.3.334-cp310-cp310-linux_x86_64.whl
    pip install --no-cache-dir -U ${SMP_URL}
    echo "Installed SMP";
fi

# Upgrade sagemaker-training package to latest
if pip show sagemaker-training; then
    pip install "sagemaker-training>4.7.4" --upgrade
fi

# For PT inference sagemaker images, replace torchserve-entrypoint.py with the latest one
# replace start_cuda_compat.sh if it's a gpu image
if [[ $LATEST_RELEASED_IMAGE_URI =~ ^763104351884\.dkr\.ecr\.us-west-2\.amazonaws\.com/pytorch-inference(.+)gpu(.+)sagemaker ]]; then
    mv /tmp/new-torchserve-entrypoint /usr/local/bin/dockerd-entrypoint.py
    mv /tmp/new_pytorch_inference_start_cuda_compat /usr/local/bin/start_cuda_compat.sh
    chmod +x /usr/local/bin/dockerd-entrypoint.py
    chmod +x /usr/local/bin/start_cuda_compat.sh
elif [[ $LATEST_RELEASED_IMAGE_URI =~ ^763104351884\.dkr\.ecr\.us-west-2\.amazonaws\.com/pytorch-inference(.+)sagemaker ]]; then
    mv /tmp/new-torchserve-entrypoint /usr/local/bin/dockerd-entrypoint.py
    chmod +x /usr/local/bin/dockerd-entrypoint.py
fi

# For PT training gpu sagemaker images, add dynamic cuda compat mounting script to entrypoint
if [[ $LATEST_RELEASED_IMAGE_URI =~ ^763104351884\.dkr\.ecr\.us-west-2\.amazonaws\.com/pytorch-training(.+)gpu(.+)sagemaker ]]; then
    mv /tmp/new_start_with_right_hostname /usr/local/bin/start_with_right_hostname.sh
    mv /tmp/new_pytorch_training_start_cuda_compat /usr/local/bin/start_cuda_compat.sh
    chmod +x /usr/local/bin/start_with_right_hostname.sh
    chmod +x /usr/local/bin/start_cuda_compat.sh
fi

pip cache purge

## Update GPG key in case Nginx exists
VARIABLE=$(apt-key list 2>&1  |  { grep -c nginx || true; }) && \
if [ $VARIABLE != 0 ]; then \
    echo "Nginx exists, thus upgrade" && \
    curl https://nginx.org/keys/nginx_signing.key | gpg --dearmor | tee /usr/share/keyrings/nginx-archive-keyring.gpg >/dev/null && \
    apt-key add /usr/share/keyrings/nginx-archive-keyring.gpg;
fi

chmod +x $PATCHING_INFO_PATH/patch-details/install_script_os.sh && \
$PATCHING_INFO_PATH/patch-details/install_script_os.sh

rm -rf /var/lib/apt/lists/* && \
  apt-get clean

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

rm -rf /tmp/* && rm -rf /opt/aws/dlc/miscellaneous_scripts
