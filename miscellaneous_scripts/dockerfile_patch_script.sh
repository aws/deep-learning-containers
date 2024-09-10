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

if [ $LATEST_RELEASED_IMAGE_URI == "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker" ]; then
    # Install EFA
    PREV_DIR=$(pwd)
    EFA_VERSION=1.34.0
    mkdir /tmp/efa \
    && cd /tmp/efa \
    && curl -O https://s3-us-west-2.amazonaws.com/aws-efa-installer/aws-efa-installer-${EFA_VERSION}.tar.gz \
    && tar -xf aws-efa-installer-${EFA_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && apt-get update \
    && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf /tmp/efa \
    && rm -rf /tmp/aws-efa-installer-${EFA_VERSION}.tar.gz \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && cd $PREV_DIR || exit

    conda remove --yes --force aws-ofi-nccl
    apt-get update && apt install -y automake libhwloc-dev
    # Install aws-ofi-nccl plugin
    LD_LIBRARY_PATH="${OPEN_MPI_PATH}/lib/:${EFA_PATH}/lib/:${LD_LIBRARY_PATH}"
    AWS_OFI_NCCL_VERSION=1.11.0
    mkdir /tmp/aws-ofi-nccl \
    && cd /tmp/aws-ofi-nccl \
    && wget https://github.com/aws/aws-ofi-nccl/releases/download/v${AWS_OFI_NCCL_VERSION}-aws/aws-ofi-nccl-${AWS_OFI_NCCL_VERSION}-aws.tar.gz \
    && tar -xf aws-ofi-nccl-${AWS_OFI_NCCL_VERSION}-aws.tar.gz \
    && cd aws-ofi-nccl-${AWS_OFI_NCCL_VERSION}-aws \
    && ./autogen.sh \
    && ./configure --prefix=/opt/aws-ofi-nccl \
                --with-mpi=${OPEN_MPI_PATH} \
                --with-libfabric=${EFA_PATH} \
                --with-cuda=${CUDA_HOME} \
                --disable-tests \
    && make \
    && make install \
    && rm -rf /tmp/aws-ofi-nccl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && cd $PREV_DIR || exit
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
