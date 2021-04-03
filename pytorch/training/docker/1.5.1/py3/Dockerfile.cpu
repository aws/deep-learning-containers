FROM ubuntu:16.04

LABEL maintainer="Amazon AI"
LABEL dlc_major_version="2"

# Add arguments to achieve the version, python and url
ARG PYTHON_VERSION=3.6.13
ARG OPEN_MPI_VERSION=4.0.1

# The smdebug pipeline relies for following format to perform string replace and trigger DLC pipeline for validating
# the nightly builds. Therefore, while updating the smdebug version, please ensure that the format is not disturbed.
ARG SMDEBUG_VERSION=0.9.4

# Python won’t try to write .pyc or .pyo files on the import of source modules
# Force stdin, stdout and stderr to be totally unbuffered. Good for logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/conda/lib"
ENV PATH=/opt/conda/bin:$PATH
ENV SAGEMAKER_TRAINING_MODULE=sagemaker_pytorch_container.training:main
ENV DGLBACKEND=pytorch

WORKDIR /

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    curl \
    emacs \
    git \
    wget \
    unzip \
    vim \
    build-essential \
    jq \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    zlib1g-dev \
    software-properties-common \
 && rm -rf /var/lib/apt/lists/*

RUN wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-$OPEN_MPI_VERSION.tar.gz \
 && gunzip -c openmpi-$OPEN_MPI_VERSION.tar.gz | tar xf - \
 && cd openmpi-$OPEN_MPI_VERSION \
 && ./configure --prefix=/home/.openmpi \
 && make all install \
 && cd .. \
 && rm openmpi-$OPEN_MPI_VERSION.tar.gz \
 && rm -rf openmpi-$OPEN_MPI_VERSION

# The ENV variables declared below are changed in the previous section
# Grouping these ENV variables in the first section causes
# ompi_info to fail. This is only observed in CPU containers
ENV PATH="$PATH:/home/.openmpi/bin"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/.openmpi/lib/"
RUN ompi_info --parsable --all | grep mpi_built_with_cuda_support:value

# Copy workaround script for incorrect hostname
COPY changehostname.c /
COPY start_with_right_hostname.sh /usr/local/bin/start_with_right_hostname.sh

RUN curl -L -o ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p /opt/conda \
 && rm ~/miniconda.sh \
 && /opt/conda/bin/conda update conda \
 && /opt/conda/bin/conda install -c conda-forge \
    python=$PYTHON_VERSION \
 && /opt/conda/bin/conda install -y -c anaconda \
    # conda 4.9.2 requires ruamel_yaml to be installed. Currently pinned at latest.
    ruamel_yaml==0.15.87 \
    numpy==1.16.4 \
    ipython==7.10.1 \
    mkl==2019.4 \
    mkl-include==2019.4 \
    cython==0.29.12 \
    typing==3.6.4 \
    "pyopenssl>=17.5.0" \
 && conda install -c conda-forge \
    opencv==4.0.1 \
 && conda install -y \
    scikit-learn==0.21.2 \
    pandas==0.25.0 \
    h5py==2.9.0 \
    requests==2.22.0 \
 && conda install -c dglteam -y dgl==0.4.3 \
 && /opt/conda/bin/conda clean -ya \
 && conda clean -ya

# The following section uninstalls torch and torchvision before installing the
# custom versions from an S3 bucket. This will need to be removed in the future
RUN pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org \
 && ln -s /opt/conda/bin/pip /usr/local/bin/pip3 \
 && pip install --no-cache-dir -U \
    # install PyYAML>=5.4,<5.5 to avoid conflict with latest awscli
    "PyYAML>=5.4,<5.5" \
    "awscli<2" \
    fastai==1.0.59 \
    scipy==1.2.2 \
    smdebug==${SMDEBUG_VERSION} \
    "sagemaker>=2,<3" \
    sagemaker-experiments==0.* \
    "sagemaker-pytorch-training<3" \
 && pip install --no-cache-dir -U https://pytorch-aws.s3-us-west-2.amazonaws.com/pytorch-1.5.1/py3/cpu/torch-1.5.1-cp36-cp36m-manylinux1_x86_64.whl \
 && pip uninstall -y torchvision \
 && pip install --no-deps --no-cache-dir -U \
    https://torchvision-build.s3.amazonaws.com/1.5.1/cpu/torchvision-0.6.1-cp36-cp36m-linux_x86_64.whl

RUN chmod +x /usr/local/bin/start_with_right_hostname.sh

ADD https://raw.githubusercontent.com/aws/deep-learning-containers/master/src/deep_learning_container.py /usr/local/bin/deep_learning_container.py

RUN chmod +x /usr/local/bin/deep_learning_container.py

RUN HOME_DIR=/root \
 && curl -o ${HOME_DIR}/oss_compliance.zip https://aws-dlinfra-utilities.s3.amazonaws.com/oss_compliance.zip \
 && unzip ${HOME_DIR}/oss_compliance.zip -d ${HOME_DIR}/ \
 && cp ${HOME_DIR}/oss_compliance/test/testOSSCompliance /usr/local/bin/testOSSCompliance \
 && chmod +x /usr/local/bin/testOSSCompliance \
 && chmod +x ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh \
 && ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh ${HOME_DIR} ${PYTHON} \
 && rm -rf ${HOME_DIR}/oss_compliance*

RUN curl -o /license.txt https://aws-dlc-licenses.s3.amazonaws.com/pytorch-1.5.1/license.txt

# Packages installed to resolve safety vulnerabilities
RUN pip install \
    psutil==5.6.7 \
    Pillow==7.1.0 \
    # cryptography required by pyopenssl and paramiko, <3.3.2 has safety vulnerabilities
    "cryptography>=3.3.2"

# Starts framework
ENTRYPOINT ["bash", "-m", "start_with_right_hostname.sh"]
CMD ["/bin/bash"]
