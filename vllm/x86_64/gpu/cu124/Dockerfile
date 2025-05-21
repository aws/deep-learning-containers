ARG PYTHON="python3"
ARG EFA_VERSION="1.40.0"
FROM docker.io/vllm/vllm-openai:v0.8.5 as final

COPY install_efa.sh install_efa.sh
COPY deep_learning_container.py /usr/local/bin/deep_learning_container.py
RUN chmod +x /usr/local/bin/deep_learning_container.py && \
    # Install EFA
    bash install_efa.sh ${EFA_VERSION} && \
    rm install_efa.sh && \
    # OSS compliance
    apt-get update && \
    apt-get install -y --allow-change-held-packages --no-install-recommends unzip && \
    HOME_DIR=/root && \
    curl -o ${HOME_DIR}/oss_compliance.zip https://aws-dlinfra-utilities.s3.amazonaws.com/oss_compliance.zip && \
    unzip ${HOME_DIR}/oss_compliance.zip -d ${HOME_DIR}/ && \
    cp ${HOME_DIR}/oss_compliance/test/testOSSCompliance /usr/local/bin/testOSSCompliance && \
    chmod +x /usr/local/bin/testOSSCompliance && \
    chmod +x ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh && \
    ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh ${HOME_DIR} ${PYTHON} && \
    rm -rf ${HOME_DIR}/oss_compliance* && \
    rm -rf /tmp/tmp* && \
    rm -rf /root/.cache | true

ENV LD_LIBRARY_PATH="/usr/local/lib:/opt/amazon/ofi-nccl/lib:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    PATH="/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/usr/local/cuda/bin:${PATH}"