ARG EFA_VERSION="1.40.0"
FROM docker.io/vllm/vllm-openai:v0.8.5 as final

COPY install_efa.sh install_efa.sh
RUN bash install_efa.sh ${EFA_VERSION} && rm install_efa.sh
ENV LD_LIBRARY_PATH="/usr/local/lib:/opt/amazon/ofi-nccl/lib:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    PATH="/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/usr/local/cuda/bin:${PATH}"