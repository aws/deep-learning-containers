# A stand-in DLC image for testing telemetry changes: Amazon Linux 2023 with the
# telemetry hook installed the way the framework Dockerfiles install it (see
# docker/base/cu130/Dockerfile). telemetry.pipeline.yml builds it from the checkout
# on every run, so the test never depends on a released image.
FROM public.ecr.aws/amazonlinux/amazonlinux:2023

# What deep_learning_container.py imports, and the `python` that bash_telemetry.sh runs
RUN dnf install -y \
  python-unversioned-command \
  python3-botocore \
  python3-packaging \
  python3-requests \
  && dnf clean all

# GPU images link /usr/local/cuda to the toolkit, so the telemetry tag takes its GPU branch
RUN mkdir /usr/local/cuda-13.0 \
  && ln -s /usr/local/cuda-13.0 /usr/local/cuda

COPY ./scripts/docker/telemetry/deep_learning_container.py /usr/local/bin/deep_learning_container.py
COPY ./scripts/docker/telemetry/bash_telemetry.sh.template /tmp/bash_telemetry.sh.template

ARG FRAMEWORK
ARG FRAMEWORK_VERSION
ARG CONTAINER_TYPE
ENV DLC_CONTAINER_TYPE=${CONTAINER_TYPE}

RUN chmod +x /usr/local/bin/deep_learning_container.py \
  && sed -e "s/{{FRAMEWORK}}/${FRAMEWORK}/g" \
    -e "s/{{FRAMEWORK_VERSION}}/${FRAMEWORK_VERSION}/g" \
    -e "s/{{CONTAINER_TYPE}}/${CONTAINER_TYPE}/g" \
    /tmp/bash_telemetry.sh.template >/usr/local/bin/bash_telemetry.sh \
  && chmod +x /usr/local/bin/bash_telemetry.sh \
  && rm /tmp/bash_telemetry.sh.template \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/etc/bashrc \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/root/.bashrc

CMD ["/bin/bash"]