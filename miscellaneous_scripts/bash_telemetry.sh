# telemetry.sh
#!/bin/bash
if [ -f /usr/local/bin/deep_learning_container.py ] && [ -z "${OPT_OUT_TRACKING}" -o "${OPT_OUT_TRACKING,,}" != "true" ]; then
    (
        python /usr/local/bin/deep_learning_container.py \
            --framework "${FRAMEWORK}" \
            --framework-version "${FRAMEWORK_VERSION}" \
            --container-type "${CONTAINER_TYPE}" \
            &>/dev/null &
    )
fi