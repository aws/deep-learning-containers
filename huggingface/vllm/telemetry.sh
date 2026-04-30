# telemetry.sh
#!/bin/bash
if [ -f /usr/local/bin/deep_learning_container.py ] && [[ -z "${OPT_OUT_TRACKING}" || "${OPT_OUT_TRACKING,,}" != "true" ]]; then
    (
        python /usr/local/bin/deep_learning_container.py \
            --framework "huggingface_vllm" \
            --framework-version "0.19.1" \
            --container-type "general" \
            &>/dev/null &
    )
fi
