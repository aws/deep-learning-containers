# telemetry.sh
#!/bin/bash
if [ -f /usr/local/bin/deep_learning_container.py ] && [[ -z "${OPT_OUT_TRACKING}" || "${OPT_OUT_TRACKING,,}" != "true" ]]; then
    (
        python /usr/local/bin/deep_learning_container.py \
            --framework "huggingface_vllm_omni" \
            --framework-version "0.20.0" \
            --container-type "general" \
            &>/dev/null &
    )
fi

