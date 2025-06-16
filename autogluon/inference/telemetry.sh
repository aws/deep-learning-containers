# telemetry.sh
#!/bin/bash
if [ -f /usr/local/bin/deep_learning_container.py ] && [[ -z "${OPT_OUT_TRACKING}" || "${OPT_OUT_TRACKING,,}" != "true" ]]; then
    (
        python /usr/local/bin/deep_learning_container.py \
            --framework "autogluon" \
            --framework-version "1.3.1" \
            --container-type "inference" \
            &>/dev/null &
    )
fi

